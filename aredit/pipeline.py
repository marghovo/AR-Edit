"""AR-Edit pipeline: chunk-by-chunk editing on the Reward-Forcing backbone.

For every chunk of `num_frame_per_block` latent frames:

  1. Source pass.   One forward on the clean source chunk at t=0 with the
                    source prompt on a freshly reset source KV cache. The
                    patched self-attention stores V_src of the eligible blocks;
                    the patched cross-attention records the saliency of the
                    removed / kept source words.
  2. Commit.        Denoising steps < inject_step from Gaussian noise with the
                    target prompt; V of blocks > v_inject_min_block is replaced
                    by V_src for the attention product.
  3. Anchor.        Denoising steps >= late_anchor_start: the edit mask is
                    built from the cross-attention of the edited words
                    (Mask construction resembles https://arxiv.org/pdf/2412.18597, https://github.com/TencentARC/DiTCtrl
                    and https://arxiv.org/pdf/2512.22118, https://isee-laboratory.github.io/ProEdit/); outside
                    the mask V is pulled back to V_src. On chunk 0 the last
                    step anchors blocks > cta_first_chunk_last_min_block with
                    the previous step's mask.
  4. Text boost.    log(text_scale) added to the cross-attention logits of the
                    target-only words at every step
                    (Text token boosting from https://arxiv.org/pdf/2507.01496,
                     https://github.com/wlaud1001/ReFlex).
  5. Context rerun. One forward at t=0 on the denoised chunk writes clean K, V
                    into the generation cache for the next chunk (follows Reward-Forcing).

Model loading, scheduler and the causal hyper-parameters come from the
Reward-Forcing base class `CausalInferencePipeline`.
"""

from __future__ import annotations
import torch
import logging
from demo_utils.memory import (
    get_cuda_free_memory_gb,
    gpu,
    move_model_to_device_with_memory_preservation,
)
from typing import List, Optional
from aredit.config import AREditConfig
from aredit.hook_state import HookState
from aredit.prompt_alignment import PromptAlignment
from aredit.edit_mask import EditMaskHook, MaskParams
from pipeline.causal_inference import CausalInferencePipeline
from aredit.attention_patches import install_attention_patches

LOGGER = logging.getLogger(__name__)

# Latent patch grid of Wan2.1 at 832x480 (patch size (1, 2, 2) over a 60x104 latent).
PATCH_HW = (30, 52)
KV_HEADS, KV_HEAD_DIM = 12, 128
TEXT_LEN = 512


class AREditPipeline(CausalInferencePipeline):
    """Streaming video editing with commit-then-anchor V sharing.

    Usage (latents are [B, T_lat, C, H, W] bfloat16, `num_frame_per_block`
    latent frames per chunk):

        pipe = AREditPipeline(backbone_config, AREditConfig(), device).to(torch.bfloat16)
        pipe.set_prompt_alignment(align_prompts(pipe.text_encoder.tokenizer, src, tar))
        z0 = pipe.prepare(x_src_chunk0, tar_prompts=[tar], src_prompts=[src])
        z1 = pipe.step(x_src_chunk1)
    """

    # Reset the source KV cache before every source pass and run it at
    # current_start=0: each chunk's V_src is computed in isolation.
    RESET_SRC_CACHE_PER_CHUNK = True

    def __init__(self, backbone_config, edit_config: AREditConfig, device,
                 generator=None, text_encoder=None, vae=None) -> None:
        super().__init__(backbone_config, device, generator=generator,
                         text_encoder=text_encoder, vae=vae)
        if edit_config.num_blocks != self.num_transformer_blocks:
            raise ValueError(
                f"edit_config.num_blocks={edit_config.num_blocks} but the backbone has "
                f"{self.num_transformer_blocks} blocks")
        self.cfg = edit_config
        self._device = device

        self.state = HookState(
            inject_min_block=self.cfg.v_inject_min_block,
            inject_alpha=self.cfg.v_inject_alpha,
            store_min_block=self.cfg.store_min_block,
            anchor_min_block_default=self.cfg.anchor_min_block,
            anchor_alpha=self.cfg.late_anchor_alpha,
            mask_block_max=self.cfg.cta_block_max,
            mask_sigma=self.cfg.cta_sigma,
            patch_hw=PATCH_HW,
        )
        install_attention_patches(self.generator.model, self.state)

        self._mask_hook: Optional[EditMaskHook] = None
        if self.cfg.late_anchor:
            self._mask_hook = EditMaskHook(
                self.state, MaskParams.from_config(self.cfg),
                current_chunk_start=lambda: self.current_start_frame)
            self.generator.model.blocks[self.cfg.cta_block_max].register_forward_hook(self._mask_hook)

        # True when the prompt alignment of the current video found edited words.
        self._anchor_enabled_for_video = False

        self.src_conditional_dict = None
        self.gen_conditional_dict = None
        self.src_kv_cache = None
        self.gen_kv_cache = None
        self.src_crossattn_cache = None
        self.gen_crossattn_cache = None
        self.current_start_frame = 0

        c = self.cfg
        LOGGER.info(
            "AR-Edit: commit steps [0,%d) blocks>%d alpha=%g | anchor=%s steps>=%d blocks>%d "
            "alpha=%g chunk0-last blocks>%s | mask blocks 1..%d combine=%s sigma=%g k=%g "
            "islands=top%d-tube rel=%g dilate=%d feather=%g | text boost x%g blocks>%d | edit_type=%s",
            c.inject_step, c.v_inject_min_block, c.v_inject_alpha,
            "on" if c.late_anchor else "off", c.late_anchor_start, c.anchor_min_block,
            c.late_anchor_alpha, c.cta_first_chunk_last_min_block, c.cta_block_max, c.cta_combine,
            c.cta_sigma, c.cta_thresh_k, c.cta_top_islands, c.cta_top_islands_rel, c.cta_dilate,
            c.cta_feather_sigma, c.text_scale, c.text_boost_min_block, c.edit_type)

    ### per-video wiring ###

    def set_prompt_alignment(self, alignment: PromptAlignment) -> None:
        """Wire the text boost and the edit-mask token sets for the next video."""
        if self.cfg.text_boost_enabled:
            self.state.set_text_boost(
                alignment.tar_alphas.to(device=self._device, dtype=torch.float32),
                scale=self.cfg.text_scale, min_block=self.cfg.text_boost_min_block)
        else:
            self.state.set_text_boost(None, scale=1.0, min_block=-1)
        self.state.set_edit_tokens(alignment.tar_gap_idxs, alignment.src_removed_idxs,
                                   alignment.src_kept_idxs)
        self._anchor_enabled_for_video = alignment.has_edit_tokens
        if self.cfg.late_anchor and not self._anchor_enabled_for_video:
            LOGGER.warning("The prompt alignment found no edited words; "
                           "anchoring is disabled for this video.")

    def enable_mask_logging(self) -> List[dict]:
        """Record every built edit mask (and its saliencies); returns the log list."""
        if self._mask_hook is None:
            raise RuntimeError("mask logging needs late_anchor=True")
        self._mask_hook.log = []
        return self._mask_hook.log

    ### caches ###

    def _init_kv_cache(self, batch_size: int, dtype, device) -> List[dict]:
        size = (self.local_attn_size * self.frame_seq_length
                if self.local_attn_size != -1 else 32760)
        return [{
            "k": torch.zeros([batch_size, size, KV_HEADS, KV_HEAD_DIM], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, size, KV_HEADS, KV_HEAD_DIM], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    def _init_crossattn_cache(self, batch_size: int, dtype, device) -> List[dict]:
        return [{
            "k": torch.zeros([batch_size, TEXT_LEN, KV_HEADS, KV_HEAD_DIM], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, TEXT_LEN, KV_HEADS, KV_HEAD_DIM], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]

    @staticmethod
    def _reset_kv_cache(kv_cache: List[dict], device) -> None:
        for block in kv_cache:
            block["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            block["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)

    @staticmethod
    def _reset_crossattn_cache(crossattn_cache: List[dict]) -> None:
        for block in crossattn_cache:
            block["is_init"] = False

    ### chunk loop ###

    def prepare(self, x_src: torch.Tensor, tar_prompts: List[str],
                src_prompts: Optional[List[str]] = None, low_memory: bool = False) -> torch.Tensor:
        """Encode both prompts, (re)initialise the caches and edit chunk 0."""
        batch_size, dtype, device = x_src.shape[0], x_src.dtype, x_src.device

        self.gen_conditional_dict = self.text_encoder(text_prompts=tar_prompts)
        self.src_conditional_dict = self.text_encoder(
            text_prompts=src_prompts if src_prompts is not None else [""])

        if low_memory:
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu,
                preserved_memory_gb=get_cuda_free_memory_gb(gpu) + 5)

        if self.src_kv_cache is None:
            self.src_kv_cache = self._init_kv_cache(batch_size, dtype, device)
            self.gen_kv_cache = self._init_kv_cache(batch_size, dtype, device)
            self.src_crossattn_cache = self._init_crossattn_cache(batch_size, dtype, device)
            self.gen_crossattn_cache = self._init_crossattn_cache(batch_size, dtype, device)
        else:
            self._reset_kv_cache(self.src_kv_cache, device)
            self._reset_kv_cache(self.gen_kv_cache, device)
            self._reset_crossattn_cache(self.src_crossattn_cache)
            self._reset_crossattn_cache(self.gen_crossattn_cache)

        self.current_start_frame = 0
        return self.step(x_src)

    def step(self, x_src: torch.Tensor) -> torch.Tensor:
        """Edit one chunk. x_src: [B, T_chunk, C, H, W] clean source latents."""
        if self.src_kv_cache is None:
            raise RuntimeError("step() called before prepare()")
        current_start = self.current_start_frame * self.frame_seq_length
        is_first_chunk = self.current_start_frame == 0

        self._source_pass(x_src, current_start, is_first_chunk)
        denoised = self._denoise(x_src, current_start, is_first_chunk)
        self.state.disarm()
        self._context_rerun(denoised, current_start)

        self.current_start_frame += x_src.shape[1]
        return denoised

    ### the three phases ###

    def _source_pass(self, x_src: torch.Tensor, current_start: int, is_first_chunk: bool) -> None:
        """Clean forward at t=0 with the source prompt: stores V_src and the source saliencies."""
        device = x_src.device
        if self.RESET_SRC_CACHE_PER_CHUNK:
            self._reset_kv_cache(self.src_kv_cache, device)
            src_current_start = 0
        else:
            src_current_start = current_start
        c0 = (self.cfg.cta_first_chunk_last_min_block
              if (self.cfg.cta_first_chunk_last_min_block is not None and is_first_chunk) else None)
        self.state.begin_source_pass(
            c0_store_min_block=c0,
            capture=self.cfg.late_anchor and self._anchor_enabled_for_video)
        self.generator(
            noisy_image_or_video=x_src,
            conditional_dict=self.src_conditional_dict,
            timestep=torch.zeros(x_src.shape[:2], device=device, dtype=torch.int64),
            kv_cache=self.src_kv_cache,
            crossattn_cache=self.src_crossattn_cache,
            current_start=src_current_start,
        )
        self.state.end_source_pass()

    def _denoise(self, x_src: torch.Tensor, current_start: int, is_first_chunk: bool) -> torch.Tensor:
        """Generation from noise with commit, anchor and text boost."""
        cfg, st = self.cfg, self.state
        batch_size, num_frames = x_src.shape[:2]
        device = x_src.device
        steps = self.denoising_step_list
        last = len(steps) - 1
        boost_steps = cfg.text_boost_steps if cfg.text_boost_steps is not None else len(steps)
        anchor_on = (cfg.late_anchor and self._anchor_enabled_for_video
                     and cfg.edit_type != "style")

        z = torch.randn_like(x_src)
        denoised = None
        for index, t in enumerate(steps):
            inject_active = index < cfg.inject_step
            anchor_active = anchor_on and index >= cfg.late_anchor_start
            capture_active = anchor_active
            # Chunk-0 exception: the last step anchors from a lower block and
            # keeps the mask built at the previous step (no rebuild).
            first_chunk_last = (cfg.cta_first_chunk_last_min_block is not None
                                and anchor_active and is_first_chunk and index == last)
            if first_chunk_last:
                capture_active = False
                st.anchor_min_block = cfg.cta_first_chunk_last_min_block
            else:
                st.anchor_min_block = cfg.anchor_min_block

            st.inject_active = inject_active
            st.anchor_active = anchor_active
            st.capture_active = capture_active

            if capture_active:
                st.begin_target_capture()
            else:
                st.capture_side = None
            if not anchor_active:
                st.anchor_mask = None

            st.mode = "inject" if (inject_active or anchor_active) else None
            st.denoise_iter = index
            st.text_boost_active = st.text_alphas is not None and index < boost_steps

            _, denoised = self.generator(
                noisy_image_or_video=z,
                conditional_dict=self.gen_conditional_dict,
                timestep=torch.ones([batch_size, num_frames], device=device, dtype=torch.int64) * t,
                kv_cache=self.gen_kv_cache,
                crossattn_cache=self.gen_crossattn_cache,
                current_start=current_start,
            )

            if index < last:
                z = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    steps[index + 1] * torch.ones([batch_size * num_frames], device=device, dtype=torch.long),
                ).unflatten(0, denoised.shape[:2])
        return denoised

    def _context_rerun(self, denoised: torch.Tensor, current_start: int) -> None:
        """Forward at t=0 on the denoised chunk: clean K, V of this chunk into the generation cache.
         Follows Reward-Forcing"""

        self.generator(
            noisy_image_or_video=denoised,
            conditional_dict=self.gen_conditional_dict,
            timestep=torch.ones(denoised.shape[:2], device=denoised.device, dtype=torch.int64)
            * self.args.context_noise,
            kv_cache=self.gen_kv_cache,
            crossattn_cache=self.gen_crossattn_cache,
            current_start=current_start,
        )

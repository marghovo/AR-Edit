"""Model loading, VAE encode / decode and the chunk editing loop."""

from __future__ import annotations

import torch
import logging
from einops import rearrange
from aredit.config import AREditConfig
from aredit.pipeline import AREditPipeline
from demo_utils.memory import DynamicSwapInstaller
from aredit.prompt_alignment import PromptAlignment, align_prompts

LOGGER = logging.getLogger(__name__)


class AREditRunner:
    """Edits whole videos: encode once, edit chunk by chunk, decode once.

    Note: While our current implementation encodes and decodes the video once; AR-Edit
    can be run with a streaming VAE (e.g. https://github.com/chenfengxu714/StreamDiffusionV2)
    and we leave its integration into our pipeline as a future work.

    """

    def __init__(self, backbone_config, edit_config: AREditConfig, device: torch.device) -> None:
        self.device = device
        self.pipeline = AREditPipeline(backbone_config, edit_config, device=device).to(dtype=torch.bfloat16)
        self.num_frame_per_block = self.pipeline.num_frame_per_block

    def load_weights(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.pipeline.generator.load_state_dict(state_dict)
        LOGGER.info("Reward-Forcing weights loaded from %s", checkpoint_path)

    def to_device(self, low_memory: bool) -> None:
        """Move the models to the GPU (text encoder is swapped dynamically when memory is tight)."""
        if low_memory:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=self.device)
        else:
            self.pipeline.text_encoder.to(device=self.device)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)

    def set_prompts(self, src_prompt: str, tar_prompt: str) -> PromptAlignment:
        """Align the prompts and wire text boost + edit-mask tokens into the pipeline."""
        alignment = align_prompts(self.pipeline.text_encoder.tokenizer, src_prompt, tar_prompt)
        self.pipeline.set_prompt_alignment(alignment)
        LOGGER.info("Prompt alignment: target-only=%s removed=%s kept=%s (gap tokens=%d)",
                    alignment.tar_gap_idxs, alignment.src_removed_idxs, alignment.src_kept_idxs,
                    alignment.num_gap_tokens)
        return alignment

    @torch.no_grad()
    def edit(self, input_video: torch.Tensor, src_prompt: str, tar_prompt: str,
             low_memory: bool = False) -> torch.Tensor:
        """input_video: [1, C, T, H, W] in [-1, 1]. Returns [T_out, H, W, C] float in [0, 1]."""
        self.pipeline.vae.model.clear_cache()
        latents = self.pipeline.vae.encode_to_latent(input_video).to(dtype=torch.bfloat16)
        n_lat = self.num_frame_per_block
        num_chunks = latents.shape[1] // n_lat
        if num_chunks < 1:
            raise ValueError(
                f"Input too short: {input_video.shape[2]} frames -> {latents.shape[1]} latent "
                f"frames < {n_lat} per chunk.")
        LOGGER.info("%d pixel frames -> %d latent frames -> %d chunks of %d latents",
                    input_video.shape[2], latents.shape[1], num_chunks, n_lat)

        outputs = [self.pipeline.prepare(
            x_src=latents[:, :n_lat], tar_prompts=[tar_prompt], src_prompts=[src_prompt],
            low_memory=low_memory)]
        for c in range(1, num_chunks):
            outputs.append(self.pipeline.step(latents[:, c * n_lat:(c + 1) * n_lat]))

        video = self.pipeline.vae.decode_to_pixel(torch.cat(outputs, dim=1), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return rearrange(video, "b t c h w -> b t h w c").cpu()[0]

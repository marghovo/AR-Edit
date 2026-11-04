"""Patched attention forwards of the Reward-Forcing / Wan2.1 transformer.

Two forwards are replaced on every transformer block:

* `aredit_self_attn_forward` replaces `CausalWanSelfAttention.forward`
  (wan/modules/causal_model.py). Additions: the V_src store
  (source pass) and `blend_current_chunk_v` (commit / anchor), marked
  ">>> AR-Edit".
* `aredit_cross_attn_forward` replaces `WanT2VCrossAttention.forward`
  (wan/modules/model.py). Additions: the text boost (ReFlex I2T-CA
  amplification, applied before the softmax, https://arxiv.org/pdf/2507.01496,
  https://github.com/wlaud1001/ReFlex) and the saliency capture that
  feeds the edit mask. When neither is active the original flash-attention
  path runs.

All communication goes through the shared `HookState`.
"""

from __future__ import annotations

import copy
import math
import types
import torch
import torch.nn.functional as F
from aredit.hook_state import HookState
from wan.modules.causal_model import causal_rope_apply
from wan.modules.attention import attention, flash_attention
from aredit.edit_mask import gaussian_smooth_2d, normalize_per_frame


STATE_ATTR = "_aredit_state"
BLOCK_ID_ATTR = "_aredit_block_id"



###  V blending (commit + anchor) on the current chunk's slot ###

def blend_current_chunk_v(state: HookState, block_id: int, v: torch.Tensor,
                          v_all: torch.Tensor) -> torch.Tensor:
    """Apply the commit and anchor hooks to the current chunk's slot of `v_all`.

    v:      [B, L_chunk, H, D]  this block's V of the current chunk (model's own)
    v_all:  [B, L_all,   H, D]  V tensor entering the attention product; the
                                current chunk occupies its trailing L_chunk rows.
    Returns `v_all` with that slot blended. The KV cache is never modified
    ("split cache": the cache keeps the model's own K, V).
    """
    if state.mode != "inject":
        return v_all
    slot_end = v_all.shape[1]
    slot_start = slot_end - v.shape[1]

    # Commit: alpha * V_src + (1 - alpha) * V_cur in blocks > inject_min_block.
    if state.inject_active and block_id > state.inject_min_block:
        v_src = state.v_store.get(block_id)
        if v_src is not None and v_src.shape == v.shape:
            v_src = v_src.to(dtype=v_all.dtype, device=v_all.device)
            alpha = state.inject_alpha
            v_cur = v_all[:, slot_start:slot_end]
            mixed = alpha * v_src + (1.0 - alpha) * v_cur
            v_all = torch.cat([v_all[:, :slot_start], mixed], dim=1)

    # Anchor: outside the edit mask pull V back to V_src in blocks > anchor_min_block.
    if state.anchor_active and block_id > state.anchor_min_block:
        v_src = state.v_store.get(block_id)
        m = state.anchor_mask                       # [B, L_chunk] float, 1 = edit (free)
        if (v_src is not None and v_src.shape == v.shape
                and m is not None and m.shape[1] == v.shape[1]):
            v_src = v_src.to(dtype=v_all.dtype, device=v_all.device)
            v_cur = v_all[:, slot_start:slot_end]
            m_b = m[:, :v.shape[1], None, None].to(dtype=v_all.dtype, device=v_all.device)
            a = float(state.anchor_alpha)
            anchored = a * v_src + (1.0 - a) * v_cur
            mixed = m_b * v_cur + (1.0 - m_b) * anchored
            v_all = torch.cat([v_all[:, :slot_start], mixed], dim=1)
    return v_all

###  Cross-attention: text boost + saliency capture ###

def aredit_cross_attn_forward(self, x, context, context_lens, crossattn_cache=None):
    """Patched `WanT2VCrossAttention.forward` (see module docstring)."""
    b, n, d = x.size(0), self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, -1, n, d)

    if crossattn_cache is not None:
        if not crossattn_cache["is_init"]:
            crossattn_cache["is_init"] = True
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
        else:
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]
    else:
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

    # >>> AR-Edit
    state: HookState = getattr(self, STATE_ATTR)
    block_id: int = getattr(self, BLOCK_ID_ATTR)
    boost_eligible = (state.text_boost_active and state.text_alphas is not None
                      and block_id > state.text_boost_min_block)
    capture_eligible = (state.capture_side is not None
                        and 1 <= block_id <= state.mask_block_max)

    if boost_eligible or capture_eligible:
        # Explicit attention so the softmax can be modified / observed.
        qt = q.permute(0, 2, 1, 3)                     # [B, H, Lq, D]
        kt = k.permute(0, 2, 1, 3)                     # [B, H, Lk, D]
        vt = v.permute(0, 2, 1, 3)                     # [B, H, Lk, D]
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * scale
        if context_lens is not None:
            Lk = scores.size(-1)
            positions = torch.arange(Lk, device=scores.device)
            pad_mask = positions.unsqueeze(0) >= context_lens.to(scores.device).unsqueeze(1)
            scores = scores.masked_fill(pad_mask.view(b, 1, 1, Lk), float("-inf"))

        if boost_eligible:
            # log(alpha_j + (1 - alpha_j) * scale) = log(scale) at target-only
            # words, 0 at aligned words. Rows still sum to one after softmax.
            Lk = scores.shape[-1]
            a = state.text_alphas[:Lk].to(device=scores.device, dtype=scores.dtype)
            mult = a + (1.0 - a) * float(state.text_scale)
            scores = scores + torch.log(mult.clamp_min(1e-12)).view(1, 1, 1, -1)

        attn = F.softmax(scores, dim=-1)               # [B, H, Lq, Lk]

        if capture_eligible:
            side = state.capture_side                  # "src" | "tar"
            ph, pw = state.patch_hw

            def block_saliency(cols: torch.Tensor) -> torch.Tensor:
                # mean over heads and selected text columns -> [B, Lq]; smoothed
                # and min-max normalised per frame so every block votes equally.
                sal = cols.mean(dim=(1, 3)).detach()
                B_, Lq_ = sal.shape
                maps = gaussian_smooth_2d(sal.float().view(B_, Lq_ // (ph * pw), ph, pw),
                                          float(state.mask_sigma))
                return normalize_per_frame(maps).reshape(B_, Lq_)

            idxs = state.src_removed_idxs if side == "src" else state.tar_gap_idxs
            if idxs:
                state.saliency[side].add(block_saliency(attn[:, :, :, idxs]))
            if side == "src" and state.src_kept_idxs:
                state.saliency["srckept"].add(block_saliency(attn[:, :, :, state.src_kept_idxs]))

        x_out = torch.matmul(attn.to(vt.dtype), vt)    # [B, H, Lq, D]
        x_out = x_out.permute(0, 2, 1, 3).contiguous() # [B, Lq, H, D]
    else:
        x_out = flash_attention(q, k, v, k_lens=context_lens)
    # <<< AR-Edit

    x_out = x_out.flatten(2)
    x_out = self.o(x_out)
    return x_out



###  Self-attention: V store + commit / anchor ###

def aredit_self_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
):
    """Patched `CausalWanSelfAttention.forward` (see module docstring)."""
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    if cache_start is None:
        cache_start = current_start

    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # >>> AR-Edit: V_src store (source pass)
    state: HookState = getattr(self, STATE_ATTR)
    block_id: int = getattr(self, BLOCK_ID_ATTR)
    if state.mode == "store":
        store_min = state.store_min_block
        if state.c0_store_min_block is not None:
            # Chunk 0 anchors a wider block range at its last step, so its
            # source pass must also store V_src of those extra blocks.
            store_min = min(store_min, state.c0_store_min_block)
        if block_id > store_min:
            state.v_store[block_id] = v.detach().clone()
    # <<< AR-Edit

    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    current_start_frame = current_start // frame_seqlen
    current_end = current_start + q.shape[1]
    total_sink_tokens = self.sink_size * frame_seqlen

    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = q.shape[1]

    evicted_tokens_exist = False
    evicted_k = None
    evicted_v = None

    if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - total_sink_tokens

        evicted_start = total_sink_tokens
        evicted_end = total_sink_tokens + num_evicted_tokens
        evicted_k = kv_cache["k"][:, evicted_start:evicted_end].clone()
        evicted_v = kv_cache["v"][:, evicted_start:evicted_end].clone()
        evicted_tokens_exist = True

        kv_cache["k"][:, total_sink_tokens:total_sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, total_sink_tokens + num_evicted_tokens:total_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, total_sink_tokens:total_sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, total_sink_tokens + num_evicted_tokens:total_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = k
        kv_cache["v"][:, local_start_index:local_end_index] = v
    else:
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = k
        kv_cache["v"][:, local_start_index:local_end_index] = v

    if evicted_tokens_exist and evicted_k is not None and evicted_k.shape[1] > 0:
        if evicted_k.shape[1] == total_sink_tokens:
            current_sink_k = kv_cache["k"][:, :total_sink_tokens]
            current_sink_v = kv_cache["v"][:, :total_sink_tokens]

            updated_sink_k, updated_sink_v = self.incremental_update(
                evicted_k, evicted_v,
                current_sink_k, current_sink_v,
            )

            kv_cache["k"][:, :total_sink_tokens] = updated_sink_k
            kv_cache["v"][:, :total_sink_tokens] = updated_sink_v
        else:
            min_len = min(evicted_k.shape[1], total_sink_tokens)
            kv_cache["k"][:, :min_len] = evicted_k[:, :min_len]
            kv_cache["v"][:, :min_len] = evicted_v[:, :min_len]

    kv_start_index = max(total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens)
    kv_end_index = local_end_index
    actual_kv_tokens = kv_end_index - kv_start_index

    k_segment = kv_cache["k"][:, kv_start_index:kv_end_index]
    v_segment = kv_cache["v"][:, kv_start_index:kv_end_index]

    if self.sink_size > 0:
        sink_k = kv_cache["k"][:, :total_sink_tokens]
        sink_v = kv_cache["v"][:, :total_sink_tokens]

        k_combined = torch.cat([sink_k, k_segment], dim=1)
        v_combined = torch.cat([sink_v, v_segment], dim=1)

        grid_sizes_kv = copy.deepcopy(grid_sizes)
        total_combined_tokens = total_sink_tokens + actual_kv_tokens
        total_frames = total_combined_tokens // frame_seqlen
        grid_sizes_kv[:, 0] = total_frames

        query_start_frame = (total_sink_tokens +
                             (local_end_index - max(total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens)) -
                             q.shape[1]) // frame_seqlen

        # >>> AR-Edit: commit / anchor on the attention operand only
        v_combined = blend_current_chunk_v(state, block_id, v, v_combined)
        # <<< AR-Edit

        x = attention(
            causal_rope_apply(q, grid_sizes, freqs, start_frame=query_start_frame).type_as(v),
            causal_rope_apply(k_combined, grid_sizes_kv, freqs, start_frame=0).type_as(v),
            v_combined,
        )
    else:
        grid_sizes_kv = copy.deepcopy(grid_sizes)
        grid_sizes_kv[:, 0] = actual_kv_tokens // frame_seqlen

        query_start_frame = (local_end_index - self.max_attention_size - q.shape[1]) // frame_seqlen

        # >>> AR-Edit: commit / anchor on the attention operand only
        v_segment = blend_current_chunk_v(state, block_id, v, v_segment)
        # <<< AR-Edit

        x = attention(
            causal_rope_apply(q, grid_sizes, freqs, start_frame=query_start_frame).type_as(v),
            causal_rope_apply(k_segment, grid_sizes_kv, freqs, start_frame=0).type_as(v),
            v_segment,
        )

    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    x = x.flatten(2)
    x = self.o(x)
    return x

###  Installation ###

def install_attention_patches(model, state: HookState) -> int:
    """Bind the patched attention forwards to every block of a `CausalWanModel`.

    Returns the number of patched blocks.
    """
    for i, block in enumerate(model.blocks):
        for module, forward in ((block.self_attn, aredit_self_attn_forward),
                                (block.cross_attn, aredit_cross_attn_forward)):
            setattr(module, BLOCK_ID_ATTR, i)
            setattr(module, STATE_ATTR, state)
            module.forward = types.MethodType(forward, module)
    return len(model.blocks)

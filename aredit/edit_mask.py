"""Edit-mask construction from cross-attention saliencies.

Pure tensor functions (no model, no state) plus `EditMaskHook`, the forward
hook that turns the saliencies accumulated in `HookState` into the anchor mask.

Background: building an edit region from the cross-attention of the edited
words follows Prompt-to-Prompt-style mask extraction as used for video DiTs in
DiTCtrl and ProEdit (https://arxiv.org/pdf/2412.18597,
https://github.com/TencentARC/DiTCtrl  and https://arxiv.org/pdf/2512.22118,
https://isee-laboratory.github.io/ProEdit/).
The specifics (block averaging, per-frame mean + k*std test, 3D "tube" islands,
dilation and feathering) are AR-Edit's.

"""

from __future__ import annotations

import torch
import logging
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
from aredit.hook_state import HookState

LOGGER = logging.getLogger(__name__)


###  Low-level spatial helpers ([B, T, H, W] maps) ###

def gaussian_smooth_2d(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable spatial Gaussian blur per frame. sigma <= 0 is a no-op."""
    if sigma <= 0:
        return maps
    B, T, H, W = maps.shape
    ks = int(6 * sigma + 1) | 1
    x = torch.arange(ks, device=maps.device, dtype=maps.dtype) - (ks // 2)
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, ks, 1)
    kw = g.view(1, 1, 1, ks)
    flat = maps.reshape(B * T, 1, H, W)
    flat = F.conv2d(flat, kh, padding=(ks // 2, 0))
    flat = F.conv2d(flat, kw, padding=(0, ks // 2))
    return flat.reshape(B, T, H, W)


def dilate_2d(m: torch.Tensor, iters: int) -> torch.Tensor:
    """3x3 max-pool dilation per frame, `iters` times."""
    B, T, H, W = m.shape
    flat = m.reshape(B * T, 1, H, W)
    for _ in range(int(iters)):
        flat = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    return flat.reshape(B, T, H, W)


def normalize_per_frame(maps: torch.Tensor) -> torch.Tensor:
    """Min-max normalisation of every frame to [0, 1]."""
    lo = maps.flatten(2).min(dim=-1).values[..., None, None]
    hi = maps.flatten(2).max(dim=-1).values[..., None, None]
    return (maps - lo) / (hi - lo + 1e-8)


def top_islands_tube(m: torch.Tensor, top_n: int, excess: Optional[torch.Tensor] = None,
                     rel: float = 0.0) -> torch.Tensor:
    """Keep only the strongest connected components of a binary mask.

    Components are found over the whole [T, H, W] block with a 3x3x3
    neighbourhood ("tubes" that also connect adjacent latent frames), so one
    keep/drop decision is taken per structure for the whole chunk.

    rel == 0: keep the `top_n` largest tubes by volume.
    rel  > 0: score every tube by (sum of its saliency margin above the
              threshold) / sqrt(volume) and keep every tube scoring at least
              rel * best (adaptive count; `top_n` is ignored).
    """
    B, T, H, W = m.shape
    keep_out = torch.zeros(B, T, H, W, device=m.device)
    for b in range(B):
        mb = (m[b] > 0.5).float().view(1, 1, T, H, W)
        base = torch.arange(1, T * H * W + 1, device=m.device,
                            dtype=torch.float32).view(1, 1, T, H, W)
        ids = base * mb
        for _ in range(T + H + W):
            new = F.max_pool3d(ids, 3, stride=1, padding=1) * mb
            if torch.equal(new, ids):
                break
            ids = new
        flat = ids.long().reshape(-1)
        vals = flat.unique()
        vals = vals[vals > 0]
        if vals.numel() == 0:
            continue
        sizes = torch.bincount(flat)[vals]
        if rel > 0 and excess is not None:
            exc_flat = excess[b].float().reshape(-1)
            mass = torch.bincount(flat, weights=exc_flat)[vals]
            score = mass / sizes.float().clamp_min(1.0).sqrt()
            top = vals[score >= rel * score.max()]
        else:
            top = vals[sizes.argsort(descending=True)[:top_n]]
        keep_out[b] = torch.isin(flat, top).float().view(T, H, W)
    return keep_out



###  Mask builder ###

@dataclass(frozen=True)
class MaskParams:
    """Parameters of `build_edit_mask` (see AREditConfig for the meaning)."""

    combine: str = "union"
    thresh_k: float = 0.5
    top_islands: int = 1
    top_islands_rel: float = 0.25
    dilate: int = 1
    feather_sigma: float = 0.7
    patch_h: int = 30
    patch_w: int = 52

    @classmethod
    def from_config(cls, cfg, patch_hw=(30, 52)) -> "MaskParams":
        """Build from an `AREditConfig`."""
        return cls(combine=cfg.cta_combine, thresh_k=cfg.cta_thresh_k,
                   top_islands=cfg.cta_top_islands, top_islands_rel=cfg.cta_top_islands_rel,
                   dilate=cfg.cta_dilate, feather_sigma=cfg.cta_feather_sigma,
                   patch_h=patch_hw[0], patch_w=patch_hw[1])


def build_edit_mask(src_raw: Optional[torch.Tensor], tar_raw: Optional[torch.Tensor],
                    kept_raw: Optional[torch.Tensor], p: MaskParams) -> Optional[torch.Tensor]:
    """Per-token edit mask used by the anchor steps.

    Inputs are [B, Lq] saliencies (each block's map smoothed and min-max
    normalised, then averaged over blocks): removed source words (`src_raw`,
    from the source pass), target-only words (`tar_raw`, from the current
    denoising forward) and kept source words (`kept_raw`, source pass).

    Returns [B, Lq] float in [0, 1] or None when the needed inputs are missing:
    1 = edit region (generated V stays free), 0 = anchored to V_src.

    combine:
      "union"    (src above mean + k*std) OR (tar above mean + k*std), then the
                 strongest tubes, dilation, feathering.  Default.
      "hadamard" anchor = A_kept * (1 - A_tar) with min-max maps (continuous,
                 no threshold); used for global scene transformations.
      "kept_inv" anchor = silhouette of the kept source words, free the rest;
                 used for background swaps.
    """
    def reshape(raw: torch.Tensor) -> torch.Tensor:
        B, Lq = raw.shape
        T = Lq // (p.patch_h * p.patch_w)
        return raw.float().view(B, T, p.patch_h, p.patch_w)

    def above_typical(raw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if raw is None:
            return None
        x = reshape(raw)
        mu = x.flatten(2).mean(dim=-1)[..., None, None]
        sd = x.flatten(2).std(dim=-1)[..., None, None]
        return x > (mu + p.thresh_k * sd)

    def excess(raw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if raw is None:
            return None
        x = reshape(raw)
        mu = x.flatten(2).mean(dim=-1)[..., None, None]
        sd = x.flatten(2).std(dim=-1)[..., None, None]
        return (x - (mu + p.thresh_k * sd)).clamp(min=0.0)

    def finish(m: torch.Tensor) -> torch.Tensor:
        if p.feather_sigma > 0:
            m = gaussian_smooth_2d(m, p.feather_sigma).clamp(0.0, 1.0)
        return m.reshape(m.shape[0], -1)

    if p.combine == "kept_inv":
        if kept_raw is None:
            return None
        k = above_typical(kept_raw).float()
        if p.top_islands > 0:
            exc_k = excess(kept_raw) if p.top_islands_rel > 0 else None
            k = top_islands_tube(k, p.top_islands, excess=exc_k, rel=p.top_islands_rel)
        if p.dilate > 0:                      # protective ring around the subject
            k = dilate_2d(k, p.dilate)
        return finish(1.0 - k)

    if p.combine == "hadamard":
        if kept_raw is None or tar_raw is None:
            return None
        a_kept = normalize_per_frame(reshape(kept_raw))
        a_tar = normalize_per_frame(reshape(tar_raw))
        if p.dilate > 0:                      # err toward releasing the edit
            a_tar = dilate_2d(a_tar, p.dilate)
        return finish(1.0 - a_kept * (1.0 - a_tar))

    if p.combine != "union":
        raise ValueError(f"unknown combine mode {p.combine!r}")

    parts = [s for s in (above_typical(src_raw), above_typical(tar_raw)) if s is not None]
    if not parts:
        return None
    m = parts[0]
    for part in parts[1:]:
        m = m | part.to(m.device)
    m = m.float()
    if p.top_islands > 0:
        exc = None
        if p.top_islands_rel > 0:
            parts_e = [e for e in (excess(src_raw), excess(tar_raw)) if e is not None]
            exc = parts_e[0]
            for e in parts_e[1:]:
                exc = torch.maximum(exc, e.to(exc.device))
        m = top_islands_tube(m, p.top_islands, excess=exc, rel=p.top_islands_rel)
    if p.dilate > 0:
        m = dilate_2d(m, p.dilate)
    return finish(m)

###  Forward hook: saliencies -> anchor mask ###

class EditMaskHook:
    """Forward hook registered on transformer block `mask_block_max`.

    Runs after that block has finished, i.e. after the last captured
    cross-attention and before the anchoring blocks of the same forward pass.
    Finalises the block means of the saliencies and writes the mask to
    `state.anchor_mask`. Optionally appends every mask to `log`.
    """

    def __init__(self, state: HookState, params: MaskParams,
                 log: Optional[List[dict]] = None, current_chunk_start=lambda: -1) -> None:
        self.state = state
        self.params = params
        self.log = log
        self._current_chunk_start = current_chunk_start

    def __call__(self, module, inputs, output) -> None:
        st = self.state
        if not st.capture_active or st.mode != "inject":
            return
        # src / srckept are filled once per chunk (source pass) and persist;
        # tar is refilled at every capture step.
        st.finalize_saliency()
        src_raw = st.saliency["src"].mean
        tar_raw = st.saliency["tar"].mean
        kept_raw = st.saliency["srckept"].mean
        if src_raw is None and tar_raw is None:
            return
        mask = build_edit_mask(src_raw, tar_raw, kept_raw, self.params)
        st.anchor_mask = mask
        if self.log is not None and mask is not None:
            def cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                return t.detach().to("cpu").to(torch.float16).clone() if t is not None else None
            self.log.append({
                "current_start_frame": int(self._current_chunk_start()),
                "denoise_iter": int(st.denoise_iter),
                "mask": cpu(mask),
                "src_raw": cpu(src_raw),
                "tar_raw": cpu(tar_raw),
                "srckept_raw": cpu(kept_raw),
            })

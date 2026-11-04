"""State shared between the pipeline and the patched attention layers.

`AREditPipeline` owns one `HookState`; every patched self-/cross-attention
holds a reference to it. The pipeline writes the per-chunk / per-step flags
before each forward pass, the attention layers read them (and write the V_src
snapshots and the saliency maps).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Mode = Optional[str]          # "store" | "inject" | None
Side = str                    # "src" | "tar" | "srckept"
SIDES: Tuple[Side, ...] = ("src", "tar", "srckept")


@dataclass
class SaliencyAccumulator:
    """Running mean over transformer blocks of a per-token saliency map [B, Lq]."""

    _sum: Optional[torch.Tensor] = None
    _count: int = 0
    mean: Optional[torch.Tensor] = None   # finalised block mean

    def add(self, x: torch.Tensor) -> None:
        self._sum = x if self._sum is None else self._sum + x
        self._count += 1

    def finalize(self) -> None:
        """Turn the running sum into `mean` (keeps the previous mean if nothing was added)."""
        if self._sum is not None and self._count > 0:
            self.mean = self._sum / self._count
        self._sum = None
        self._count = 0

    def reset(self) -> None:
        self._sum = None
        self._count = 0
        self.mean = None


@dataclass
class HookState:
    """Everything the patched attention layers need to know.

    Static fields are set once from the config; per-video fields by
    `AREditPipeline.set_prompt_alignment`; per-chunk / per-step fields by the
    pipeline's chunk loop.
    """

    ### static ###
    inject_min_block: int             # commit: blocks > this receive V_src
    inject_alpha: float
    store_min_block: int              # source pass: blocks > this store V_src
    anchor_min_block_default: int     # anchor: blocks > this (regular steps)
    anchor_alpha: float
    mask_block_max: int               # saliency captured at blocks 1..this
    mask_sigma: float                 # per-block smoothing of the saliency
    patch_hw: Tuple[int, int] = (30, 52)

    ### per video ###
    text_alphas: Optional[torch.Tensor] = None   # [L_text]: 1 = aligned, 0 = target-only
    text_scale: float = 1.0
    text_boost_min_block: int = -1
    tar_gap_idxs: List[int] = field(default_factory=list)
    src_removed_idxs: List[int] = field(default_factory=list)
    src_kept_idxs: List[int] = field(default_factory=list)

    ### per chunk / per step ###
    mode: Mode = None
    v_store: Dict[int, torch.Tensor] = field(default_factory=dict)   # block id -> V_src
    c0_store_min_block: Optional[int] = None    # chunk-0 widening of the store
    inject_active: bool = False                 # commit step
    anchor_active: bool = False                 # anchor step
    capture_active: bool = False                # saliency capture + mask build this step
    anchor_min_block: int = -1                  # anchor range for the current step
    anchor_mask: Optional[torch.Tensor] = None  # [B, Lq] float, 1 = edit region (free)
    text_boost_active: bool = False
    capture_side: Optional[str] = None          # None | "src" | "tar"
    saliency: Dict[Side, SaliencyAccumulator] = field(
        default_factory=lambda: {s: SaliencyAccumulator() for s in SIDES})
    denoise_iter: int = -1

    def __post_init__(self) -> None:
        self.anchor_min_block = self.anchor_min_block_default

    ### per video ###

    def set_edit_tokens(self, tar_gap_idxs: List[int], src_removed_idxs: List[int],
                        src_kept_idxs: List[int]) -> None:
        self.tar_gap_idxs = list(tar_gap_idxs)
        self.src_removed_idxs = list(src_removed_idxs)
        self.src_kept_idxs = list(src_kept_idxs)
        for acc in self.saliency.values():
            acc.reset()
        self.anchor_mask = None

    def set_text_boost(self, alphas: Optional[torch.Tensor], scale: float, min_block: int) -> None:
        self.text_alphas = alphas
        self.text_scale = float(scale)
        self.text_boost_min_block = int(min_block)

    ### per chunk ###

    def begin_source_pass(self, c0_store_min_block: Optional[int], capture: bool) -> None:
        self.mode = "store"
        self.v_store.clear()
        self.c0_store_min_block = c0_store_min_block
        if capture:
            self.saliency["src"].reset()
            self.saliency["srckept"].reset()
            self.capture_side = "src"
        else:
            self.capture_side = None

    def end_source_pass(self) -> None:
        self.capture_side = None

    def begin_target_capture(self) -> None:
        self.saliency["tar"].reset()
        self.capture_side = "tar"

    def finalize_saliency(self) -> None:
        for acc in self.saliency.values():
            acc.finalize()

    def disarm(self) -> None:
        """Turn every hook off (before the context rerun)."""
        self.mode = None
        self.text_boost_active = False
        self.inject_active = False
        self.anchor_active = False
        self.capture_active = False
        self.capture_side = None
        self.anchor_mask = None
        self.anchor_min_block = self.anchor_min_block_default
        self.c0_store_min_block = None

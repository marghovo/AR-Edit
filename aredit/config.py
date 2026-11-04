"""Editing hyper-parameters of AR-Edit.

The backbone settings (denoising schedule, KV-cache window, chunk size) live in
the yaml config; everything that concerns the *edit* lives here, with the
default recipe as default values. The CLI flags are generated from this class.
"""

from __future__ import annotations
import argparse
from typing import Any, Dict, Optional
from dataclasses import asdict, dataclass, fields

COMBINE_MODES = ("union", "hadamard", "kept_inv")
EDIT_TYPES = ("change", "add", "remove", "style")

# (flag name, help text). Keys must match the dataclass field names.
_HELP: Dict[str, str] = {
    "inject_step": "Commit: denoising steps [0, inject_step) receive V_src.",
    "v_inject_min_block": "Commit: blocks with id > this receive V_src.",
    "v_inject_alpha": "Commit: V = alpha * V_src + (1 - alpha) * V_cur.",
    "late_anchor_start": "Anchor: first anchored denoising step.",
    "late_anchor_min_block": "Anchor: blocks with id > this are anchored. Default: --v_inject_min_block.",
    "late_anchor_alpha": "Anchor: pull toward V_src outside the edit mask (1 = replace).",
    "cta_first_chunk_last_min_block": "Chunk 0, last step: anchor blocks > this value with the previous "
                                      "step's mask. Negative value = off.",
    "cta_block_max": "Edit mask: saliency = mean over the cross-attention of blocks 1..this.",
    "cta_sigma": "Edit mask: Gaussian smoothing (latent patches) of each block's saliency.",
    "cta_thresh_k": "Edit mask: edit region = saliency > mean + k * std (per frame).",
    "cta_top_islands": "Edit mask: keep the strongest connected regions (0 = keep everything).",
    "cta_top_islands_rel": "Edit mask: keep every region scoring >= rel * best (0 = fixed top-N).",
    "cta_dilate": "Edit mask: 3x3 dilation iterations.",
    "cta_feather_sigma": "Edit mask: Gaussian feathering of the final mask to smooth out the boundaries (0 = hard).",
    "cta_combine": "Edit mask: union = removed OR target-only words (default); hadamard = continuous "
                   "kept x (1 - target) map (global scene changes); kept_inv = anchor only the "
                   "kept-word silhouette (background swaps).",
    "edit_type": "'style' disables the anchor for this video; the other values are informational.",
    "text_scale": "Text boost of the target-only words: log(scale) is added to their cross-attention "
                  "logits before the softmax. 1.0 disables the boost.",
    "text_boost_min_block": "Text boost: cross-attention of blocks > this (-1 = all blocks).",
    "text_boost_steps": "Text boost: number of denoising steps with the boost. Default: all.",
}


@dataclass
class AREditConfig:
    """All editing arguments. Defaults = the AR-Edit recipe."""

    ### commit (V injection from the clean source pass) ###
    inject_step: int = 2
    v_inject_min_block: int = 22
    v_inject_alpha: float = 1.0

    ### anchor (masked re-anchoring to V_src at the late steps) ###
    late_anchor: bool = True
    late_anchor_start: int = 2
    late_anchor_min_block: Optional[int] = None          # None -> v_inject_min_block
    late_anchor_alpha: float = 1.0
    cta_first_chunk_last_min_block: Optional[int] = 14   # None -> off

    ### edit mask ###
    cta_block_max: int = 19
    cta_sigma: float = 0.75
    cta_thresh_k: float = 0.5
    cta_top_islands: int = 1
    cta_top_islands_rel: float = 0.25
    cta_dilate: int = 1
    cta_feather_sigma: float = 0.7
    cta_combine: str = "union"
    edit_type: str = "change"

    ### text boost ###
    text_scale: float = 4.0
    text_boost_min_block: int = -1
    text_boost_steps: Optional[int] = None               # None -> all steps

    ### backbone facts needed for validation ###
    num_blocks: int = 30

    def __post_init__(self) -> None:
        self.validate()

    ### derived ###

    @property
    def anchor_min_block(self) -> int:
        """Blocks > anchor_min_block are anchored at the late steps."""
        return (self.late_anchor_min_block if self.late_anchor_min_block is not None
                else self.v_inject_min_block)

    @property
    def store_min_block(self) -> int:
        """Blocks > store_min_block store V_src during the source pass (commit and anchor share one store)."""
        if self.late_anchor:
            return min(self.v_inject_min_block, self.anchor_min_block)
        return self.v_inject_min_block

    @property
    def text_boost_enabled(self) -> bool:
        return self.text_scale != 1.0

    ### validation ###

    def validate(self) -> None:
        if self.inject_step < 0:
            raise ValueError("inject_step must be >= 0")
        if not -1 <= self.v_inject_min_block < self.num_blocks:
            raise ValueError(f"v_inject_min_block must be in [-1, {self.num_blocks - 1}]")
        if self.late_anchor_start < 0:
            raise ValueError("late_anchor_start must be >= 0")
        if not 1 <= self.cta_block_max < self.num_blocks:
            raise ValueError(f"cta_block_max must be in [1, {self.num_blocks - 1}]")
        if self.cta_combine not in COMBINE_MODES:
            raise ValueError(f"cta_combine must be one of {COMBINE_MODES}; got {self.cta_combine!r}")
        if self.edit_type not in EDIT_TYPES:
            raise ValueError(f"edit_type must be one of {EDIT_TYPES}; got {self.edit_type!r}")
        if self.cta_top_islands_rel < 0:
            raise ValueError("cta_top_islands_rel must be >= 0")
        if self.text_scale <= 0:
            raise ValueError("text_scale must be > 0")
        if (self.cta_first_chunk_last_min_block is not None
                and not -1 <= self.cta_first_chunk_last_min_block < self.num_blocks):
            raise ValueError("cta_first_chunk_last_min_block out of range")

    ### (de)serialisation ###

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    ### CLI ###

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add one flag per field (defaults from the dataclass)."""
        group = parser.add_argument_group("editing (defaults = AR-Edit recipe)")
        defaults = cls()
        for f in fields(cls):
            if f.name in ("late_anchor", "num_blocks"):
                continue
            default = getattr(defaults, f.name)
            kwargs: Dict[str, Any] = {"default": default, "help": _HELP[f.name]}
            if f.name == "cta_combine":
                kwargs["choices"] = COMBINE_MODES
            elif f.name == "edit_type":
                kwargs["choices"] = EDIT_TYPES
            elif f.name in ("late_anchor_min_block", "text_boost_steps"):
                kwargs["type"] = int                           # Optional[int], None = default
            elif f.name == "cta_first_chunk_last_min_block":
                kwargs["type"] = int                           # negative value = off
            elif f.name in ("v_inject_alpha", "late_anchor_alpha", "cta_sigma", "cta_thresh_k",
                            "cta_top_islands_rel", "cta_feather_sigma", "text_scale"):
                kwargs["type"] = float
            else:
                kwargs["type"] = int
            group.add_argument(f"--{f.name}", **kwargs)
        group.add_argument("--no_anchor", action="store_true",
                           help="Disable the masked anchor steps (commit only).")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AREditConfig":
        values: Dict[str, Any] = {}
        for f in fields(cls):
            if hasattr(args, f.name):
                values[f.name] = getattr(args, f.name)
        if getattr(args, "no_anchor", False):
            values["late_anchor"] = False
        c0 = values.get("cta_first_chunk_last_min_block")
        if c0 is not None and c0 < 0:
            values["cta_first_chunk_last_min_block"] = None
        return cls(**values)

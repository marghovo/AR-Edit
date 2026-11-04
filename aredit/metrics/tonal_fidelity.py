"""Tonal Fidelity (TF) and TF-u: Estimates burning and tonal artifacts in the edited video?

Definition
----------
V channel   per-pixel max over RGB, scaled to [0, 1] (the HSV "value")
highlights  pixels with V >= 0.90
histogram   50 uniform bins on [0.90, 1.00], pooled over ALL frames of the
            clip, normalised to a probability distribution
KL          KL(p_edited || p_source)
TF          1 / (1 + KL) (Equation 3)

TF    : histogram over the whole frame.
TF-u  : histogram over the UNEDITED region only. Which side of the mask is
        "unedited" depends on the edit category:
          background edits (change_background, change_style, global,
          global_scene_transformation): the subject is what stays unedited,
          so the given mask is used;
          every other category: the subject is edited, so the mask
          complement is used.

Inputs: mp4 files, uint8. A mask video pixel is "inside the mask" where max(R, G, B) >= 128.

This module depends on numpy and decord only.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

V_LOW: float = 0.90
V_HIGH: float = 1.00
N_BINS: int = 50
EPS: float = 1e-12
MASK_THRESHOLD: int = 128
BIN_EDGES: np.ndarray = np.linspace(V_LOW, V_HIGH, N_BINS + 1)

BACKGROUND_EDIT_CATEGORIES = frozenset({
    "change_background",
    "change_style",
    "global",
    "global_scene_transformation",
})

REGION_MASK = "fg(mask)"
REGION_COMPLEMENT = "bg(~mask)"


### I/O ###

def load_video(path: str) -> np.ndarray:
    """All frames of an mp4 as uint8, shape (T, H, W, 3)."""
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    return reader.get_batch(range(len(reader))).asnumpy().astype(np.uint8)


def load_mask(path: str, threshold: int = MASK_THRESHOLD) -> np.ndarray:
    """Boolean (T, H, W): True where the mask video is bright."""
    frames = load_video(path)
    return frames.max(axis=-1) >= threshold


### math ###

def value_channel(rgb: np.ndarray) -> np.ndarray:
    """HSV value in [0, 1] as float32, shape (..., H, W) for an (..., H, W, 3) uint8 input."""
    return rgb.max(-1).astype(np.float32) / 255.0


def highlight_histogram(values: np.ndarray) -> np.ndarray:
    """Normalised N_BINS histogram of the highlight values (V >= V_LOW) in `values`.

    Returns the zero vector when there are no highlights.
    """
    highlights = values[values >= V_LOW]
    if highlights.size == 0:
        return np.zeros(N_BINS, dtype=np.float64)
    counts, _ = np.histogram(highlights, bins=BIN_EDGES)
    counts = counts.astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with EPS smoothing and renormalisation of both inputs."""
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def tf_from_kl(kl: float) -> float:
    """Equation 3"""
    return 1.0 / (1.0 + kl)


def unedited_region(category: str) -> str:
    """Which side of the mask is the unedited region for this edit category."""
    return REGION_MASK if category in BACKGROUND_EDIT_CATEGORIES else REGION_COMPLEMENT


### result ###

@dataclass(frozen=True)
class TonalFidelityResult:
    kl: float
    tf: float
    kl_u: Optional[float] = None
    tf_u: Optional[float] = None
    region: Optional[str] = None
    category: Optional[str] = None

    @property
    def has_unedited(self) -> bool:
        return self.kl_u is not None

    def to_dict(self) -> Dict[str, object]:
        """{kl, tf} plus {kl_u, tf_u, region, category} when a mask was given."""
        out: Dict[str, object] = {"kl": self.kl, "tf": self.tf}
        if self.has_unedited:
            out.update({"kl_u": self.kl_u, "tf_u": self.tf_u,
                        "region": self.region, "category": self.category})
        return out


### core ###

def tonal_fidelity_from_arrays(source_rgb: np.ndarray,
                               edited_rgb: np.ndarray,
                               mask: Optional[np.ndarray] = None,
                               category: str = "") -> TonalFidelityResult:
    """TF (and TF-u when `mask` is given) from frames.

    source_rgb, edited_rgb : uint8 (T, H, W, 3)
    mask                   : bool (T, H, W), True = edit-region mask
    category               : edit category, selects the unedited region for TF-u
    """
    source_v = value_channel(source_rgb)
    edited_v = value_channel(edited_rgb)
    n_frames = min(len(source_v), len(edited_v))
    source_v, edited_v = source_v[:n_frames], edited_v[:n_frames]

    kl = kl_divergence(highlight_histogram(edited_v.ravel()),
                       highlight_histogram(source_v.ravel()))
    if mask is None:
        return TonalFidelityResult(kl=kl, tf=tf_from_kl(kl))

    n_frames = min(n_frames, len(mask))
    source_v, edited_v, mask = source_v[:n_frames], edited_v[:n_frames], mask[:n_frames]
    region = unedited_region(category)
    select = mask if region == REGION_MASK else ~mask
    kl_u = kl_divergence(highlight_histogram(edited_v[select]),
                         highlight_histogram(source_v[select]))
    return TonalFidelityResult(kl=kl, tf=tf_from_kl(kl), kl_u=kl_u, tf_u=tf_from_kl(kl_u),
                               region=region, category=category)


def tonal_fidelity(source_video: str, edited_video: str,
                   mask_video: Optional[str] = None,
                   category: str = "") -> TonalFidelityResult:
    """TF for a source / edited pair; TF-u as well when `mask_video` is given."""
    mask = load_mask(mask_video) if mask_video is not None else None
    return tonal_fidelity_from_arrays(load_video(source_video), load_video(edited_video),
                                      mask=mask, category=category)

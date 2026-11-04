"""Video / prompt I/O for the AR-Edit CLI."""

from __future__ import annotations

import os
import torch
import logging
import numpy as np
import torchvision
from einops import rearrange
from typing import Optional, Tuple
from torchvision.io import write_video
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


LOGGER = logging.getLogger(__name__)


def load_mp4_as_tensor(video_path: str, max_frames: Optional[int] = None,
                       resize_hw: Optional[Tuple[int, int]] = None,
                       normalize: bool = True) -> torch.Tensor:
    """Load an mp4 as a float tensor [C, T, H, W] (in [-1, 1] when `normalize`)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        _, t, _, _ = video.shape
        video = torch.stack(
            [TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0
    return video


def read_first_prompt(path: str) -> str:
    """First non-empty line of a text file."""
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No prompt found in {path}")
    return lines[0]


def _label_bar(text: str, width: int, height: int = 32) -> np.ndarray:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    img = Image.new("RGB", (width, height), (0, 0, 0))
    ImageDraw.Draw(img).text((10, 4), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """[T, H, W, C] float in [0, 1] -> uint8."""
    return (video.clamp(0, 1).numpy() * 255.0).astype(np.uint8)


def save_results(output: torch.Tensor, input_video: torch.Tensor, save_dir: str, fps: int) -> None:
    """Write output.mp4, input.mp4 (trimmed to the output length) and compare.mp4.

    output:      [T, H, W, C] float in [0, 1]
    input_video: [1, C, T, H, W] float in [-1, 1]
    """
    os.makedirs(save_dir, exist_ok=True)
    output_np = to_uint8_frames(output)
    input_np = input_video[0].permute(1, 2, 3, 0).cpu().float()
    input_np = to_uint8_frames(input_np * 0.5 + 0.5)[:output_np.shape[0]]

    write_video(os.path.join(save_dir, "output.mp4"), torch.from_numpy(output_np), fps=fps)
    write_video(os.path.join(save_dir, "input.mp4"), torch.from_numpy(input_np), fps=fps)

    w = output_np.shape[2]
    bar_in, bar_out = _label_bar("Input", w), _label_bar("AR-Edit", w)
    n = min(output_np.shape[0], input_np.shape[0])
    frames = [np.concatenate([np.concatenate([bar_in, input_np[i]], axis=0),
                              np.concatenate([bar_out, output_np[i]], axis=0)], axis=1)
              for i in range(n)]
    write_video(os.path.join(save_dir, "compare.mp4"), torch.from_numpy(np.stack(frames)),
                fps=fps, options={"crf": "17"})
    LOGGER.info("Results written to %s", save_dir)

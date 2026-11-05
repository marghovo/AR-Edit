"""AR-Edit command line: edit one video with a source prompt and a target prompt.

Example:
    python inference_aredit.py \
        --checkpoint_path checkpoints/Reward-Forcing-T2V-1.3B/rewardforcing.pt \
        --wan_checkpoint_dir checkpoints \
        --video_path examples/zebra.mp4 \
        --src_prompt "A zebra walks across the savanna." \
        --tar_prompt "A horse walks across the savanna." \
        --output_folder outputs

All editing flags default to the AR-Edit recipe (see `aredit/config.py`).
"""

from __future__ import annotations

import os
import json
import torch
import logging
import argparse
from omegaconf import OmegaConf
from utils.misc import set_seed
from aredit.config import AREditConfig
from aredit.runner import AREditRunner
from demo_utils.memory import get_cuda_free_memory_gb
from aredit.video_io import load_mp4_as_tensor, read_first_prompt, save_results

LOGGER = logging.getLogger("aredit")
LOW_MEMORY_GB = 40


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AR-Edit: streaming video editing on Reward-Forcing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("paths")
    io.add_argument("--config_path", default="configs/aredit_reward_forcing.yaml",
                    help="Backbone settings (yaml).")
    io.add_argument("--checkpoint_path", required=True,
                    help="Reward-Forcing generator weights (rewardforcing.pt).")
    io.add_argument("--wan_checkpoint_dir", default=None,
                    help="Directory containing Wan2.1-T2V-1.3B/ (T5 encoder, VAE, DiT config). "
                         "Overrides checkpoint_path in the yaml.")
    io.add_argument("--video_path", required=True)
    io.add_argument("--video_name", default=None, help="Output sub-folder. Default: video file stem.")
    io.add_argument("--output_folder", required=True)
    io.add_argument("--cta_save_masks", default=None,
                    help="Optional .pt path: per-step edit masks and saliencies.")
    pr = p.add_argument_group("prompts (inline string or text file, first non-empty line)")
    pr.add_argument("--src_prompt", default=None, help="Source (input video) prompt.")
    pr.add_argument("--tar_prompt", default=None, help="Target (edit) prompt.")
    pr.add_argument("--src_prompt_file", default=None)
    pr.add_argument("--tar_prompt_file", default=None)
    vid = p.add_argument_group("video")
    vid.add_argument("--height", type=int, default=480)
    vid.add_argument("--width", type=int, default=832)
    vid.add_argument("--fps", type=int, default=16)
    vid.add_argument("--num_frames", type=int, default=81,
                     help="Max pixel frames read from the input (1 + 4k; 81 -> 7 chunks).")
    vid.add_argument("--trim_start_s", type=float, default=0.0,
                     help="Skip this many seconds of the input before reading the frames.")
    vid.add_argument("--seed", type=int, default=0)
    AREditConfig.add_cli_arguments(p)
    return p


def resolve_prompts(args: argparse.Namespace) -> tuple[str, str]:
    if (args.src_prompt is None) == (args.src_prompt_file is None):
        raise SystemExit("Give exactly one of --src_prompt / --src_prompt_file.")
    if (args.tar_prompt is None) == (args.tar_prompt_file is None):
        raise SystemExit("Give exactly one of --tar_prompt / --tar_prompt_file.")
    src = args.src_prompt if args.src_prompt is not None else read_first_prompt(args.src_prompt_file)
    tar = args.tar_prompt if args.tar_prompt is not None else read_first_prompt(args.tar_prompt_file)
    if not src.strip():
        raise SystemExit("The source prompt must not be empty: it drives the text boost and the edit mask.")
    return src, tar


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    src_prompt, tar_prompt = resolve_prompts(args)
    edit_config = AREditConfig.from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    free_gb = get_cuda_free_memory_gb(device)
    low_memory = free_gb < LOW_MEMORY_GB
    LOGGER.info("Free VRAM %.1f GB (low-memory mode: %s)", free_gb, low_memory)

    backbone_config = OmegaConf.load(args.config_path)
    if args.wan_checkpoint_dir is not None:
        backbone_config.checkpoint_path = args.wan_checkpoint_dir

    runner = AREditRunner(backbone_config, edit_config, device)
    runner.load_weights(args.checkpoint_path)
    runner.to_device(low_memory)

    input_video = load_mp4_as_tensor(
        args.video_path, max_frames=args.num_frames, resize_hw=(args.height, args.width),
        trim_start_s=args.trim_start_s,
    ).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    LOGGER.info("Input video shape: %s", tuple(input_video.shape))

    runner.set_prompts(src_prompt, tar_prompt)
    mask_log = runner.pipeline.enable_mask_logging() if args.cta_save_masks else None

    output = runner.edit(input_video, src_prompt, tar_prompt, low_memory=low_memory)

    video_name = args.video_name or os.path.splitext(os.path.basename(args.video_path))[0]
    save_dir = os.path.join(args.output_folder, video_name)
    save_results(output, input_video, save_dir, args.fps)
    with open(os.path.join(save_dir, "run_args.json"), "w") as f:
        json.dump({"cli": vars(args), "src_prompt": src_prompt, "tar_prompt": tar_prompt,
                   "edit_config": edit_config.to_dict()}, f, indent=2)
    if mask_log is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.cta_save_masks)), exist_ok=True)
        torch.save(mask_log, args.cta_save_masks)
        LOGGER.info("Edit masks saved to %s", args.cta_save_masks)


if __name__ == "__main__":
    main()

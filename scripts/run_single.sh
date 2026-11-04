#!/bin/bash
# Edit one video with the AR-Edit default recipe.
# Usage: bash scripts/run_single.sh <video.mp4> "<source prompt>" "<target prompt>" [output_dir]
set -eu
VIDEO=$1
SRC=$2
TAR=$3
OUT=${4:-outputs}
CKPT_DIR=${CKPT_DIR:-checkpoints}

python inference_aredit.py \
    --checkpoint_path "$CKPT_DIR/Reward-Forcing-T2V-1.3B/rewardforcing.pt" \
    --wan_checkpoint_dir "$CKPT_DIR" \
    --video_path "$VIDEO" \
    --src_prompt "$SRC" \
    --tar_prompt "$TAR" \
    --output_folder "$OUT" \
    --seed 0

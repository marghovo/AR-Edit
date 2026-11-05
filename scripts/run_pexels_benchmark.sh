#!/bin/bash
# AR-Edit on a benchmark JSON, one clip per SLURM array task, with the
# per-category mask routing used in the paper:
#   global_scene_transformation : --cta_combine hadamard --edit_type change
#   change_style                : --cta_combine union    --edit_type change
#   change_background           : --cta_combine kept_inv --edit_type <json edit_type>
#   everything else             : --cta_combine union    --edit_type <json edit_type>
#
# Defaults to the released benchmark (benchmark/ar_edit_pexels_benchmark.json, videos
# downloaded to benchmark/videos with benchmark/download_pexels.py). Any JSON works whose
# entries have: id, file_name (relative to VIDEOS_DIR) or video_path, source_prompt (or
# caption), target_prompt, edit_type (change | add | remove | style), category (or an id
# of the form <prefix>-<category>-<number>) and optionally trim_start_s.
#
# Usage:
#   sbatch --array=0-119 scripts/run_pexels_benchmark.sh
# with OUT_DIR, CKPT_DIR (and the SLURM account / partition lines below) adapted to your
# cluster. Without SLURM: CLIP_IDX=<i> bash scripts/run_pexels_benchmark.sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=aredit
#SBATCH --output=logs/aredit_%A_%a.out
set -eu

BENCH_JSON=${BENCH_JSON:-benchmark/ar_edit_pexels_benchmark.json}
VIDEOS_DIR=${VIDEOS_DIR:-benchmark/videos}
OUT_DIR=${OUT_DIR:?output directory}
CKPT_DIR=${CKPT_DIR:-checkpoints}
CLIP_IDX=${CLIP_IDX:-${SLURM_ARRAY_TASK_ID:?set CLIP_IDX or run as a SLURM array}}
mkdir -p "$OUT_DIR/_masks"

field() { python3 -c "import json,sys;e=json.load(open('$BENCH_JSON'))[$CLIP_IDX];print(e.get('$1') or '')"; }
CID=$(field id)
VIDEO=$(field video_path); VIDEO=${VIDEO:-$VIDEOS_DIR/$(field file_name)}
SRC=$(field source_prompt); SRC=${SRC:-$(field caption)}
TAR=$(field target_prompt)
TRIM=$(field trim_start_s); TRIM=${TRIM:-0}
ETYPE=$(field edit_type); ETYPE=${ETYPE:-change}
CAT=$(field category); CAT=${CAT:-$(echo "$CID" | cut -d- -f2)}

case "$CAT" in
    global_scene_transformation) ROUTE="--cta_combine hadamard --edit_type change" ;;
    change_style)                ROUTE="--cta_combine union --edit_type change" ;;
    change_background)           ROUTE="--cta_combine kept_inv --edit_type $ETYPE" ;;
    *)                           ROUTE="--cta_combine union --edit_type $ETYPE" ;;
esac
echo "=== clip $CLIP_IDX  id=$CID  category=$CAT  route=[$ROUTE]"

python inference_aredit.py \
    --checkpoint_path "$CKPT_DIR/Reward-Forcing-T2V-1.3B/rewardforcing.pt" \
    --wan_checkpoint_dir "$CKPT_DIR" \
    --video_path "$VIDEO" \
    --video_name "$CID" \
    --src_prompt "$SRC" \
    --tar_prompt "$TAR" \
    --output_folder "$OUT_DIR" \
    --trim_start_s "$TRIM" \
    --cta_save_masks "$OUT_DIR/_masks/$CID.pt" \
    $ROUTE \
    --seed 0
echo "=== done $CID"

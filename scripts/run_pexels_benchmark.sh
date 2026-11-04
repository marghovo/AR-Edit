#!/bin/bash
# AR-Edit on a benchmark JSON, one clip per SLURM array task, with the
# per-category mask routing used in the paper:
#   global_scene_transformation : --cta_combine hadamard --edit_type change
#   change_style                : --cta_combine union    --edit_type change
#   change_background           : --cta_combine kept_inv --edit_type <json edit_type>
#   everything else             : --cta_combine union    --edit_type <json edit_type>
#
# JSON entries need: id, video_path, caption (source prompt), target_prompt,
# edit_type (change | add | remove | style) and category (or an id of the form
# <prefix>-<category>-<number>).
#
# Usage:
#   sbatch --array=0-$((N-1)) scripts/run_pexels_benchmark.sh
# with BENCH_JSON, OUT_DIR, CKPT_DIR (and the SLURM account / partition lines
# below) adapted to your cluster. Without SLURM: CLIP_IDX=<i> bash scripts/run_pexels_benchmark.sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=aredit
#SBATCH --output=logs/aredit_%A_%a.out
set -eu

BENCH_JSON=${BENCH_JSON:?path to the benchmark json}
OUT_DIR=${OUT_DIR:?output directory}
CKPT_DIR=${CKPT_DIR:-checkpoints}
CLIP_IDX=${CLIP_IDX:-${SLURM_ARRAY_TASK_ID:?set CLIP_IDX or run as a SLURM array}}
mkdir -p "$OUT_DIR/_masks"

field() { python3 -c "import json,sys;e=json.load(open('$BENCH_JSON'))[$CLIP_IDX];print(e.get('$1') or '')"; }
CID=$(field id)
VIDEO=$(field video_path)
SRC=$(field caption)
TAR=$(field target_prompt)
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
    --cta_save_masks "$OUT_DIR/_masks/$CID.pt" \
    $ROUTE \
    --seed 0
echo "=== done $CID"

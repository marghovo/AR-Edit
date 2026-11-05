# AR-Edit Pexels benchmark

120 hand-curated video clips from [Pexels](https://www.pexels.com), 12 in each of 10 editing categories, each with a
source prompt describing the video and a target prompt that changes only the edited words. This is the benchmark of
the AR-Edit paper. The clips are not redistributed here: `ar_edit_pexels_benchmark.json` lists them with their Pexels
links, and `download_pexels.py` fetches them.

## Download

```bash
export PEXELS_API_KEY=...            # free key from https://www.pexels.com/api/
python benchmark/download_pexels.py benchmark/ar_edit_pexels_benchmark.json benchmark/videos
```

## Run AR-Edit on the benchmark

```bash
export OUT_DIR=outputs/benchmark CKPT_DIR=checkpoints
sbatch --array=0-119 scripts/run_pexels_benchmark.sh      # or: CLIP_IDX=3 bash scripts/run_pexels_benchmark.sh
```

The runner reads the JSON (`BENCH_JSON`, default `benchmark/ar_edit_pexels_benchmark.json`), the videos from
`VIDEOS_DIR` (default `benchmark/videos`), applies the per-category mask routing of the paper and writes
`$OUT_DIR/<id>/{output,input,compare}.mp4`. Each clip is processed as in the paper: the first 81 frames from
`trim_start_s` at the file's native frame rate, stretched to 832x480, and the result is written at 16 fps.

## Record fields

| Field | Meaning |
|---|---|
| `id` | `pexels-<category>-<pexels_id>`, the clip id used for every output folder |
| `category`, `edit_type`, `edit_object` | one of the 10 categories; `change`, `add` or `style`; the edited noun when one was annotated |
| `source_prompt`, `target_prompt` | caption of the video and the edit prompt; they differ only in the edited words |
| `pexels_id`, `pexels_url`, `uploader`, `uploader_url`, `license` | the Pexels page of the clip and its author; all clips are under the Pexels License |
| `file_name` | `<category>/<pexels slug>__<pexels_id>.mp4`, the name the download script writes |
| `file_width`, `file_height`, `file_fps`, `file_frames`, `file_duration_s` | the rendition used; `pexels_max_*` is the largest rendition Pexels lists |
| `trim_start_s` | start of the benchmark segment inside the file (0 for 118 clips, 2.0 s and 5.0 s for two) |
| `window_frames`, `window_duration_s` | the segment the model edits: 81 frames from `trim_start_s` at the native frame rate |
| `flow_motion`, `flow_residual`, `flow_pct_moving` | Farneback optical flow between the frames at 10% and 60% of the file: mean magnitude, mean magnitude after removing the global camera motion, share of pixels with residual above 1.5 px |
| `aesthetic` | LAION Aesthetic V2 score of the frame at 50% of the file |
| `window_motion` | mean absolute frame-to-frame difference (64x64 grey, 0-255) over the 81-frame segment |
| `has_people` | whether people are visible (5 clips, never as the main subject) |

## Statistics

Means over the 12 clips of each category. Motion, flow residual and % moving come from the optical flow above,
duration is the full source clip.

| Category | Videos | Aesthetic | Motion | Flow res. | % mov. | Dur. (s) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Motion modification | 12 | 5.10 | 4.33 | 4.88 | 61.4 | 20.34 |
| Global scene transformation | 12 | 5.39 | 8.74 | 8.64 | 76.6 | 17.82 |
| Object insertion | 12 | 5.19 | 10.21 | 7.77 | 86.9 | 34.63 |
| Object swapping | 12 | 4.94 | 7.34 | 7.33 | 70.8 | 10.53 |
| Change attribute: pose | 12 | 5.11 | 8.48 | 9.41 | 88.5 | 18.15 |
| Change attribute: material | 12 | 5.05 | 9.53 | 8.86 | 77.2 | 20.68 |
| Change style | 12 | 5.06 | 8.83 | 9.12 | 84.6 | 25.36 |
| Change background | 12 | 5.24 | 7.30 | 7.25 | 71.2 | 15.13 |
| Change attribute: content | 12 | 5.11 | 8.76 | 9.34 | 78.4 | 15.08 |
| Change attribute: color | 12 | 5.12 | 9.29 | 7.68 | 60.6 | 31.13 |
| **Total** | **120** | **5.13** | **8.28** | **8.03** | **75.6** | **20.89** |

## Selection rules

Please see Appendix C in the paper for benchmark construction details. 
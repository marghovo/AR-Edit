# AR-Edit project page

Static site: `index.html` plus `static/`. No build step; open it through a static server that
supports byte-range requests (Safari needs them to play video, and seeking needs them everywhere):
`npx serve` or `pip install rangehttpserver && python3 -m RangeHTTPServer` in this folder.
Plain `python3 -m http.server` does not. GitHub Pages works as is.

## Before publishing

- **Paper PDF**: the "Read the paper" and "Paper" buttons point to `static/paper.pdf`. Drop the
  camera-ready PDF there, or change the two `href="static/paper.pdf"` links in `index.html`
  to the arXiv / OpenReview URL.
- **Code**: links go to `https://github.com/marghovo/AR-Edit`.
- **Social preview**: `static/img/og.jpg`. Some platforms need an absolute URL in the
  `og:image` meta tag once the page has its final address.

## What lives where

| Path | Content |
| --- | --- |
| `static/videos/<clip>/*.mp4` | Comparison clips re-encoded from `../ablation_visual_results/*/raw` with the first 4 frames cut (77 frames, 16 fps), with a JPEG poster of the new first frame next to each |
| `static/data/media.json` | Index of which file is which method / setting, written by `tools/build_media.py` |
| `static/js/data.js` | Everything the page reads: media index, prompts, the Q/K/V cosine-similarity curves (read from the vector paper figures), and the piglet commit-then-anchor mask |
| `static/videos/recon/*.mp4` | Reconstruction videos, one per recipe, made from `../recon_grid_donkey_vt0_nfe1_2col_v4_videos` by `tools/build_media.py` (same 4-frame cut) |
| `tools/build_media.py` | Regenerates `static/videos` and `media.json` from the raw clips (`TRIM = 4` sets how many frames are cut; keep it in sync with `TRIM` in `static/js/main.js`) |

Only `index.html` and `static/` need to be deployed (about 120 MB, mostly video).

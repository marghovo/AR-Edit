"""Re-encode the raw comparison videos in ../ablation_visual_results into web-ready
H.264 clips + JPEG posters under static/videos, and write static/data/media.json.

The first TRIM frames of every clip are cut, so each web clip starts at raw frame TRIM
(81 -> 77 frames at 16 fps). static/js/main.js knows about this offset (TRIM there too).

Usage (from the site/ folder):  python3 tools/build_media.py [--force]
"""
import json, hashlib, subprocess, sys, os, re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
RAW = SITE.parent / 'ablation_visual_results'
OUT = SITE / 'static' / 'videos'
FORCE = '--force' in sys.argv
TRIM = 4          # drop raw frames 0..3; the web clip starts at raw frame 4

CLIP_SLUG = {
    'pexels-object_swapping-29640886': 'tiger-lion',
    'pexels-change_attribute_content-12079729': 'cow',
    'pexels-change_attribute_material-17118516': 'airplane',
    'pexels-global_scene_transformation-33376227': 'train',
    'pexels-change_attribute_content-5136530': 'jellyfish',
    'pexels-change_attribute_color-7613268': 'pitcher',
    'attr-pose-small-brown-piglet-to-cat-walking': 'piglet',
    'motion-gray-rabbit-green-to-winter-snow': 'rabbit',
    'zebra-walking-to-full-sprint': 'zebra',
    'attr-color-coffee-machine-kitchen-setting': 'coffee',
    'pexels-object_swapping-7097519': 'violin',
    'pexels-object_insertion-5866170': 'lantern',
    'pexels-change_attribute_color-31428087': 'seagull',
    'pexels-change_attribute_material-8328090': 'goblet',
    'pexels-change_attribute_pose-14318923': 'tree',
    'pexels-change_style-35087119': 'comic-dog',
    'pexels-object_swapping-6853902': 'cat-dog',
    'pexels-global_scene_transformation-32236429': 'opera',
    'pexels-change_attribute_color-33425923': 'butterfly',
    'pexels-global_scene_transformation-14160118': 'street',
}
SETS = {  # set key -> folder
    'rf': 'baselines_pexels_no_streamedit',
    'rf_classic': 'baselines_aredit_no_streamedit',
    'sf': 'baselines_pexels_selfforcing',
    'sched': 's_and_s_cta',
    'blocks': 'bmin_and_bcta',
    'comp': 'method_components',
    'boost': 'text_boost_scale',
}

def slug(s):
    s = s.replace('(default)', '').replace('(no CtA)', 'noCtA').replace('(no text boost)', '')
    s = re.sub(r'[$\\{}]|mathrm|min', lambda m: 'min' if m.group(0) == 'min' else '', s)
    s = re.sub(r'[^A-Za-z0-9]+', '-', s).strip('-').lower()
    return s

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def probe(p, entry):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', f'stream={entry}',
                        '-of', 'csv=p=0', str(p)], capture_output=True, text=True)
    return r.stdout.strip()

def nframes(p):
    try:
        return int(probe(p, 'nb_frames'))
    except ValueError:
        return -1

def encode(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = nframes(src) - TRIM
    if dst.exists() and dst.with_suffix('.jpg').exists() and nframes(dst) == want and not FORCE:
        return  # already trimmed
    tmp = dst.with_name(dst.stem + '.part.mp4')
    # Cutting frames needs a re-encode. CRF 20 / preset slow is visually lossless here
    # (SSIM ~0.99 against the raw clip); faststart moves the moov atom up front for streaming.
    rate = probe(src, 'r_frame_rate')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-an',
                    '-vf', f"select='gte(n,{TRIM})',setpts=N/FRAME_RATE/TB", '-r', rate,
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-profile:v', 'high', '-movflags', '+faststart', str(tmp)], check=True)
    assert nframes(tmp) == want, (src, nframes(tmp), want)
    os.replace(tmp, dst)
    # poster = the clip's new first frame
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(dst), '-frames:v', '1',
                    '-q:v', '5', str(dst.with_suffix('.jpg'))], check=True)

media = {'sets': {}}
seen = {}  # md5 -> relative path
for key, folder in SETS.items():
    man = json.load(open(RAW / folder / 'raw' / 'manifest.json'))
    cfg = json.load(open(RAW / folder / 'config.json'))
    defaults = [r['label'] for r in cfg['rows'] if r.get('default')]
    entry = {'rows': man['rows'], 'default': defaults[0] if defaults else None,
             'captions': cfg.get('captions', {}), 'clips': {}}
    for clip, rows in man['clips'].items():
        cs = CLIP_SLUG[clip]
        entry['clips'][cs] = {'id': clip, 'files': {}}
        for row, info in rows.items():
            src = RAW / folder / info['file']
            h = md5(src)
            if h in seen:
                rel = seen[h]
            else:
                name = 'input' if row == 'Input' else f'{key}-{slug(row)}'
                rel = f'{cs}/{name}.mp4'
                # avoid collisions between different inputs of the same clip
                if rel in seen.values():
                    rel = f'{cs}/{name}-{key}.mp4'
                encode(src, OUT / rel)
                seen[h] = rel
            entry['clips'][cs]['files'][row] = rel
            print(key, cs, row, '->', rel, flush=True)
    media['sets'][key] = entry

# reconstruction recipes: the donkey clip behind the recon figure, one video per recipe
RECON = SITE.parent / 'recon_grid_donkey_vt0_nfe1_2col_v4_videos'
RECON_NAME = {'L1': 'source', 'L2': 'inversion', 'L3': 'inv-v', 'L4': 'inv-fwd-0', 'L5': 'inv-fwd-625',
              'L6': 'inv-fwd-833', 'L7': 'inv-fwd-938', 'R1': 'inv-fwd-1000', 'R2': 'rand-v-t',
              'R3': 'rand-fwd-625', 'R4': 'rand-fwd-833', 'R5': 'rand-fwd-938', 'R6': 'rand-fwd-1000', 'R7': 'ours'}
if RECON.exists():
    for v in json.load(open(RECON / 'manifest.json'))['videos']:
        name = RECON_NAME[v['figure_position']]
        encode(RECON / v['file'], OUT / 'recon' / f'{name}.mp4')
        print('recon', v['figure_position'], v['label_on_figure'], '->', f'recon/{name}.mp4', flush=True)

json.dump(media, open(SITE / 'static' / 'data' / 'media.json', 'w'), indent=1)
tot = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
print('DONE', len(seen), 'unique videos', round(tot / 1e6, 1), 'MB')

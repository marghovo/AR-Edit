#!/usr/bin/env python3
"""Download the 120 benchmark clips from Pexels into <out_dir>/<category>/<file_name>.

Needs a free Pexels API key (https://www.pexels.com/api/):  PEXELS_API_KEY=... python download_pexels.py ar_edit_pexels_benchmark.json videos/
For every clip the rendition whose width x height equals file_width x file_height in the JSON is taken (the one used in
the paper); if Pexels no longer serves that rendition, the closest one is taken and a warning is printed.
"""
import json, os, sys, time, urllib.request
bench, out_dir = sys.argv[1], sys.argv[2]
key = os.environ.get('PEXELS_API_KEY') or sys.exit('set PEXELS_API_KEY')
for e in json.load(open(bench)):
    dst = os.path.join(out_dir, e['file_name'])
    if os.path.exists(dst): continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(f"https://api.pexels.com/videos/videos/{e['pexels_id']}", headers={'Authorization': key})
    meta = json.load(urllib.request.urlopen(req))
    files = [f for f in meta['video_files'] if f.get('file_type', '').startswith('video/mp4')]
    want = (e['file_width'], e['file_height'])
    exact = [f for f in files if (f['width'], f['height']) == want]
    pick = exact[0] if exact else min(files, key=lambda f: abs(f['width'] * f['height'] - want[0] * want[1]))
    if not exact: print(f"warning: {e['id']}: no {want[0]}x{want[1]} rendition, taking {pick['width']}x{pick['height']}")
    urllib.request.urlretrieve(pick['link'], dst); print('downloaded', dst); time.sleep(0.5)

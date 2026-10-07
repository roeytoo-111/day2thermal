"""Flag thermal frames that carry burned-in colour overlays (the live system draws GREEN boxes/marks into some thermal
streams). Those frames must not be training images (a model could learn "rectangle = drone") nor generation pairs.

Frame-exact: cv2 sequential decode of the (remuxed) thermal mp4, index = 0-based true frame. A frame is flagged when
it has >= --min-px saturated-green pixels (G > --g-min, R and B < --rb-max) at full 640x512 resolution.
(First version used G - max(R,B) > 40 and flagged compression chroma speckle on terrain: 2026-10-07 QA.)

    python3 src/data/thermal_overlay_flags.py --mp4-dir data/boson_work/mp4 --out data/boson_work/overlay_flags.json
"""
import os, sys, json, glob, argparse
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mp4-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-px", type=int, default=24)
ap.add_argument("--g-min", type=int, default=180)
ap.add_argument("--rb-max", type=int, default=100)
ap.add_argument("--sheet-dir", default=None, help="write a QA sheet of flagged frames per session here")
a = ap.parse_args()
res = {}
for f in sorted(glob.glob(os.path.join(a.mp4_dir, "*_thermal.mp4"))):
    s = os.path.basename(f).replace("_thermal.mp4", "")
    cap = cv2.VideoCapture(f)
    counts, i, tiles = [], 0, []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        # PURE overlay green only (the live system draws saturated green); chroma speckle on textured terrain from
        # the thermal stream's compression is greenish/magenta but never this saturated
        m = (im[..., 1] > a.g_min) & (im[..., 0] < a.rb_max) & (im[..., 2] < a.rb_max)
        n = int(m.sum()); counts.append(n)
        if a.sheet_dir and n >= a.min_px and len(tiles) < 8 and i % 7 == 0:
            ys, xs = np.nonzero(m); cx, cy = int(xs.mean()), int(ys.mean())
            c = im[max(cy - 60, 0):cy + 60, max(cx - 80, 0):cx + 80]
            tiles.append(cv2.resize(c, (240, 180)))
        i += 1
    counts = np.array(counts)
    flag = np.flatnonzero(counts >= a.min_px).tolist()
    res[s] = {"n_frames": len(counts), "flag": flag, "max_px": int(counts.max()) if len(counts) else 0}
    print(f"{s}: {len(counts)} frames, {len(flag)} flagged ({len(flag) / max(len(counts), 1):.0%}), max {res[s]['max_px']} px", flush=True)
    if a.sheet_dir and tiles:
        os.makedirs(a.sheet_dir, exist_ok=True)
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.sheet_dir, s + ".jpg"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))
json.dump(res, open(a.out, "w"))

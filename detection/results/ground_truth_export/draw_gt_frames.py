"""Draw ground-truth boxes from thermal_ground_truth.csv onto frames of a video, to eyeball that frame numbers and
boxes line up with the video (they should sit exactly on the drone).

    python3 draw_gt_frames.py --video 2026-07-02_scenario1_thermal.mp4 --csv thermal_ground_truth.csv \
        --video-name 2026-07-02_scenario1_thermal.mp4 --n 100 --out gt_check

Writes gt_check/frames/*.png (full frames, box drawn, a crop-zoom inset), gt_check/overview.mp4 (1 frame/s) and
gt_check/contact_sheet.jpg. Frames are decoded IN ORDER (frame index 0 = first frame); no seeking.
Needs: opencv-python, numpy, pandas.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="path to the video file")
ap.add_argument("--csv", default="thermal_ground_truth.csv")
ap.add_argument("--video-name", default=None, help="value of the csv 'video' column (default: the video's file name)")
ap.add_argument("--n", type=int, default=100, help="frames to draw")
ap.add_argument("--empty-frac", type=float, default=0.1, help="share of the frames that are has_drone=0 (box-free)")
ap.add_argument("--out", default="gt_check")
a = ap.parse_args()

d = pd.read_csv(a.csv)
d = d[d.video == (a.video_name or os.path.basename(a.video))]
if d.empty:
    raise SystemExit(f"no rows for that video name; the csv has: {sorted(pd.read_csv(a.csv).video.unique())}")
pos, neg = d[d.has_drone == 1], d[d.has_drone == 0]
n_neg = min(int(round(a.n * a.empty_frac)), len(neg))
pick = pd.concat([pos.iloc[np.linspace(0, len(pos) - 1, min(a.n - n_neg, len(pos))).astype(int)],
                  neg.iloc[np.linspace(0, len(neg) - 1, n_neg).astype(int)] if n_neg else neg.iloc[:0]]).sort_values("frame")
want = {int(r.frame): r for r in pick.itertuples()}
os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
vw = None
tiles = []
i = 0
while i <= max(want):
    ok, im = cap.read()                      # sequential decode: i is the 0-based frame index
    if not ok:
        break
    if i in want:
        r = want[i]
        vis = im.copy()
        if r.has_drone:
            x0, y0, x1, y1 = int(r.x0), int(r.y0), int(r.x1), int(r.y1)
            cv2.rectangle(vis, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 255, 255), 1)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2          # zoomed inset of the box area, top-left corner
            c = im[max(cy - 24, 0):cy + 24, max(cx - 24, 0):cx + 24]
            c = cv2.resize(c, (192, 192), interpolation=cv2.INTER_NEAREST)
            vis[0:192, 0:192] = c
            cv2.rectangle(vis, (0, 0), (192, 192), (0, 255, 255), 1)
        label = f"frame {i}  t={i / fps:.1f}s  {'DRONE' if r.has_drone else 'no drone (checked)'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (0, vis.shape[0] - th - 14), (tw + 12, vis.shape[0]), (0, 0, 0), -1)   # legible on bright terrain
        cv2.putText(vis, label, (6, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(a.out, "frames", f"frame_{i:06d}.png"), vis)
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(a.out, "overview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.0,
                                 (vis.shape[1], vis.shape[0]))
        vw.write(vis)
        tiles.append(cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)))
    i += 1
if vw is not None:
    vw.release()
cols = 5
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(os.path.join(a.out, "contact_sheet.jpg"),
            np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"drew {len(want)} frames ({int(pick.has_drone.sum())} with a drone) -> {a.out}/ "
      f"(frames/, overview.mp4, contact_sheet.jpg)")

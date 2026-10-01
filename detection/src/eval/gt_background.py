"""
Tag every labelled drone by what it is seen against: smooth background (sky) or textured (terrain,
horizon clutter). Our interceptor flies BELOW the target and looks up, so the operational case is the
sky subset; score_models.py reports it separately ("*_sky" sets).

Background measure: robust (MAD) std of a 3-sigma high-pass in a ring of --margin px around the GT box (box
+ a 6 px halo excluded, so a hot target's own glow doesn't count as clutter). <= --max-rough -> sky.
Checked on contact sheets (sky row / terrain row) for 0708 and 0730.

    python3 src/eval/gt_background.py --video data/videos/thermal.mp4 \\
        --boxes data/recall_ground_truth/boxes.csv --out data/recall_ground_truth/gt_bg.csv --sheet /tmp/bg_0708.jpg
"""
import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_boxes import FrameReader   # noqa: E402


def ring_rough(gray, box, margin, halo=6):
    """Robust (MAD) std of the high-pass in a ring around the box. The box is excluded with a `halo`: the
    3-sigma blur spreads a hot target's own energy a few px outward, which would read as clutter."""
    g = gray.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    H, W = g.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    X0, Y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    w = hp[Y0:min(y1 + margin, H), X0:min(x1 + margin, W)].copy()
    w[max(y0 - halo - Y0, 0):max(y1 + halo - Y0, 0), max(x0 - halo - X0, 0):max(x1 + halo - X0, 0)] = np.nan
    v = w[~np.isnan(w)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--max-rough", type=float, default=3.0, help="robust ring std; sky ~0.5-2.5")
    ap.add_argument("--sheet", default=None, help="QA contact sheet: sky row(s) then terrain row(s)")
    a = ap.parse_args()
    b = pd.read_csv(a.boxes)
    pos = b[b.verdict == "drone"].copy()
    fr = FrameReader(a.video, needed=pos.frame_id.astype(int).tolist())
    rough, crops = [], []
    for r in pos.itertuples():
        im = fr.read(int(r.frame_id))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        box = (r.x0, r.y0, r.x1, r.y1)
        rough.append(ring_rough(g, box, a.margin))
        cx, cy = int((r.x0 + r.x1) / 2), int((r.y0 + r.y1) / 2)
        c = im[max(cy - 32, 0):cy + 32, max(cx - 32, 0):cx + 32]
        crops.append(cv2.resize(c, (96, 96), interpolation=cv2.INTER_NEAREST) if c.size else np.zeros((96, 96, 3), np.uint8))
    pos["ring_rough"] = rough
    pos["bg"] = np.where(pos.ring_rough <= a.max_rough, "sky", "terrain")
    pos[["frame_id", "ring_rough", "bg"]].to_csv(a.out, index=False)
    print(f"{a.out}: {len(pos)} drones -> {pos.bg.value_counts().to_dict()} "
          f"(ring high-pass std <= {a.max_rough} = sky; median {np.median(rough):.2f})")
    if a.sheet:
        rows = []
        for kind in ("sky", "terrain"):
            idx = np.flatnonzero(pos.bg.values == kind)
            pick = idx[np.linspace(0, len(idx) - 1, min(12, len(idx))).astype(int)] if len(idx) else []
            tiles = [crops[i].copy() for i in pick]
            for t, i in zip(tiles, pick):
                cv2.putText(t, f"{kind[0]} {rough[i]:.1f}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            tiles += [np.zeros((96, 96, 3), np.uint8)] * (12 - len(tiles))
            rows.append(np.hstack(tiles))
        cv2.imwrite(a.sheet, np.vstack(rows))


if __name__ == "__main__":
    main()

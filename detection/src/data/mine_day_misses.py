r"""
Review tiles for drones the DAY model misses but the thermal model sees: the fw4 weakness (small, low-contrast drones
against cumulus / overcast; all 6 Boson misses were of this kind) and the hole build_rgb_addon.py cannot fill, since
its positives need a day detection by construction.

Source: build_boson_labels.py rows with status 'tneg' (thermal model fires >= its tneg threshold, the day model has
no drone-class detection >= 0.25 inside the thermal view). These are real drones the day model misses OR thermal
false alarms; a human decides with review_rgb_tiles.py (d = drone visible in the day tile, even faintly; f = nothing
there -> the tile becomes a negative; b = bird).
The thermal box is mapped into the 4K day frame (thermal lens undistortion, then the inverse of the segment's
H_day_to_undistorted_thermal) and a 640 tile is cut around it. Thermal detections are kept only when they form a
track (>= --min-track hits within +-15 frames, within 25 thermal px), which removes most single-frame speckle.
Tiles are appended to <addon>/review/tiles + tiles.csv (same format as build_rgb_addon.py: rv_* after finalize).

    python3 src/data/mine_day_misses.py --labels-dir data/flight_20261014/labels --reg-dir data/flight_20261014/reg \
        --day-dir data/flight_20261014/mp4 --addon data/flight_20261014/rgb_addon
"""
import os
import sys
import glob
import json
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from day2thermal.video_pairs import und_from_dist   # noqa: E402

TILE, MARGIN = 640, 48


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True); ap.add_argument("--reg-dir", required=True)
    ap.add_argument("--day-dir", required=True); ap.add_argument("--addon", required=True)
    ap.add_argument("--holdout", default=""); ap.add_argument("--min-therm", type=float, default=0.3)
    ap.add_argument("--min-track", type=int, default=3); ap.add_argument("--step", type=int, default=10,
                                                                          help="min day frames between tiles of a session")
    ap.add_argument("--max-per-session", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hold = {s for s in a.holdout.split(",") if s}
    tdir = os.path.join(a.addon, "review", "tiles"); os.makedirs(tdir, exist_ok=True)
    csv = os.path.join(a.addon, "review", "tiles.csv")
    old = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    have = set(old.tile) if len(old) else set()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))):
        s = os.path.basename(f)[:-4]
        if s in hold or s.startswith("hard_"):
            continue
        d = pd.read_csv(f)
        t = d[(d.status == "tneg") & (d.therm_conf >= a.min_therm)].copy()
        if t.empty:
            continue
        t["cx"], t["cy"] = (t.x0 + t.x1) / 2, (t.y0 + t.y1) / 2
        fr, cx, cy = t.frame.values, t.cx.values, t.cy.values
        ok = [((np.abs(fr - fr[i]) <= 15) & (np.hypot(cx - cx[i], cy - cy[i]) <= 25)).sum() >= a.min_track for i in range(len(t))]
        t = t[ok].sort_values("therm_conf", ascending=False)
        reg = json.load(open(os.path.join(a.reg_dir, s, "registration.json")))
        lam, th_size = reg["lens"]["lambda"], tuple(reg["thermal_size"])
        Hinv = {g["id"]: np.linalg.inv(np.array(g["H_day_to_undistorted_thermal"])) for g in reg["segments"]}
        chosen = []
        for r in t.itertuples():
            if all(abs(r.day_frame - c.day_frame) >= a.step for c in chosen):
                chosen.append(r)
            if len(chosen) >= a.max_per_session:
                break
        want = {int(r.day_frame): r for r in chosen}
        if not want:
            continue
        cap = cv2.VideoCapture(os.path.join(a.day_dir, f"{s}_day.mp4"))
        i, n = 0, 0
        while i <= max(want):
            if i not in want:
                if not cap.grab():
                    break
                i += 1; continue
            ok, im = cap.read()
            if not ok:
                break
            r = want[i]; H, W = im.shape[:2]
            xs, ys = np.array([r.x0, r.x1, r.x1, r.x0]), np.array([r.y0, r.y0, r.y1, r.y1])
            ux, uy = und_from_dist(xs, ys, lam, th_size)
            p = cv2.perspectiveTransform(np.stack([ux, uy], 1).reshape(-1, 1, 2).astype(np.float64), Hinv[r.seg]).reshape(-1, 2)
            box = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
            bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            tx = int(np.clip(bcx - TILE / 2 + rng.integers(-TILE // 4, TILE // 4 + 1), 0, W - TILE))
            ty = int(np.clip(bcy - TILE / 2 + rng.integers(-TILE // 4, TILE // 4 + 1), 0, H - TILE))
            if not (tx + MARGIN <= bcx <= tx + TILE - MARGIN and ty + MARGIN <= bcy <= ty + TILE - MARGIN):
                i += 1; continue
            stem = f"miss{s}_{i:06d}"
            if stem + ".jpg" not in have:
                cv2.imwrite(os.path.join(tdir, stem + ".jpg"), im[ty:ty + TILE, tx:tx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"tile": stem + ".jpg", "session": s, "day_frame": i, "rgb_conf": f"th{r.therm_conf:.2f}",
                             "x0": round(box[0] - tx, 1), "y0": round(box[1] - ty, 1), "x1": round(box[2] - tx, 1),
                             "y1": round(box[3] - ty, 1), "decision": ""})
                n += 1
            i += 1
        print(f"{s}: {len(t)} tracked tneg detections -> {n} day tiles", flush=True)
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
    out.to_csv(csv, index=False)
    print(f"{len(rows)} new review tiles -> {csv}")


if __name__ == "__main__":
    main()

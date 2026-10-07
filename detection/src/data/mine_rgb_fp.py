r"""
Mine day-model false positives at a LOW threshold (suggestion of Daniel's boss) as hard-negative tiles for the RGB
dataset (Fixed_wing_v3 addon). Where the drone is known -- frames whose status is 'pos' (day AND thermal model agree
on it; Daniel checked these boxes on the training sessions) -- every OTHER day detection (conf >= --conf, any class)
whose centre is more than --min-dist 4K px from the drone is a false alarm (one drone per flight): clouds, terrain,
birds. A 640x640 tile is cut around it; if the drone also falls fully inside that tile it gets its uav label, if it
would be cut by the tile edge the tile is skipped. Tiles go to <addon>/train as fp_*.jpg (empty label = negative).
Held-out sessions are skipped.

    python3 src/data/mine_rgb_fp.py --labels-dir data/boson_work/labels --rgb-dir data/boson_work/rgbdets \
        --day-dir data/boson_work/mp4 --addon data/boson_work/Fixed_wing_v3_boson
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np
import pandas as pd

TILE, UAV = 640, 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True); ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--day-dir", required=True); ap.add_argument("--addon", required=True)
    ap.add_argument("--holdout", default="08_58_15,10_15_59")
    ap.add_argument("--conf", type=float, default=0.05); ap.add_argument("--min-dist", type=float, default=300)
    ap.add_argument("--step", type=int, default=3, help="every N-th distinct confirmed day frame")
    ap.add_argument("--per-frame", type=int, default=2); ap.add_argument("--max-tiles", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hold = set(a.holdout.split(","))
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.addon, "train", sub), exist_ok=True)
    plan = {}
    for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))):
        s = os.path.basename(f)[:-4]
        if s in hold:
            continue
        d = pd.read_csv(f)
        pos = d[(d.status == "pos") & d.dx0.notna()].drop_duplicates("day_frame").iloc[::a.step]
        rgb = {e["src_frame"]: e["detections"] for e in json.load(open(os.path.join(a.rgb_dir, f"{s}.json")))}
        for r in pos.itertuples():
            drone = (float(r.dx0), float(r.dy0), float(r.dx1), float(r.dy1))
            dcx, dcy = (drone[0] + drone[2]) / 2, (drone[1] + drone[3]) / 2
            fps = [x for x in rgb.get(int(r.day_frame), []) if x["conf"] >= a.conf and
                   np.hypot((x["bbox"][0] + x["bbox"][2]) / 2 - dcx, (x["bbox"][1] + x["bbox"][3]) / 2 - dcy) > a.min_dist]
            fps = sorted(fps, key=lambda x: -x["conf"])[:a.per_frame]
            if fps:
                plan.setdefault(s, {})[int(r.day_frame)] = (drone, fps)
    n_all = sum(len(v) for v in plan.values())
    keep_frac = min(1.0, a.max_tiles / max(1, sum(len(fp) for v in plan.values() for _, fp in v.values())))
    written, with_drone, classes = 0, 0, {}
    for s, frames in plan.items():
        cap = cv2.VideoCapture(os.path.join(a.day_dir, f"{s}_day.mp4"))
        i, last = 0, max(frames)
        while i <= last:
            if i not in frames:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, im = cap.read()
            if not ok:
                break
            H, W = im.shape[:2]
            drone, fps = frames[i]
            for k, x in enumerate(fps):
                if rng.random() > keep_frac:
                    continue
                cx, cy = (x["bbox"][0] + x["bbox"][2]) / 2, (x["bbox"][1] + x["bbox"][3]) / 2
                tx = int(np.clip(cx - TILE / 2 + rng.integers(-150, 151), 0, W - TILE))
                ty = int(np.clip(cy - TILE / 2 + rng.integers(-150, 151), 0, H - TILE))
                inside = drone[0] >= tx and drone[1] >= ty and drone[2] <= tx + TILE and drone[3] <= ty + TILE
                cut = not inside and drone[2] > tx and drone[0] < tx + TILE and drone[3] > ty and drone[1] < ty + TILE
                if cut:
                    continue
                stem = f"fp_boson{s}_{i:06d}_{k}"
                cv2.imwrite(os.path.join(a.addon, "train", "images", stem + ".jpg"), im[ty:ty + TILE, tx:tx + TILE],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                line = ""
                if inside:
                    line = (f"{UAV} {((drone[0] + drone[2]) / 2 - tx) / TILE:.6f} {((drone[1] + drone[3]) / 2 - ty) / TILE:.6f} "
                            f"{(drone[2] - drone[0]) / TILE:.6f} {(drone[3] - drone[1]) / TILE:.6f}\n")
                    with_drone += 1
                open(os.path.join(a.addon, "train", "labels", stem + ".txt"), "w").write(line)
                classes[x.get("cls", "?")] = classes.get(x.get("cls", "?"), 0) + 1
                written += 1
            i += 1
    print(f"mined {written} false-positive tiles from {n_all} confirmed day frames ({with_drone} also contain the "
          f"labelled drone); FP classes as predicted: {classes}")


if __name__ == "__main__":
    main()

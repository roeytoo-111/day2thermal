r"""
Day-camera (RGB) dataset addon for Fixed_wing_v3 (Roboflow layout: 640x640 tiles, names [bird, airplane, uav]) built from
the Boson Oct-06 4K day videos, using the thermal camera as the independent witness.

  train/images, train/labels   tiles cut around drones that the thermal model CONFIRMED (status 'pos' in
                               build_boson_labels.py: RGB detection + thermal detection on the same object). Class
                               uav = 2. One tile per distinct day frame, thinned (--step distinct frames apart);
                               the tile position is random with the box well inside it. Other objects in the tile
                               stay unlabelled (a bird next to the drone would be a missing label; rare).
  test/images, test/labels     same for the held-out sessions (never train on these)
  review/tiles/*.jpg + review/tiles.csv
                               tiles around RGB detections the thermal model did NOT confirm ('hard'), in the
                               thermal field of view. Each is either a thermal miss (a TP for the day model) or an RGB
                               false alarm (a hard negative -- "FPs are great"). A human decides with
                               review_rgb_tiles.py; nothing here is auto-labelled.
  data.yaml                    train = [Fixed_wing_v3/train, this train]; val = Fixed_wing_v3/valid (unchanged)

    python3 src/data/build_rgb_addon.py --labels-dir data/boson_work/labels --day-dir data/boson_work/mp4 \
        --v3 ../ANN-Detection/data/datasets/Fixed_wing_v3 --holdout 08_58_15,10_15_59 --out data/boson_work/Fixed_wing_v3_boson
"""
import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd

TILE, MARGIN, UAV = 640, 32, 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--day-dir", required=True)
    p.add_argument("--v3", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout", default="")
    p.add_argument("--step", type=int, default=3, help="keep every N-th distinct day frame (10 fps -> ~3 fps)")
    p.add_argument("--hard-step", type=int, default=6)
    p.add_argument("--hard-max", type=int, default=400, help="review tiles per run, spread over sessions")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def place_tile(box, W, H, rng):
    x0, y0, x1, y1 = box
    if x1 - x0 > TILE - 2 * MARGIN or y1 - y0 > TILE - 2 * MARGIN:
        return None
    lo_x, hi_x = max(0, int(x1) + MARGIN - TILE), min(W - TILE, int(x0) - MARGIN)
    lo_y, hi_y = max(0, int(y1) + MARGIN - TILE), min(H - TILE, int(y0) - MARGIN)
    if lo_x > hi_x or lo_y > hi_y:
        return None
    return int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1))


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    hold = {s for s in a.holdout.split(",") if s}
    os.makedirs(os.path.join(a.out, "review", "tiles"), exist_ok=True)
    review_rows, counts = [], {"train": 0, "test": 0}
    files = [f for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))) if not os.path.basename(f).startswith("hard_")]
    n_sess_hard = max(len(files) - len(hold), 1)
    for f in files:
        s = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        split = "test" if s in hold else "train"
        pos = d[(d.status == "pos") & d.dx0.notna()].drop_duplicates("day_frame")
        pos = pos.iloc[::a.step]
        hard = d[(d.status == "hard") & d.dx0.notna()].drop_duplicates("day_frame").iloc[::a.hard_step]
        if split == "train" and len(hard):
            hard = hard.sample(min(len(hard), max(a.hard_max // n_sess_hard, 1)), random_state=a.seed)
        elif split == "test":
            hard = hard.iloc[:0]
        want = {int(r.day_frame): ("pos", r) for r in pos.itertuples()}
        for r in hard.itertuples():
            want.setdefault(int(r.day_frame), ("hard", r))
        if not want:
            continue
        cap = cv2.VideoCapture(os.path.join(a.day_dir, f"{s}_day.mp4"))
        i, last = 0, max(want)
        while i <= last:
            if i in want:
                ok, im = cap.read()
                if not ok:
                    break
                kind, r = want[i]
                H, W = im.shape[:2]
                box = (float(r.dx0), float(r.dy0), float(r.dx1), float(r.dy1))
                pos_xy = place_tile(box, W, H, rng)
                if pos_xy is not None:
                    tx, ty = pos_xy
                    tile = im[ty:ty + TILE, tx:tx + TILE]
                    stem = f"boson{s}_{i:06d}"
                    lx = ((box[0] + box[2]) / 2 - tx) / TILE, ((box[1] + box[3]) / 2 - ty) / TILE
                    lw = (box[2] - box[0]) / TILE, (box[3] - box[1]) / TILE
                    if kind == "pos":
                        for sub in ("images", "labels"):
                            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
                        cv2.imwrite(os.path.join(a.out, split, "images", stem + ".jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        open(os.path.join(a.out, split, "labels", stem + ".txt"), "w").write(
                            f"{UAV} {lx[0]:.6f} {lx[1]:.6f} {lw[0]:.6f} {lw[1]:.6f}\n")
                        counts[split] += 1
                    else:
                        cv2.imwrite(os.path.join(a.out, "review", "tiles", stem + ".jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        review_rows.append({"tile": stem + ".jpg", "session": s, "day_frame": i, "rgb_conf": r.rgb_conf,
                                            "x0": round(box[0] - tx, 1), "y0": round(box[1] - ty, 1),
                                            "x1": round(box[2] - tx, 1), "y1": round(box[3] - ty, 1), "decision": ""})
            else:
                if not cap.grab():
                    break
            i += 1
        print(f"{s}: pos tiles so far {counts}, review tiles {len(review_rows)}", flush=True)
    pd.DataFrame(review_rows).to_csv(os.path.join(a.out, "review", "tiles.csv"), index=False)
    ab = os.path.abspath
    open(os.path.join(a.out, "data.yaml"), "w").write(
        f"train:\n  - {ab(os.path.join(a.v3, 'train', 'images'))}\n  - {ab(os.path.join(a.out, 'train', 'images'))}\n"
        f"val: {ab(os.path.join(a.v3, 'valid', 'images'))}\ntest: {ab(os.path.join(a.out, 'test', 'images'))}\n"
        f"nc: 3\nnames: ['bird', 'airplane', 'uav']\n")
    print(f"done: {counts}, review tiles {len(review_rows)} -> {a.out}")


if __name__ == "__main__":
    main()

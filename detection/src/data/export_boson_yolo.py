r"""
Labels (build_boson_labels.py) -> YOLO images/labels at native thermal resolution (640x512), plus the review queue.

  train/   pos frames (label box, class 1 = thermal-uav) thinned to every --pos-step-th frame, plus clean negatives
           (empty label file) at --neg-step; all of each TRAIN session except its last --val-frac (+ --gap-s gap)
  val/     that last contiguous chunk of every train session (early stopping / model selection)
  test/    HELD-OUT sessions, every --pos-step-th pos frame and --neg-step-th neg frame, pseudo-labels (RGB+thermal
           agreement): NOT ground truth, a sanity check until a human labels them
  hard_<session>.csv   RGB-saw-it-thermal-did-not frames, in review_rgbdet.py format (frame_id, verdict=drone, box)
Overlay-marked frames never appear (build_boson_labels.py already dropped them).

    python3 src/data/export_boson_yolo.py --labels-dir data/boson_work/labels --mp4-dir data/boson_work/mp4 \
        --holdout 08_58_15,10_15_59 --out data/boson_work/yolo_boson
"""
import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--mp4-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout", default="", help="comma-separated sessions used only as test")
    p.add_argument("--pos-step", type=int, default=6, help="keep every N-th thermal frame among positives (30 fps -> 5 fps)")
    p.add_argument("--neg-step", type=int, default=30, help="keep every N-th thermal frame among clean negatives")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--gap-s", type=float, default=3.0)
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def main():
    a = parse_args()
    hold = {s for s in a.holdout.split(",") if s}
    todo = {}      # session -> list of (frame, split, label_line or None)
    counts = {}
    for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))):
        s = os.path.basename(f)[:-4]
        if s.startswith("hard_"):
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        fps = 60.0 if s in ("09_58_37", "10_00_40") else a.fps          # true thermal fps of these two sessions
        if s in hold:
            splits = lambda fr, lo, hi: "test"
        else:
            span = d.frame.max() - d.frame.min()
            cut = d.frame.max() - a.val_frac * span
            gap = a.gap_s * fps
            splits = lambda fr, lo, hi: "val" if fr > cut else ("drop" if fr > cut - gap else "train")
        rows = []
        for r in d.itertuples():
            if r.status == "pos" and (r.frame % a.pos_step == 0):
                kind = "pos"
            elif r.status == "neg" and (r.frame % a.neg_step == 0):
                kind = "neg"
            else:
                continue
            sp = splits(r.frame, d.frame.min(), d.frame.max())
            if sp == "drop":
                continue
            line = None
            if kind == "pos":
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                line = f"1 {(x0 + x1) / 2 / 640:.6f} {(y0 + y1) / 2 / 512:.6f} {(x1 - x0) / 640:.6f} {(y1 - y0) / 512:.6f}"
            rows.append((int(r.frame), sp, line))
            counts[(sp, kind)] = counts.get((sp, kind), 0) + 1
        todo[s] = rows
        h = d[d.status == "hard"]
        if len(h):
            hh = pd.DataFrame({"frame_id": h.frame, "orig_label": "", "verdict": "drone", "x0": h.x0, "y0": h.y0,
                               "x1": h.x1, "y1": h.y1, "source": "rgb_only:" + h.rgb_conf.astype(str), "verified": "",
                               "reviewed": ""})
            hh = hh[hh.frame_id % 3 == 0]                       # 10 fps is plenty for a review queue
            os.makedirs(a.out, exist_ok=True)
            hh.to_csv(os.path.join(a.out, f"hard_{s}.csv"), index=False)
    for s, rows in todo.items():
        want = {fr: (sp, line) for fr, sp, line in rows}
        cap = cv2.VideoCapture(os.path.join(a.mp4_dir, f"{s}_thermal.mp4"))
        i, last = 0, max(want) if want else -1
        while i <= last:
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                sp, line = want[i]
                for sub in ("images", "labels"):
                    os.makedirs(os.path.join(a.out, sp, sub), exist_ok=True)
                stem = f"b{s}_{i:06d}"
                cv2.imwrite(os.path.join(a.out, sp, "images", stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(a.out, sp, "labels", stem + ".txt"), "w") as f:
                    f.write((line + "\n") if line else "")
            i += 1
    print("exported:", {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())})


if __name__ == "__main__":
    main()

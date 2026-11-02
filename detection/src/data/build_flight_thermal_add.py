r"""
Thermal training frames from one labelled flight (tools/labeling work dir: labels.json + thermal/NNNNNN.jpg), "few but good":
  yours   every frame Daniel reviewed: box -> uav, "no drone" -> empty (negative)
  auto    unflagged auto boxes only (flagged-unreviewed = unknown, left out), every --stride-th in time order
  never   bulk-marked "not airborne" frames (labels.json meta bulk_none) unless they carry a box; frames outside --window (not airborne: the drone may be on the ground, so NOT negatives), and auto boxes in
          frames with the green overlay square burned into the thermal (07_38: 936 of 10,646 frames, 839 auto boxes sat
          on the square, not on the drone)
Time split: frames after --val-from go to valid, a --gap before it is dropped. Output YOLO layout, uav = 0 (bird = 1 in
the target thermal sets), stems <name>_<frame6>.

    python3 src/data/build_flight_thermal_add.py --work ~/label_work/f0738_1020 --name f0738 --window 14100 33000 \
        --val-from 30200 --gap 600 --stride 6 --out ../data/thermal_add_f0738
"""
import os
import json
import shutil
import argparse

import cv2
import numpy as np


def green(img):
    return (img[..., 1] > 180) & (img[..., 0] < 100) & (img[..., 2] < 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, nargs=2, required=True); ap.add_argument("--val-from", type=int, required=True)
    ap.add_argument("--gap", type=int, default=600); ap.add_argument("--stride", type=int, default=6)
    a = ap.parse_args()
    work = os.path.expanduser(a.work)
    LJ = json.load(open(os.path.join(work, "labels.json"))); T = LJ["thermal"]
    # "no drone" records inside a bulk "not airborne" range are the bulk mark, not a review (they look the same: no src)
    bulk = [b["frames"] for b in LJ["meta"].get("bulk_none", []) if b["cam"] == "thermal"]
    in_bulk = lambda f: any(b0 <= f <= b1 for b0, b1 in bulk)
    lo, hi = a.window
    mine, auto = [], []
    for k, v in T.items():
        f = int(k)
        if not lo <= f <= hi:
            continue
        reviewed = v.get("src") != "auto" or v.get("reviewed")
        if reviewed and v.get("state") != "box" and in_bulk(f):
            continue
        if reviewed and not v.get("unsure"):
            mine.append(f)
        elif not reviewed and v.get("state") == "box" and not v.get("flags"):
            auto.append(f)
    auto = sorted(auto)[::a.stride]
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    for s in ("train", "valid"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, sub))
    n = {"train_box": 0, "train_empty": 0, "valid_box": 0, "valid_empty": 0, "gap": 0, "overlay_dropped": 0, "missing": 0}
    for f, src in [(f, "yours") for f in mine] + [(f, "auto") for f in auto]:
        p = os.path.join(work, "thermal", f"{f:06d}.jpg")
        if not os.path.exists(p):
            n["missing"] += 1; continue
        if a.val_from - a.gap <= f < a.val_from:
            n["gap"] += 1; continue
        img = cv2.imread(p); H, W = img.shape[:2]
        v = T[str(f)]
        if src == "auto" and green(img).sum() >= 4:
            n["overlay_dropped"] += 1; continue
        split = "valid" if f >= a.val_from else "train"
        stem = f"{a.name}_{f:06d}"
        shutil.copy2(p, os.path.join(a.out, split, "images", stem + ".jpg"))
        line = ""
        if v.get("state") == "box" and v.get("box"):
            x, y, w, h = v["box"]
            line = f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n"
        open(os.path.join(a.out, split, "labels", stem + ".txt"), "w").write(line)
        n[f"{split}_{'box' if line else 'empty'}"] += 1
    json.dump({"args": vars(a), "counts": n, "yours": len(mine), "auto_after_stride": len(auto)}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(n, "| yours", len(mine), "auto after stride", len(auto))


if __name__ == "__main__":
    main()

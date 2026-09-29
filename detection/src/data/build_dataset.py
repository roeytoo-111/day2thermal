"""
Build a YOLO training dataset from a base dataset plus labelled new sessions, with an honest val split
and optional polarity-inversion augmentation.

  train = base train  +  every --add session's train split  (+ inverted copies, if --invert_frac > 0)
  val   = every --add session's val split (contiguous chunk of a recording, held out from its train part);
          the base 'valid' split is NOT used by default: it shares sessions with base train (optimistic
          for model selection).
  test  = base test (unchanged; the real test sets are the 07-08 video and the 07-30 session, scored
          with compute_recall_from_gt.py, never used here)

Polarity inversion (--invert_frac): a fraction of train images whose drones are all BRIGHT get a copy
with intensities inverted (255 - I). In LWIR a drone is bright on cold sky but DARK over sun-heated
terrain (notebook 2026-09-30); every drone in the base set is bright. Inversion is unphysical for the
scene as a whole (sky becomes hot), so it is an ablation, not a default -- real dark examples from the
new sessions come first.

    python3 src/data/build_dataset.py --base data/thermal-1-noleak \
        --add data/new_sessions/yolo_0623 --add data/new_sessions/yolo_0715 \
        --out data/thermal-2-sessions [--invert_frac 0.3] [--oversample 3]
"""

import os
import glob
import json
import shutil
import argparse

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Merge base YOLO dataset + labelled sessions; optional inversion aug.")
    p.add_argument("--base", required=True, help="Base YOLO dataset (train/valid/test + data.yaml).")
    p.add_argument("--add", action="append", default=[], help="Exported session dir (train/, val/), repeatable.")
    p.add_argument("--out", required=True)
    p.add_argument("--oversample", type=int, default=1,
                   help="Copies of each added-session TRAIN image (they are few vs the base set).")
    p.add_argument("--invert_frac", type=float, default=0.0,
                   help="Fraction of eligible train images (bright drones only) to add as inverted copies.")
    p.add_argument("--keep_base_valid", action="store_true", help="Also put base valid/ into val (not recommended).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def pairs(split_dir):
    for img in sorted(glob.glob(os.path.join(split_dir, "images", "*"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        yield img, os.path.join(split_dir, "labels", stem + ".txt")


def put(img, lbl, out_split, name):
    ext = os.path.splitext(img)[1]
    link_or_copy(img, os.path.join(out_split, "images", name + ext))
    if os.path.exists(lbl):
        link_or_copy(lbl, os.path.join(out_split, "labels", name + ".txt"))
    else:
        open(os.path.join(out_split, "labels", name + ".txt"), "w").close()


def all_drones_bright(img_path, lbl_path, uav_class=1):
    """True if the image has >=1 drone box and every drone is brighter than its surround."""
    if not os.path.exists(lbl_path):
        return False
    rows = [l.split() for l in open(lbl_path) if l.strip()]
    rows = [r for r in rows if int(r[0]) == uav_class]
    if not rows:
        return False
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    H, W = g.shape
    for r in rows:
        cx, cy, w, h = float(r[1]) * W, float(r[2]) * H, float(r[3]) * W, float(r[4]) * H
        x0, y0, x1, y1 = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)), int(min(cx + w / 2 + 1, W)), int(min(cy + h / 2 + 1, H))
        m = max(4, int(max(w, h) / 2))
        X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)
        ring = g[Y0:Y1, X0:X1].copy()
        mask = np.ones_like(ring, bool)
        mask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        if mask.sum() == 0 or x1 <= x0 or y1 <= y0:
            return False
        bg = np.median(ring[mask])
        box = g[y0:y1, x0:x1]
        if box.max() - bg < bg - box.min():
            return False
    return True


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; remove it first.")
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, split, sub))
    counts = {"train_base": 0, "train_added": 0, "train_inverted": 0, "valid": 0, "test": 0}

    for img, lbl in pairs(os.path.join(a.base, "train")):
        put(img, lbl, os.path.join(a.out, "train"), os.path.splitext(os.path.basename(img))[0])
        counts["train_base"] += 1
    for img, lbl in pairs(os.path.join(a.base, "test")):
        put(img, lbl, os.path.join(a.out, "test"), os.path.splitext(os.path.basename(img))[0])
        counts["test"] += 1
    if a.keep_base_valid:
        for img, lbl in pairs(os.path.join(a.base, "valid")):
            put(img, lbl, os.path.join(a.out, "valid"), os.path.splitext(os.path.basename(img))[0])
            counts["valid"] += 1

    for d in a.add:
        tag = os.path.basename(os.path.normpath(d))
        for img, lbl in pairs(os.path.join(d, "train")):
            stem = os.path.splitext(os.path.basename(img))[0]
            for k in range(a.oversample):
                put(img, lbl, os.path.join(a.out, "train"), f"{stem}__os{k}" if a.oversample > 1 else stem)
                counts["train_added"] += 1
        for img, lbl in pairs(os.path.join(d, "val")):
            put(img, lbl, os.path.join(a.out, "valid"), os.path.splitext(os.path.basename(img))[0])
            counts["valid"] += 1
        print(f"  added {tag}")

    if a.invert_frac > 0:
        cand = [(img, lbl) for img, lbl in pairs(os.path.join(a.out, "train"))
                if "__inv" not in img and "__os" not in img.replace("__os0", "") and all_drones_bright(img, lbl)]
        pick = rng.choice(len(cand), int(round(a.invert_frac * len(cand))), replace=False) if cand else []
        for i in pick:
            img, lbl = cand[i]
            stem = os.path.splitext(os.path.basename(img))[0]
            g = cv2.imread(img)
            cv2.imwrite(os.path.join(a.out, "train", "images", stem + "__inv.jpg"), 255 - g,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(lbl, os.path.join(a.out, "train", "labels", stem + "__inv.txt"))
            counts["train_inverted"] += 1
        print(f"  inversion: {len(cand)} eligible (all drones bright), {counts['train_inverted']} inverted copies")

    with open(os.path.join(a.base, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update({"train": "../train/images", "val": "../valid/images", "test": "../test/images"})
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(a.out, "build_config.json"), "w") as f:
        json.dump({**vars(a), "counts": counts}, f, indent=2)
    print(f"wrote {a.out}: {counts}")


if __name__ == "__main__":
    main()

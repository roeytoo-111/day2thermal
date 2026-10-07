r"""
Make OLD-camera training images look like the NEW (deployed) thermal camera: crop a window of 1/--factor of the frame
around a drone and upscale it back to the frame size. The 2026-10-06 camera has 2.2x more pixels per degree than the
old one (registration scale 0.301 vs 0.135 thermal px per 4K day px) and its edges are 2-3x softer, which is what an
upscale by ~2.2 produces. Labels move with the crop (a box is kept if >= 60% of it lies in the window, then clipped).
About --neg-frac of the outputs are random windows of images without drones (background statistics).

Skips images whose name starts with any --skip prefix (new-camera Boson frames and their pastes are already native).
Output: <out>/train/{images,labels} (a build_dataset.py --add source).

    python3 src/data/zoomin_aug.py --src data/thermal-3-pasteG --out data/boson_work/zoomin_old --n 2000
"""
import os
import glob
import argparse

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--factor", type=float, default=2.2)
    p.add_argument("--neg-frac", type=float, default=0.2)
    p.add_argument("--skip", default="b08_,b09_,b10_,bgboson")
    p.add_argument("--uav-class", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_labels(p):
    if not os.path.exists(p):
        return []
    return [[float(v) for v in l.split()] for l in open(p) if l.strip()]


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    skip = tuple(s for s in a.skip.split(",") if s)
    imgs = [f for f in sorted(glob.glob(os.path.join(a.src, "train", "images", "*")))
            if not os.path.basename(f).startswith(skip) and "__os" not in os.path.basename(f)]
    pos, neg = [], []
    for f in imgs:
        lab = read_labels(os.path.splitext(f.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt")
        (pos if any(int(l[0]) == a.uav_class for l in lab) else neg).append((f, lab))
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out, "train", sub), exist_ok=True)
    n_neg = int(round(a.n * a.neg_frac))
    plan = [pos[i] for i in rng.integers(0, len(pos), a.n - n_neg)] + \
           ([neg[i] for i in rng.integers(0, len(neg), n_neg)] if neg else [])
    written, kept_boxes = 0, 0
    for k, (f, lab) in enumerate(plan):
        im = cv2.imread(f)
        if im is None:
            continue
        H, W = im.shape[:2]
        w, h = int(round(W / a.factor)), int(round(H / a.factor))
        drones = [l for l in lab if int(l[0]) == a.uav_class]
        if drones:
            c, cx, cy, bw, bh = drones[int(rng.integers(len(drones)))]
            x0, y0, x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H
            if x1 - x0 > w * 0.9 or y1 - y0 > h * 0.9:
                continue                                     # drone too big to zoom into
            lo_x, hi_x = max(0, int(x1) - w + 2), min(W - w, int(x0) - 2)
            lo_y, hi_y = max(0, int(y1) - h + 2), min(H - h, int(y0) - 2)
            if lo_x > hi_x or lo_y > hi_y:
                continue
            ox, oy = int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1))
        else:
            ox, oy = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        crop = cv2.resize(im[oy:oy + h, ox:ox + w], (W, H), interpolation=cv2.INTER_LINEAR)
        lines = []
        for c, cx, cy, bw, bh in lab:
            x0, y0, x1, y1 = (cx - bw / 2) * W - ox, (cy - bh / 2) * H - oy, (cx + bw / 2) * W - ox, (cy + bh / 2) * H - oy
            area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.6 * area:
                continue
            sx, sy = W / w, H / h
            X0, Y0, X1, Y1 = cx0 * sx, cy0 * sy, cx1 * sx, cy1 * sy
            lines.append(f"{int(c)} {(X0 + X1) / 2 / W:.6f} {(Y0 + Y1) / 2 / H:.6f} {(X1 - X0) / W:.6f} {(Y1 - Y0) / H:.6f}")
            kept_boxes += int(c) == a.uav_class
        stem = f"zoomin_{k:05d}__" + os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.out, "train", "images", stem + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(os.path.join(a.out, "train", "labels", stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    print(f"zoom-in x{a.factor}: {written} images ({kept_boxes} drone boxes) from {len(pos)} drone / {len(neg)} empty "
          f"old-camera images -> {a.out}")


if __name__ == "__main__":
    main()

r"""
Dirty-lens augmentation for thermal training images, modelled on the 13_21 flight (2026-10-26), whose lens was dirty:
  veil       a bright veil hugging the frame border (rounded-rectangle profile with a steep edge), with smooth random
             variation (blurred noise at 1/4-1/2 of the frame): +60..+160 grey levels, then the camera's auto-gain
             re-stretch (1st-99.5th percentile), which darkens the centre as in 13_21
  contrast   compression toward the local mean (local contrast x 0.45..0.85)
  smudge     1-3 elliptical patches blended with a strongly blurred copy (sigma 3..8 px)
  blur       slight global blur (sigma 0.4..1.4 px)
13_21 vs 13_18 (same day and terrain, clean lens): median Laplacian std 3.1 vs 11.5; washed-out bright borders with a dark
centre. Labels are copied unchanged; a box is dropped (image kept) when the drone's contrast against its ring falls below
--min-contrast grey levels after the augmentation (it would no longer be visible, so no longer a fair label).

    python3 src/data/aug_dirty_lens.py --src data/thermal-10-M0f0738-u0/train --out ../data/aug_dirtylens --frac 0.3 --seed 0
"""
import os
import glob
import random
import argparse

import cv2
import numpy as np


def dirty(img, rng):
    g = img.astype(np.float32)
    H, W = g.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # rounded-rectangle distance from the centre (superellipse): the 13_21 veil hugs the frame border, dark centre
    q = rng.uniform(2.5, 4.0)
    r = ((np.abs(xx - W / 2) / (W / 2)) ** q + (np.abs(yy - H / 2) / (H / 2)) ** q) ** (1 / q)
    r0 = rng.uniform(0.35, 0.70)
    t = np.clip((r - r0) / (1.0 - r0 + 1e-6), 0, 1); edge = t * t * (3 - 2 * t)
    s = rng.uniform(0.25, 0.5) * max(H, W)
    low = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), s / 3)
    low = (low - low.min()) / (np.ptp(low) + 1e-6)
    veil = rng.uniform(60, 160) * np.clip(0.8 * edge + 0.35 * low * edge + 0.15 * low, 0, 1)
    k = rng.uniform(0.45, 0.85)
    mean = cv2.GaussianBlur(g, (0, 0), 25)
    out = mean + (g - mean) * k + veil[..., None]
    blurred = cv2.GaussianBlur(out, (0, 0), rng.uniform(3, 8))
    for _ in range(int(rng.integers(1, 4))):
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), (int(rng.uniform(40, 160)), int(rng.uniform(30, 120))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 25)[..., None] * rng.uniform(0.5, 1.0)
        out = out * (1 - m) + blurred * m
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.4))
    lo, hi = np.percentile(out, 1), np.percentile(out, 99.5)          # the camera's auto-gain re-stretches the frame:
    out = (out - lo) / max(hi - lo, 1e-6) * 255                        # the bright border pushes the centre darker
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_ok(img, line, min_c):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); H, W = g.shape
    c, cx, cy, w, h = [float(v) for v in line.split()[:5]]
    x0, y0, x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W) + 1, int((cy + h / 2) * H) + 1
    p = max(3, int(0.75 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    reg = g[Y0:Y1, X0:X1]; m = np.ones(reg.shape, bool); m[max(y0 - Y0, 0):y1 - Y0, max(x0 - X0, 0):x1 - X0] = False
    inner = g[max(y0, 0):y1, max(x0, 0):x1]
    if not m.any() or inner.size == 0:
        return True
    med = np.median(reg[m])
    return max(np.percentile(inner, 95) - med, med - np.percentile(inner, 5)) >= min_c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a YOLO split dir with images/ and labels/")
    ap.add_argument("--out", required=True); ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-contrast", type=float, default=12.0)
    ap.add_argument("--prefix", default="dl_")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); random.seed(a.seed)
    imgs = sorted(glob.glob(os.path.join(a.src, "images", "*")))
    pick = random.sample(imgs, int(len(imgs) * a.frac))
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    kept = dropped = 0
    for p in pick:
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            continue
        out = dirty(img, rng)
        lp = os.path.join(a.src, "labels", stem + ".txt")
        lines = [l for l in open(lp)] if os.path.exists(lp) else []
        good = []
        for l in lines:
            if l.strip() and contrast_ok(out, l, a.min_contrast):
                good.append(l); kept += 1
            elif l.strip():
                dropped += 1
        cv2.imwrite(os.path.join(a.out, "images", a.prefix + stem + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(os.path.join(a.out, "labels", a.prefix + stem + ".txt"), "w").write("".join(good))
    print(f"{len(pick)} dirty-lens copies; boxes kept {kept}, dropped as invisible {dropped} -> {a.out}")


if __name__ == "__main__":
    main()

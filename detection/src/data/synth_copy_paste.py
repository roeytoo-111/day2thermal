"""
Option A synthetic data: paste small thermal drone targets onto REAL thermal
backgrounds, with exact boxes. No generative model involved.

Why: in the leak-free training set (thermal-1-noleak) the median drone box is
~40 px (512x512 frames) and only ~9% of boxes are < 16 px, while in the real
deployment video even the drones the model catches have a median of ~13 px
(640x512) -- the misses are smaller still. At 3-20 px an LWIR drone is a
bright, high-contrast blob (measured: 100% bright polarity, contrast ~130-180
grey levels over sky, SNR ~20), which is simple to render faithfully.

Two target sources, both from the SOURCE dataset's training split only (so
nothing from the evaluation video can enter):
  * real crops  -- labelled thermal-uav boxes, cut out with a soft
                   foreground mask and downscaled to the target size
                   (downscaling a large drone gives a physically plausible
                   small blob with real shape structure);
  * parametric  -- a smooth body plus 1-4 hot spots (motors/battery),
                   for shape diversity beyond the few real drone types.

Compositing is done in contrast space: pixel = local_bg + alpha * contrast,
contrast sampled from the measured real distribution for that size, then a
small PSF blur and matched sensor noise inside the pasted region, then JPEG
re-encode like the source images.

Backgrounds are the source training images themselves (existing labels are
kept; pastes avoid existing boxes).

v2 (2026-09-29), after v1 taught "any small bright point = drone" (it fired on
static terrain hot spots; notebook 2026-09-29 section 7):
  * sky-only placement: local roughness (std of a 3-sigma high-pass in the
    paste window + 16 px) <= --max_roughness, and no pre-existing bright speck
    (high-pass > --speck_thresh) in that window. Real terrain hot spots stay in
    the same images UNLABELLED, i.e. act as negatives next to the positives.
  * real crops only (--parametric_frac 0), 1-2 pastes per image,
  * synthetic share of the final train set = --synth_frac (0.25), not 50%.
v1 behaviour: --max_roughness inf --speck_thresh inf --parametric_frac 0.3
--max_targets 3 --n_images <#train>. Output = a full YOLO dataset: original
train + synthetic images in train/, valid/ and test/ untouched (hard links),
plus synth_manifest.csv (every paste's parameters) and a QA contact sheet.

Usage:
    python3 src/data/synth_copy_paste.py --dataset_dir data/thermal-1-noleak \
        --output_dir data/thermal-1-noleak-synthA --n_images 2759
"""

import os
import csv
import json
import glob
import shutil
import argparse

import cv2
import numpy as np
import yaml

SPLITS = ["train", "valid", "test"]


def parse_args():
    p = argparse.ArgumentParser(description="Copy-paste small thermal drones onto real thermal backgrounds.")
    p.add_argument("--dataset_dir", required=True, help="Leak-free source YOLO dataset.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_images", type=int, default=None,
                   help="Synthetic images to add (overrides --synth_frac).")
    p.add_argument("--synth_frac", type=float, default=0.25,
                   help="Synthetic share of the final train set (default 0.25 -> n = frac/(1-frac) * #real).")
    p.add_argument("--max_roughness", type=float, default=3.0,
                   help="Sky-only: max local high-pass std around a paste (real-drone contexts: median 3.6; "
                        "~70%% of random spots are <= 3). 'inf' disables.")
    p.add_argument("--speck_thresh", type=float, default=25.0,
                   help="Reject a spot if the background already has a bright speck (high-pass > this) nearby.")
    p.add_argument("--tag", default="synthA2", help="Filename prefix of synthetic images.")
    p.add_argument("--uav_class", type=int, default=1, help="Class id of the drone class (thermal-uav).")
    p.add_argument("--min_size", type=float, default=3.0, help="Min target size, sqrt(w*h) px at image scale.")
    p.add_argument("--max_size", type=float, default=24.0, help="Max target size, sqrt(w*h) px.")
    p.add_argument("--max_targets", type=int, default=2, help="Pastes per image: uniform 1..this.")
    p.add_argument("--parametric_frac", type=float, default=0.0, help="Fraction of pastes that are parametric blobs.")
    p.add_argument("--min_crop_size", type=float, default=24.0,
                   help="Only real boxes at least this big are used as crop sources (need shape to downscale).")
    p.add_argument("--max_bg_level", type=float, default=200.0,
                   help="Skip placements whose local background median exceeds this (target would be clipped away).")
    p.add_argument("--min_snr", type=float, default=4.0,
                   help="Reject a paste whose achieved peak rise is below this many local-noise std (invisible target).")
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_labels(path):
    if not os.path.exists(path):
        return []
    return [l.split() for l in open(path) if l.strip()]


def yolo_to_xyxy(l, W, H):
    cx, cy, w, h = (float(v) for v in l[1:5])
    return cx * W - w * W / 2, cy * H - h * H / 2, cx * W + w * W / 2, cy * H + h * H / 2


# ------------------------------------------------------------------ targets
def extract_crops(dataset_dir, uav_class, min_crop):
    """Soft-masked drone sprites from labelled boxes: (contrast-normalised mask in [0,1])."""
    sprites = []
    for lbl in sorted(glob.glob(os.path.join(dataset_dir, "train", "labels", "*.txt"))):
        rows = [l for l in read_labels(lbl) if int(l[0]) == uav_class]
        if not rows:
            continue
        stem = os.path.splitext(os.path.basename(lbl))[0]
        imgs = glob.glob(os.path.join(dataset_dir, "train", "images", stem + ".*"))
        if not imgs:
            continue
        g = cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        H, W = g.shape
        for l in rows:
            x0, y0, x1, y1 = yolo_to_xyxy(l, W, H)
            if np.sqrt((x1 - x0) * (y1 - y0)) < min_crop:
                continue
            x0, y0 = max(int(x0), 0), max(int(y0), 0)
            x1, y1 = min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)
            pad = max(4, int(0.25 * max(x1 - x0, y1 - y0)))
            X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
            patch = g[Y0:Y1, X0:X1]
            ring = np.ones_like(patch, bool)
            ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
            if ring.sum() < 10:
                continue
            bg = np.median(patch[ring])
            box = patch[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
            peak = box.max() - bg
            if peak < 25:          # low-contrast / dark-polarity box: not a clean hot-target sprite
                continue
            # soft foreground: how far above local background, relative to the peak
            m = np.clip((box - bg) / peak, 0, 1)
            # an isolated target fades out before the box edge; a bright region that runs off the
            # edges is a saturated background patch or overlay, not a drone
            edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
            if (edge > 0.5).mean() > 0.3:
                continue
            m[m < 0.15] = 0
            if m.sum() < 4:
                continue
            ys, xs = np.nonzero(m)
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            sprites.append(m.astype(np.float32))
    return sprites


def parametric_sprite(rng, res=64):
    """Smooth elongated body + 1-4 hot spots; returns mask in [0,1] at res x res."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) - res / 2
    th = rng.uniform(0, np.pi)
    c, s = np.cos(th), np.sin(th)
    u, v = c * xx + s * yy, -s * xx + c * yy
    a = res * rng.uniform(0.18, 0.35)
    b = a * rng.uniform(0.3, 1.0)
    m = np.exp(-(u / a) ** 2 - (v / b) ** 2) * rng.uniform(0.35, 0.8)
    for _ in range(rng.integers(1, 5)):
        px, py = rng.normal(0, a * 0.6), rng.normal(0, a * 0.6)
        r = res * rng.uniform(0.04, 0.10)
        m += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * r * r))
    m = np.clip(m / m.max(), 0, 1)
    m[m < 0.1] = 0
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def sample_size(rng, lo, hi):
    """Log-uniform in [lo, hi]: equal weight per octave, so 3-6 px gets as much as 12-24 px."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_contrast(rng, size):
    """Grey levels above local background; from the measured noleak distribution
    (small targets: median ~130-180, occasionally faint)."""
    if rng.random() < 0.2:
        return float(rng.uniform(40, 80))       # faint tail -- hard examples
    return float(np.clip(rng.normal(150 if size < 16 else 140, 35), 60, 230))


# ------------------------------------------------------------------ compositing
def resize_sprite(m, size, rng):
    h, w = m.shape
    aspect = w / h * rng.uniform(0.8, 1.25)
    tw = max(1, int(round(size * np.sqrt(aspect))))
    th = max(1, int(round(size / np.sqrt(aspect))))
    out = cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        out = np.rot90(out, k)
    return np.ascontiguousarray(out)


def sky_ok(hp, x0, y0, x1, y1, max_rough, speck, margin=16):
    """Smooth, speck-free neighbourhood (hp = image minus its 3-sigma blur)."""
    H, W = hp.shape
    w = hp[max(y0 - margin, 0):min(y1 + margin, H), max(x0 - margin, 0):min(x1 + margin, W)]
    return float(w.std()) <= max_rough and float(w.max()) <= speck


def boxes_overlap(a, b, margin):
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def paste(img, m, contrast, x, y, rng, min_snr):
    """Composite sprite m at (x, y) top-left in contrast space; returns tight box, or None (img untouched)
    when the target would not be visible: achieved peak rise < max(20 levels, min_snr * local noise)."""
    H, W = img.shape
    h, w = m.shape
    pad = 3
    X0, Y0, X1, Y1 = x - pad, y - pad, x + w + pad, y + h + pad
    if X0 < 0 or Y0 < 0 or X1 > W or Y1 > H:
        return None
    region = img[Y0:Y1, X0:X1].copy()
    alpha = np.zeros_like(region)
    alpha[pad:pad + h, pad:pad + w] = m
    sigma = rng.uniform(0.4, 0.9)                                   # optics PSF
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    hf = region - cv2.GaussianBlur(region, (0, 0), 2)
    noise = rng.normal(0, max(hf.std(), 0.5), region.shape).astype(np.float32)
    out = np.clip(region + alpha * contrast + (alpha > 0.05) * noise * 0.5, 0, 255)
    rise = float((out - region).max())
    ys, xs = np.nonzero(alpha > 0.2)
    if len(xs) == 0 or rise < max(20.0, min_snr * max(hf.std(), 0.5)):
        return None
    img[Y0:Y1, X0:X1] = out
    # 1 px margin, like a human-drawn box around a small target
    return (max(X0 + xs.min() - 1, 0), max(Y0 + ys.min() - 1, 0), min(X0 + xs.max() + 2, W), min(Y0 + ys.max() + 2, H))


def contact_sheet(samples, path, tile=160):
    tiles = []
    for img, boxes in samples:
        v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for b in boxes:
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            half = 24
            x0, y0 = int(np.clip(cx - half, 0, v.shape[1] - 2 * half)), int(np.clip(cy - half, 0, v.shape[0] - 2 * half))
            crop = v[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
            crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_NEAREST)
            s = tile / (2 * half)
            cv2.rectangle(crop, (int((b[0] - x0) * s) - 2, int((b[1] - y0) * s) - 2),
                          (int((b[2] - x0) * s) + 2, int((b[3] - y0) * s) + 2), (0, 200, 255), 1)
            tiles.append(crop)
            if len(tiles) >= 48:
                break
        if len(tiles) >= 48:
            break
    while len(tiles) % 8:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(path, np.vstack(rows))


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    if os.path.exists(a.output_dir):
        raise SystemExit(f"{a.output_dir} already exists; remove it first.")

    sprites = extract_crops(a.dataset_dir, a.uav_class, a.min_crop_size)
    print(f"Real drone sprites extracted: {len(sprites)}")
    if not sprites:
        raise SystemExit("No usable drone crops found.")

    # copy the source dataset (hard links)
    for split in SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.output_dir, split, sub))
            for f in glob.glob(os.path.join(a.dataset_dir, split, sub, "*")):
                link_or_copy(f, os.path.join(a.output_dir, split, sub, os.path.basename(f)))
    with open(os.path.join(a.dataset_dir, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(a.output_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    bgs = sorted(glob.glob(os.path.join(a.dataset_dir, "train", "images", "*")))
    n = a.n_images or int(round(a.synth_frac / (1 - a.synth_frac) * len(bgs)))
    order = rng.permutation(np.resize(np.arange(len(bgs)), n))
    manifest = open(os.path.join(a.output_dir, "synth_manifest.csv"), "w", newline="")
    mw = csv.writer(manifest)
    mw.writerow(["out_image", "background", "source", "sprite_id", "size", "contrast", "bg_level", "x0", "y0", "x1", "y1"])
    qa, n_pastes, n_skipped = [], 0, 0

    for i, bi in enumerate(order):
        bg_path = bgs[bi]
        stem = os.path.splitext(os.path.basename(bg_path))[0]
        img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        H, W = img.shape
        labels = read_labels(os.path.join(a.dataset_dir, "train", "labels", stem + ".txt"))
        occupied = [yolo_to_xyxy(l, W, H) for l in labels]
        hp = img - cv2.GaussianBlur(img, (0, 0), 3)
        new_boxes, new_rows = [], []
        for _ in range(int(rng.integers(1, a.max_targets + 1))):
            size = sample_size(rng, a.min_size, a.max_size)
            if rng.random() < a.parametric_frac:
                src, sid, m = "parametric", -1, parametric_sprite(rng)
            else:
                sid = int(rng.integers(len(sprites)))
                src, m = "real_crop", sprites[sid]
            m = resize_sprite(m, size, rng)
            contrast = sample_contrast(rng, size)
            for _attempt in range(60):
                x = int(rng.integers(4, W - m.shape[1] - 4))
                y = int(rng.integers(4, H - m.shape[0] - 4))
                cand = (x, y, x + m.shape[1], y + m.shape[0])
                if any(boxes_overlap(cand, o, 6) for o in occupied):
                    continue
                if not sky_ok(hp, *cand, a.max_roughness, a.speck_thresh):
                    continue
                bg_level = float(np.median(img[max(y - 8, 0):y + m.shape[0] + 8, max(x - 8, 0):x + m.shape[1] + 8]))
                if bg_level > a.max_bg_level:
                    continue
                box = paste(img, m, contrast, x, y, rng, a.min_snr)
                if box is None:
                    continue
                occupied.append(box)
                new_boxes.append(box)
                new_rows.append([src, sid, round(size, 2), round(contrast, 1), round(bg_level, 1), *box])
                break
            else:
                n_skipped += 1
        if not new_boxes:
            continue
        out_stem = f"{a.tag}_{i:06d}__{stem}"
        out_u8 = img.round().astype(np.uint8)
        cv2.imwrite(os.path.join(a.output_dir, "train", "images", out_stem + ".jpg"),
                    cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        with open(os.path.join(a.output_dir, "train", "labels", out_stem + ".txt"), "w") as f:
            for l in labels:
                f.write(" ".join(l) + "\n")
            for b in new_boxes:
                f.write(f"{a.uav_class} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                        f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n")
        for r in new_rows:
            mw.writerow([out_stem + ".jpg", os.path.basename(bg_path), *r])
        n_pastes += len(new_boxes)
        if len(qa) < 24:
            qa.append((out_u8, new_boxes))

    manifest.close()
    with open(os.path.join(a.output_dir, "synth_config.json"), "w") as f:
        json.dump({**vars(a), "n_real_train": len(bgs), "n_synthetic_target": n,
                   "n_pasted": n_pastes, "n_skipped": n_skipped}, f, indent=2, default=str)
    contact_sheet(qa, os.path.join(a.output_dir, "synth_qa_contact_sheet.png"))
    n_train = len(os.listdir(os.path.join(a.output_dir, "train", "images")))
    print(f"Wrote {a.output_dir}: {n_train} train images ({n_train - len(bgs)} synthetic, {n_pastes} pasted drones, "
          f"{n_skipped} pastes skipped for lack of a valid spot).")


if __name__ == "__main__":
    main()

"""Spatially align RGB frames to the thermal camera geometry and produce the
train/val folder structure consumed by training.

Design choices
--------------
* RGB is warped into the *thermal* frame ("rgb2thermal"): thermal pixels are
  never resampled, so radiometric values stay untouched.
* For a rigidly mounted camera pair a single homography is valid for the whole
  recording. For scenes at large distance (air-to-air UAV footage, ground seen
  from altitude) parallax is negligible and one homography is essentially
  exact; for close-range ground scenes expect residual parallax and prefer a
  crop that excludes the near field.
* Train/val split is done in contiguous temporal chunks, otherwise adjacent
  (nearly identical) video frames leak between train and val.

Modes
-----
manual   Recommended. Provide >= 4 point correspondences in a JSON file:
             {"rgb": [[x, y], ...], "thermal": [[x, y], ...]}
         Use ``--dump-pair N`` first to export a raw pair for point picking.
ecc      Automatic: maximizes ECC between Sobel edge maps (cross-modality
         friendly), affine model, over the frames in --calib-frames.
identity Cameras already aligned (or same optical axis): pure scale/letterbox.

Example
-------
python -m day2thermal.register --raw data/flight01/raw --out data/flight01/aligned \
    --mode manual --points calib/flight01_points.json --auto-crop --val-frac 0.1
"""
import argparse
import os
import sys

import cv2
import numpy as np

from .utils import (ensure_dir, list_images, load_json, read_rgb,
                    read_thermal, save_json)


def to_gray8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def edge_map(gray8):
    g = cv2.GaussianBlur(gray8.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    m -= m.min()
    mx = m.max()
    if mx > 0:
        m /= mx
    return m


def scale_letterbox_H(rgb_wh, th_wh):
    rw, rh = rgb_wh
    tw, th_ = th_wh
    s = min(tw / rw, th_ / rh)
    return np.array([[s, 0, (tw - s * rw) / 2],
                     [0, s, (th_ - s * rh) / 2],
                     [0, 0, 1]], np.float32)


def estimate_manual(points_json):
    d = load_json(points_json)
    src = np.array(d["rgb"], np.float32)
    dst = np.array(d["thermal"], np.float32)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need >= 4 matching points in both lists")
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("homography estimation failed - check your points")
    return H.astype(np.float32)


def estimate_ecc(rgb_img, th_img, iterations=300):
    tg = edge_map(to_gray8(th_img))
    rg = edge_map(to_gray8(rgb_img))
    th_h, th_w = tg.shape
    r_h, r_w = rg.shape
    init = scale_letterbox_H((r_w, r_h), (th_w, th_h))
    warp = init[:2].copy()
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6)
    cc, warp = cv2.findTransformECC(tg, rg, warp, cv2.MOTION_AFFINE, crit,
                                    None, 5)
    H = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
    return H, float(cc)


def valid_crop_from_H(H, rgb_wh, th_wh, margin=2):
    """Axis-aligned rectangle inside the warped-RGB footprint, intersected
    with the thermal frame. Assumes mild rotation (rigid rigs)."""
    w, h = rgb_wh
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       np.float32).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x0 = max(0, int(np.ceil(max(wc[0, 0], wc[3, 0]))) + margin)
    x1 = min(th_wh[0], int(np.floor(min(wc[1, 0], wc[2, 0]))) - margin)
    y0 = max(0, int(np.ceil(max(wc[0, 1], wc[1, 1]))) + margin)
    y1 = min(th_wh[1], int(np.floor(min(wc[2, 1], wc[3, 1]))) - margin)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="dir containing rgb/ and thermal/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["manual", "ecc", "identity"], default="ecc")
    ap.add_argument("--points", help="JSON with point correspondences (manual mode)")
    ap.add_argument("--calib-frames", default="0",
                    help="comma-separated frame indices used for ECC")
    ap.add_argument("--crop", help="x,y,w,h crop in thermal coordinates")
    ap.add_argument("--auto-crop", action="store_true",
                    help="crop to the valid (fully overlapping) region")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=200,
                    help="frames per temporal chunk for the train/val split")
    ap.add_argument("--dump-pair", type=int, default=None,
                    help="export raw pair N for manual point picking, then exit")
    args = ap.parse_args()

    rgb_dir = os.path.join(args.raw, "rgb")
    th_dir = os.path.join(args.raw, "thermal")
    names = sorted(set(list_images(rgb_dir)) & set(list_images(th_dir)))
    if not names:
        sys.exit(f"no paired frames found under {args.raw}")

    if args.dump_pair is not None:
        i = min(args.dump_pair, len(names) - 1)
        prev = ensure_dir(os.path.join(args.out, "calib_preview"))
        for sub, d in (("rgb", rgb_dir), ("thermal", th_dir)):
            img = cv2.imread(os.path.join(d, names[i]), cv2.IMREAD_UNCHANGED)
            if img.ndim == 2 and img.dtype == np.uint16:
                img = to_gray8(img)
            cv2.imwrite(os.path.join(prev, f"{sub}_{names[i]}"), img)
        print(f"wrote calibration pair {names[i]} to {prev}\n"
              f"Pick >= 4 corresponding points (x, y) in each image and store "
              f"them as JSON: {{\"rgb\": [[x,y],...], \"thermal\": [[x,y],...]}}")
        return

    first_th = read_thermal(os.path.join(th_dir, names[0]))
    th_h, th_w = first_th.shape
    first_rgb = read_rgb(os.path.join(rgb_dir, names[0]))
    r_h, r_w = first_rgb.shape[:2]

    cc = None
    if args.mode == "manual":
        if not args.points:
            sys.exit("--mode manual requires --points")
        H = estimate_manual(args.points)
    elif args.mode == "identity":
        H = scale_letterbox_H((r_w, r_h), (th_w, th_h))
    else:  # ecc
        best = None
        for idx in [int(i) for i in args.calib_frames.split(",")]:
            idx = min(idx, len(names) - 1)
            rgb_i = read_rgb(os.path.join(rgb_dir, names[idx]))
            th_i = read_thermal(os.path.join(th_dir, names[idx]))
            try:
                H_i, cc_i = estimate_ecc(rgb_i, th_i)
            except cv2.error as e:
                print(f"[ecc] frame {idx} failed: {e}")
                continue
            print(f"[ecc] frame {idx}: correlation {cc_i:.4f}")
            if best is None or cc_i > best[1]:
                best = (H_i, cc_i)
        if best is None:
            sys.exit("ECC failed on all calibration frames. Cross-modality "
                     "auto-alignment is fragile; please use --mode manual.")
        H, cc = best

    crop = None
    if args.crop:
        crop = [int(v) for v in args.crop.split(",")]
    elif args.auto_crop:
        crop = valid_crop_from_H(H, (r_w, r_h), (th_w, th_h))
        if crop is None:
            print("[warn] auto-crop found no valid rectangle; keeping full frame")

    # temporal-chunk split
    nch = max(1, (len(names) + args.chunk - 1) // args.chunk)
    n_val = max(1, int(round(args.val_frac * nch))) if args.val_frac > 0 else 0
    val_chunks = set(np.linspace(0, nch - 1, n_val, dtype=int)) if n_val else set()

    for split in ("train", "val"):
        ensure_dir(os.path.join(args.out, split, "rgb"))
        ensure_dir(os.path.join(args.out, split, "thermal"))

    counts = {"train": 0, "val": 0}
    for i, name in enumerate(names):
        split = "val" if (i // args.chunk) in val_chunks else "train"
        rgb = read_rgb(os.path.join(rgb_dir, name))
        th = cv2.imread(os.path.join(th_dir, name), cv2.IMREAD_UNCHANGED)
        if th.ndim == 3:
            th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
        rgb_w = cv2.warpPerspective(rgb, H, (th_w, th_h), flags=cv2.INTER_LINEAR)
        if crop:
            x, y, w, h = crop
            rgb_w = rgb_w[y:y + h, x:x + w]
            th = th[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(args.out, split, "rgb", name), rgb_w)
        cv2.imwrite(os.path.join(args.out, split, "thermal", name), th)
        counts[split] += 1

    save_json({"mode": args.mode, "H": H.tolist(), "ecc_cc": cc,
               "crop": crop, "thermal_size": [th_w, th_h],
               "rgb_size": [r_w, r_h]},
              os.path.join(args.out, "registration.json"))
    print(f"aligned {counts['train']} train / {counts['val']} val pairs -> {args.out}")


if __name__ == "__main__":
    main()

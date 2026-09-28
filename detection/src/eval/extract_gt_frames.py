"""
Extract frames from the ground-truth thermal video and compute the same
domain-signal stats (sharpness, brightness, resolution) used in
audit_thermal_data.py, so training-data classes can be compared directly
against what the real deployment footage actually looks like.

If you already have YOLO-format labels for (some of) the extracted frames,
point --labels_dir at them and this will also compute box-based stats
(area-ratio, size buckets) to compare against thermal-uav in the 17k set.

If you pass --annotations_csv and --domain_csv (the two CSVs
audit_thermal_data.py writes), this will print a side-by-side comparison
against the thermal-uav class automatically.

Usage:
    # Sample every 10th frame (default) -- fast, good for a distribution check
    python3 extract_gt_frames.py --video_path data/videos/thermal.mp4 \
        --output_dir gt_audit_out \
        --annotations_csv audit_out/annotations_audit.csv \
        --domain_csv audit_out/image_domain_stats.csv

    # Extract every frame (needed if you want to align with an existing
    # per-frame label/tracker output that's indexed by original frame number)
    python3 extract_gt_frames.py --video_path data/videos/thermal.mp4 \
        --output_dir gt_audit_out --extract_all
"""

import os
import glob
import argparse
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def parse_args():
    p = argparse.ArgumentParser(description="Extract + audit ground-truth video frames against the training set.")
    p.add_argument("--video_path", required=True, help="Path to the ground-truth video (e.g. data/videos/thermal.mp4).")
    p.add_argument("--output_dir", default="./gt_audit_out", help="Where to write extracted frames + CSVs.")
    p.add_argument("--sample_every", type=int, default=10,
                    help="Keep every Nth frame (default: 10, ~2.5k frames from a 24.5k-frame video). Ignored if --extract_all.")
    p.add_argument("--extract_all", action="store_true",
                    help="Extract every frame, sequentially numbered from 1. Use this if you need to align with "
                         "existing per-frame labels/tracker output indexed by original frame number.")
    p.add_argument("--img_ext", default="png", choices=["png", "jpg"], help="Extracted frame format (default: png).")
    p.add_argument("--labels_dir", default=None,
                    help="Optional dir of YOLO-format .txt labels, one per extracted frame (same base filename), "
                         "single class assumed (uav). Enables box-based stats (area-ratio, size bucket).")
    p.add_argument("--annotations_csv", default=None,
                    help="Optional: annotations_audit.csv from audit_thermal_data.py, to compare against thermal-uav.")
    p.add_argument("--domain_csv", default=None,
                    help="Optional: image_domain_stats.csv from audit_thermal_data.py, to compare against thermal-uav.")
    return p.parse_args()


def extract_frames(video_path, frames_dir, sample_every, extract_all, img_ext):
    os.makedirs(frames_dir, exist_ok=True)
    if extract_all:
        vf = None
    else:
        vf = f"select=not(mod(n\\,{sample_every}))"

    out_pattern = os.path.join(frames_dir, f"frame_%06d.{img_ext}")
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if vf:
        cmd += ["-vf", vf, "-vsync", "vfr"]
    else:
        cmd += ["-vsync", "0"]
    cmd += ["-q:v", "2", out_pattern]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError("ffmpeg extraction failed -- see stderr above.")

    frames = sorted(glob.glob(os.path.join(frames_dir, f"frame_*.{img_ext}")))
    print(f"Extracted {len(frames)} frames to {frames_dir}")
    return frames


def compute_domain_stats(img_path):
    """Same metric set as audit_thermal_data.py's compute_domain_stats, kept
    identical so the two CSVs are directly comparable."""
    try:
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            arr = np.asarray(im_rgb).astype(np.float32)
            w, h = im_rgb.size
    except Exception:
        return None

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_divergence = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0
    brightness_mean = float(arr.mean())
    brightness_std = float(arr.std())
    gray_flat = np.asarray(im_rgb.convert("L")).astype(np.float32)
    percentiles = np.percentile(gray_flat, [1, 5, 50, 95, 99])

    sharpness = None
    noise_floor = None
    if _HAS_CV2:
        gray = np.asarray(Image.open(img_path).convert("L")).astype(np.uint8)
        sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        h_, w_ = gray.shape
        bh, bw = max(1, h_ // 8), max(1, w_ // 8)
        best_var = None
        for by in range(8):
            for bx in range(8):
                block = gray[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                if block.size == 0:
                    continue
                v = float(block.std())
                if best_var is None or v < best_var:
                    best_var = v
        noise_floor = round(best_var, 3) if best_var is not None else None

    return {
        "width": w, "height": h,
        "aspect_ratio": round(w / h, 3) if h else None,
        "channel_divergence": round(channel_divergence, 3),
        "brightness_mean": round(brightness_mean, 2),
        "brightness_std": round(brightness_std, 2),
        "sharpness": sharpness,
        "noise_floor_flattest_block": noise_floor,
        "p1": round(float(percentiles[0]), 2),
        "p5": round(float(percentiles[1]), 2),
        "p50": round(float(percentiles[2]), 2),
        "p95": round(float(percentiles[3]), 2),
        "p99": round(float(percentiles[4]), 2),
    }


def load_yolo_box_stats(frame_path, labels_dir, img_w, img_h):
    """If a matching label file exists, return per-box area-ratio + size bucket rows."""
    if not labels_dir:
        return []
    base = os.path.splitext(os.path.basename(frame_path))[0]
    label_path = os.path.join(labels_dir, base + ".txt")
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return []
    rows = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 or not parts[0].isdigit():
                continue
            norm_w, norm_h = float(parts[3]), float(parts[4])
            w_px, h_px = norm_w * img_w, norm_h * img_h
            rows.append({
                "frame": base,
                "bbox_w_px": w_px, "bbox_h_px": h_px,
                "bbox_area_ratio": norm_w * norm_h,
            })
    return rows


def report_gt_stats(domain_df, box_df):
    print("\n" + "=" * 74)
    print("           GROUND-TRUTH VIDEO — FRAME-LEVEL DOMAIN STATS")
    print("=" * 74)
    cols = [c for c in ["sharpness", "brightness_mean", "channel_divergence"] if c in domain_df.columns]
    summary = domain_df[cols].median().round(2)
    print("Median per-frame stats:")
    print(summary.to_string())
    print(f"\nResolution: {domain_df['width'].median():.0f} x {domain_df['height'].median():.0f} (median)")

    if not box_df.empty:
        print("\n" + "=" * 74)
        print("           GROUND-TRUTH VIDEO — BOX STATS (from --labels_dir)")
        print("=" * 74)
        area_pct = (box_df["bbox_area_ratio"] * 100)
        print(f"Boxes found: {len(box_df)} across {box_df['frame'].nunique()} frames")
        print(f"Box area as %% of frame -- median: {area_pct.median():.3f}%%  "
              f"mean: {area_pct.mean():.3f}%%  p90: {area_pct.quantile(0.9):.3f}%%")
        print(f"Box width (px) -- median: {box_df['bbox_w_px'].median():.1f}  "
              f"mean: {box_df['bbox_w_px'].mean():.1f}")
    else:
        print("\n(No box stats -- pass --labels_dir with matching YOLO .txt files per frame to add this.)")


def report_comparison_vs_thermal_uav(domain_df, box_df, annotations_csv, domain_csv):
    if not annotations_csv or not domain_csv:
        return
    if not os.path.exists(annotations_csv) or not os.path.exists(domain_csv):
        print(f"\n(Comparison skipped -- couldn't find {annotations_csv} / {domain_csv})")
        return

    ann = pd.read_csv(annotations_csv)
    dom = pd.read_csv(domain_csv)

    uav_images = ann[ann["class_name"] == "thermal-uav"]["image_path"].unique()
    uav_dom = dom[dom["image_path"].isin(uav_images)]

    if uav_dom.empty:
        print("\n(Comparison skipped -- no thermal-uav rows found in the provided CSVs.)")
        return

    print("\n" + "=" * 74)
    print("      TRAINING SET (thermal-uav) vs. GROUND-TRUTH VIDEO — COMPARISON")
    print("=" * 74)
    rows = []
    for col in ["sharpness", "brightness_mean", "noise_floor_flattest_block", "p1", "p5", "p50", "p95", "p99"]:
        if col in uav_dom.columns and col in domain_df.columns:
            rows.append({
                "metric": col,
                "thermal-uav (17k set) median": round(uav_dom[col].median(), 2),
                "GT video median": round(domain_df[col].median(), 2),
            })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nnoise_floor_flattest_block: std within the flattest 8x8-grid block -- disentangles sensor")
        print("noise character from scene-content-driven sharpness. p1/p5/p50/p95/p99: full intensity")
        print("distribution shape -- two sources can share a mean/std while differing a lot here, e.g. if")
        print("one clips/saturates at temperature extremes and the other doesn't.")

    uav_ann = ann[ann["class_name"] == "thermal-uav"]
    if "bbox_area_px" in uav_ann.columns and not box_df.empty:
        uav_img_dims = dom.set_index("image_path")[["width", "height"]]
        merged = uav_ann.merge(uav_img_dims, on="image_path", how="left")
        merged = merged.dropna(subset=["width", "height"])
        uav_area_ratio = (merged["bbox_w_px"] * merged["bbox_h_px"]) / (merged["width"] * merged["height"]) * 100
        gt_area_ratio = box_df["bbox_area_ratio"] * 100
        print(f"\nBox area %% of frame -- thermal-uav (17k) median: {uav_area_ratio.median():.3f}%%  "
              f"vs GT video median: {gt_area_ratio.median():.3f}%%")
        if gt_area_ratio.median() < uav_area_ratio.median() * 0.5:
            print("  ⚠ GT video targets are noticeably SMALLER than the training set's thermal-uav boxes -- "
                  "the training data may be under-representing the hardest (smallest) real targets.")


def main():
    args = parse_args()
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    frames = extract_frames(args.video_path, frames_dir, args.sample_every, args.extract_all, args.img_ext)

    if not frames:
        print("No frames extracted -- nothing to do.")
        return

    domain_records = []
    box_records = []
    for fpath in frames:
        stats = compute_domain_stats(fpath)
        if stats is None:
            continue
        domain_records.append({"frame": os.path.splitext(os.path.basename(fpath))[0], "image_path": fpath, **stats})
        box_records.extend(load_yolo_box_stats(fpath, args.labels_dir, stats["width"], stats["height"]))

    domain_df = pd.DataFrame(domain_records)
    box_df = pd.DataFrame(box_records)

    domain_csv_out = os.path.join(args.output_dir, "gt_frame_domain_stats.csv")
    domain_df.to_csv(domain_csv_out, index=False)
    print(f"\nSaved: {domain_csv_out}")
    if not box_df.empty:
        box_csv_out = os.path.join(args.output_dir, "gt_frame_box_stats.csv")
        box_df.to_csv(box_csv_out, index=False)
        print(f"Saved: {box_csv_out}")

    report_gt_stats(domain_df, box_df)
    report_comparison_vs_thermal_uav(domain_df, box_df, args.annotations_csv, args.domain_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
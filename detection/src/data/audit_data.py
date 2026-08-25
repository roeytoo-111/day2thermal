"""
Audit the thermal UAV detection YOLO dataset.

Adds three things beyond a standard class-balance audit, specifically because
this dataset mixes at least one non-thermal source (class "0") with real IR
frames ("bird", "thermal-uav"):

  1. Per-image dimensions are read from disk, not assumed fixed — mixed
     sources almost certainly have different resolutions, and using a wrong
     fixed size silently corrupts every bbox pixel-size calculation.
  2. A "domain signal" table per class: grayscale-ness (R≈G≈B, expected for
     a real single-channel IR sensor saved as 3-channel), brightness stats,
     and resolution. This gives a numeric handle on which classes are likely
     contaminated with non-thermal imagery, instead of eyeballing samples.
  3. Contact-sheet PNGs (grid of sampled images with boxes drawn) per class,
     saved to disk. No GUI / interactive loop needed.

Usage:
    python3 audit_thermal_data.py --dataset_dir ./data/thermal-7eu6d
    python3 audit_thermal_data.py --dataset_dir ./data/thermal-7eu6d --no-visual
"""

import os
import glob
import random
import argparse

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

IGNORED_LABEL_FILES = {"README.roboflow.txt", "classes.txt", "labels.txt"}
SPLITS = ["train", "valid", "test"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the thermal UAV YOLO dataset: splits, classes, bbox size buckets, domain signal, spot checks."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                         help="Path to dataset dir containing data.yaml and train/valid/test subdirs.")
    parser.add_argument("--sample_size", type=int, default=25,
                         help="Number of images per class in each contact sheet (default: 25).")
    parser.add_argument("--grid_cols", type=int, default=5,
                         help="Columns in the contact sheet grid (default: 5).")
    parser.add_argument("--thumb_size", type=int, default=220,
                         help="Thumbnail size (px, square) in the contact sheet (default: 220).")
    parser.add_argument("--output_dir", type=str, default="./audit_out",
                         help="Where to write CSVs and contact sheets (default: ./audit_out).")
    parser.add_argument("--no-visual", action="store_true",
                         help="Skip contact-sheet generation (CSV/console reports only).")
    parser.add_argument("--iou_threshold", type=float, default=0.4,
                         help="IoU above which two boxes in the same image are flagged as overlapping/duplicate (default: 0.4).")
    parser.add_argument("--run_ocr", action="store_true",
                         help="Run OCR on box crops to flag boxes that are likely text/watermarks rather than drones. "
                              "Requires pytesseract + system tesseract-ocr (+ tesseract-ocr-ara for Arabic).")
    parser.add_argument("--ocr_max_per_class", type=int, default=1000,
                         help="Cap on how many boxes per class get OCR'd, sampled randomly (default: 1000). OCR is slow; raise if you want full coverage.")
    parser.add_argument("--ocr_lang", type=str, default="ara+eng",
                         help="Tesseract language string (default: 'ara+eng'). Add more with '+', e.g. 'ara+eng+chi_sim'.")
    parser.add_argument("--ocr_min_box_px", type=int, default=14,
                         help="Skip OCR on boxes narrower than this many px — too small to contain legible text (default: 14).")
    return parser.parse_args()


def load_dataset_config(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Warning: '{yaml_path}' not found. Using raw class_id as name.")
        return {}
    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
    raw_names = data_config.get("names", {})
    if isinstance(raw_names, list):
        return {i: name for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        return {int(k): v for k, v in raw_names.items()}
    return {}


def find_matching_image(dataset_dir, split, base_name):
    valid_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    images_dir = os.path.join(dataset_dir, split, "images")
    for ext in valid_exts:
        potential_img = os.path.join(images_dir, base_name + ext)
        if os.path.exists(potential_img):
            return potential_img
    return None


_image_size_cache = {}
_domain_stats_cache = {}


def get_image_size(img_path):
    """Real (width, height) from disk, cached."""
    if img_path in _image_size_cache:
        return _image_size_cache[img_path]
    if not img_path or not os.path.exists(img_path):
        _image_size_cache[img_path] = None
        return None
    try:
        with Image.open(img_path) as im:
            size = im.size  # (w, h)
    except Exception:
        size = None
    _image_size_cache[img_path] = size
    return size


def compute_domain_stats(img_path):
    """
    Cheap per-image signal to separate real single-channel IR sensor output
    (near-identical R/G/B, narrow dynamic range) from color media/RGB photos
    (channels diverge, sharper/wider dynamic range, often larger/odd resolutions).
    """
    if img_path in _domain_stats_cache:
        return _domain_stats_cache[img_path]
    if not img_path or not os.path.exists(img_path):
        _domain_stats_cache[img_path] = None
        return None
    try:
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            arr = np.asarray(im_rgb).astype(np.float32)
            w, h = im_rgb.size
    except Exception:
        _domain_stats_cache[img_path] = None
        return None

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Mean absolute channel divergence: ~0 for true grayscale sensor data,
    # meaningfully >0 for color photos. NOTE: if the dataset was already
    # normalized to grayscale before export (as this one was), this will be
    # ~0 for every class and is NOT a usable domain signal on its own —
    # kept here for completeness, but see sharpness below instead.
    channel_divergence = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0
    brightness_mean = float(arr.mean())
    brightness_std = float(arr.std())
    gray_flat = np.asarray(im.convert("L")).astype(np.float32)
    percentiles = np.percentile(gray_flat, [1, 5, 50, 95, 99])

    # Laplacian variance: a standard blur/sharpness proxy. Real long-range IR
    # sensor frames of small distant targets tend to be low-detail/blurry;
    # close-up media/propaganda photos tend to be sharp and texture-rich even
    # after grayscale conversion and downsizing. Higher = sharper/more detail.
    sharpness = None
    noise_floor = None
    if _HAS_CV2:
        gray = np.asarray(im.convert("L")).astype(np.uint8)
        sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        # Noise floor: std within the FLATTEST 8x8 grid block, not the whole
        # image. Disentangles "sensor noise character" from "lots of real
        # scene edges" -- a clean synthetic render and a noisy real sensor
        # frame can have similar Laplacian variance overall while differing
        # a lot in flat-region noise specifically.
        h, w = gray.shape
        bh, bw = max(1, h // 8), max(1, w // 8)
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

    stats = {
        "width": w,
        "height": h,
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
    _domain_stats_cache[img_path] = stats
    return stats


def parse_yolo_annotations(dataset_dir):
    records = []
    image_domain_records = []

    for split in SPLITS:
        labels_path = os.path.join(dataset_dir, split, "labels", "*.txt")
        label_files = glob.glob(labels_path)

        for file_path in label_files:
            file_name = os.path.basename(file_path)
            if file_name in IGNORED_LABEL_FILES or file_name.startswith("README"):
                continue

            base_name = os.path.splitext(file_name)[0]
            img_path = find_matching_image(dataset_dir, split, base_name)
            img_size = get_image_size(img_path)
            img_w, img_h = img_size if img_size else (None, None)

            dstats = compute_domain_stats(img_path)
            if dstats is not None:
                image_domain_records.append({
                    "split": split, "file_name": file_name, "image_path": img_path,
                    **dstats,
                })

            if os.path.getsize(file_path) == 0:
                records.append({
                    "split": split, "file_name": file_name, "image_path": img_path,
                    "class_id": -1, "img_w": img_w, "img_h": img_h,
                    "bbox_w_px": 0.0, "bbox_h_px": 0.0, "bbox_area_px": 0.0,
                    "is_background": True,
                })
                continue

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5 or not parts[0].isdigit():
                        continue
                    class_id = int(parts[0])
                    norm_w, norm_h = float(parts[3]), float(parts[4])

                    cx, cy = float(parts[1]), float(parts[2])

                    if img_w is None:
                        # Image missing/unreadable: keep the row but flag pixel
                        # sizes as unknown rather than silently guessing wrong.
                        w_px = h_px = area_px = area_ratio = None
                    else:
                        w_px = norm_w * img_w
                        h_px = norm_h * img_h
                        area_px = w_px * h_px
                        area_ratio = (norm_w * norm_h)  # already a fraction of full image area

                    records.append({
                        "split": split, "file_name": file_name, "image_path": img_path,
                        "class_id": class_id, "img_w": img_w, "img_h": img_h,
                        "cx": cx, "cy": cy, "norm_w": norm_w, "norm_h": norm_h,
                        "bbox_w_px": w_px, "bbox_h_px": h_px, "bbox_area_px": area_px,
                        "bbox_area_ratio": area_ratio,
                        "is_background": False,
                    })

    return pd.DataFrame(records), pd.DataFrame(image_domain_records)


def report_overall_distribution(df):
    total_unique_images = df["image_path"].nunique()
    print("\n" + "=" * 74)
    print("                     OVERALL DATA DISTRIBUTION")
    print("=" * 74)
    print(f"{'Class Name':<15} | {'BBox Count':<12} | {'BBox %':<8} | {'Unique Images':<13}")
    print("-" * 60)
    counts_bbox = df["class_name"].value_counts()
    for name in counts_bbox.index:
        cnt = counts_bbox[name]
        pct = (cnt / len(df)) * 100
        img_cnt = df[df["class_name"] == name]["image_path"].nunique()
        print(f"{str(name):<15} | {cnt:<12} | {pct:.2f}%   | {img_cnt:<13}")
    print("-" * 60)
    print(f"{'Total':<15} | {len(df):<12} | 100.00%  | {total_unique_images:<13}")


def report_split_breakdown(df):
    print("\n" + "=" * 74)
    print("                  DATASET SPLIT BREAKDOWN BY CLASS")
    print("=" * 74)
    for name in df["class_name"].unique():
        class_df = df[df["class_name"] == name]
        class_total_bboxes = len(class_df)
        class_total_images = class_df["image_path"].nunique()
        print(f"Class: [{str(name).upper()}]")
        print(f"  -> {class_total_bboxes} BBoxes across {class_total_images} unique images")
        for split in SPLITS:
            split_df = class_df[class_df["split"] == split]
            split_bboxes = len(split_df)
            split_images = split_df["image_path"].nunique()
            bbox_pct = (split_bboxes / class_total_bboxes * 100) if class_total_bboxes else 0
            img_pct = (split_images / class_total_images * 100) if class_total_images else 0
            print(f"  - {split:<6} : {split_bboxes:<5} BBoxes ({bbox_pct:.1f}%) | {split_images:<5} Images ({img_pct:.1f}%)")
        print("-" * 55)


def get_stride_bucket(w_px):
    if w_px is None:
        return "Unknown (image size unavailable)"
    if w_px < 16:
        return "Small (<16px width)   [Stride 8]"
    elif w_px <= 64:
        return "Medium (16-64px width) [Stride 16]"
    else:
        return "Large (>64px width)   [Stride 32]"


def report_target_size_analysis(df):
    print("\n" + "=" * 74)
    print("           TARGET SIZE STRIDE-LEVEL ANALYSIS (ALL CLASSES)")
    print("=" * 74)
    box_df = df[(~df["is_background"]) & (df["class_id"] != -1)].copy()
    for class_name in sorted(box_df["class_name"].unique(), key=str):
        cls_df = box_df[box_df["class_name"] == class_name].copy()
        total = len(cls_df)
        if total == 0:
            continue
        print(f"\n--- Class: [{str(class_name).upper()}]  (Total boxes: {total}) ---")
        cls_df["stride_bucket"] = cls_df["bbox_w_px"].apply(get_stride_bucket)
        bucket_counts = cls_df["stride_bucket"].value_counts()
        for bucket_name, b_cnt in bucket_counts.items():
            b_pct = (b_cnt / total) * 100
            sub = cls_df[cls_df["stride_bucket"] == bucket_name]
            avg_w = sub["bbox_w_px"].mean()
            avg_h = sub["bbox_h_px"].mean()
            avg_area = sub["bbox_area_px"].mean()
            print(f"  * {bucket_name:<38} count={b_cnt:<6} ({b_pct:5.1f}%)  "
                  f"mean {avg_w:.1f}x{avg_h:.1f}px  area={avg_area:.0f}px^2")


def report_domain_signal(df, image_domain_df):
    """
    Flag classes whose images look statistically different from the rest —
    this is the check aimed squarely at surfacing a class-0-style contamination.
    """
    print("\n" + "=" * 74)
    print("        DOMAIN SIGNAL BY CLASS (grayscale-ness / resolution)")
    print("=" * 74)
    print("channel_divergence ~0  => real single-channel IR sensor frame")
    print("channel_divergence high => likely a color RGB/media image, not thermal\n")

    # Map each image to the set of classes present in it (an image can have
    # multiple boxes/classes — attribute domain stats to every class it touches).
    img_to_classes = (
        df[df["class_id"] != -1]
        .groupby("image_path")["class_name"]
        .apply(lambda s: set(s))
    )

    rows = []
    for _, row in image_domain_df.iterrows():
        classes = img_to_classes.get(row["image_path"], set())
        for c in classes:
            rows.append({**row.to_dict(), "class_name": c})
    per_class_domain = pd.DataFrame(rows)

    if per_class_domain.empty:
        print("No class-attributable images found for domain analysis.")
        return

    agg_dict = {
        "n_images": ("image_path", "nunique"),
        "median_channel_divergence": ("channel_divergence", "median"),
        "median_width": ("width", "median"),
        "median_height": ("height", "median"),
        "median_brightness": ("brightness_mean", "median"),
    }
    if "sharpness" in per_class_domain.columns and per_class_domain["sharpness"].notna().any():
        agg_dict["median_sharpness"] = ("sharpness", "median")
    summary = per_class_domain.groupby("class_name").agg(**agg_dict).round(2)
    print(summary.to_string())

    overall_median_div = per_class_domain["channel_divergence"].median()
    print(f"\nDataset-wide median channel_divergence: {overall_median_div:.2f}")
    if per_class_domain["channel_divergence"].max() < 1e-6:
        print("  (All classes are ~0 — this dataset was already normalized to grayscale before export.")
        print("   channel_divergence can't distinguish real IR vs. converted media photos here; see sharpness/box-area-ratio instead.)")
    for cname, row in summary.iterrows():
        if row["median_channel_divergence"] > max(2.0, overall_median_div * 3):
            print(f"  ⚠ Class '{cname}' looks like COLOR imagery (divergence "
                  f"{row['median_channel_divergence']:.2f} vs dataset median "
                  f"{overall_median_div:.2f}) — likely non-thermal source, review before training.")

    if "median_sharpness" in summary.columns:
        overall_median_sharp = per_class_domain["sharpness"].median()
        print(f"\nDataset-wide median sharpness (Laplacian variance): {overall_median_sharp:.2f}")
        for cname, row in summary.iterrows():
            if row["median_sharpness"] > overall_median_sharp * 2.5:
                print(f"  ⚠ Class '{cname}' is much SHARPER than the rest ({row['median_sharpness']:.2f} vs "
                      f"median {overall_median_sharp:.2f}) — consistent with close-up photography rather than a "
                      f"long-range sensor blob. Review contact sheet for this class.")


def report_box_area_ratio(df):
    """
    What fraction of the frame each box covers, per class — a composition
    signal that's already visible in the size-bucket table but worth calling
    out directly: a real long-range sensor target is a small fraction of the
    frame; a close-up photo's subject fills much of it.
    """
    print("\n" + "=" * 74)
    print("             BOX-TO-IMAGE AREA RATIO BY CLASS (composition check)")
    print("=" * 74)
    box_df = df[(~df["is_background"]) & (df["class_id"] != -1) & df["bbox_area_ratio"].notna()]
    if box_df.empty:
        print("No boxes with area-ratio data available.")
        return
    summary = box_df.groupby("class_name")["bbox_area_ratio"].agg(
        median="median", mean="mean", p90=lambda s: s.quantile(0.9)
    )
    summary = (summary * 100).round(2)  # as % of frame area
    summary.columns = ["median_%_of_frame", "mean_%_of_frame", "p90_%_of_frame"]
    print(summary.to_string())
    print("\n(Values are the box's area as a % of the full frame. A media/propaganda")
    print(" close-up will skew much higher here than a real distant sensor target.)")


def compute_iou(box_a, box_b):
    """boxes are (cx, cy, w, h) normalized."""
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def find_overlapping_boxes(df, output_dir, iou_threshold=0.4):
    """
    Flag pairs of boxes in the same image whose IoU exceeds the threshold —
    candidates for duplicate/redundant labels to manually prune.
    """
    print("\n" + "=" * 74)
    print(f"          OVERLAPPING BOX CHECK (IoU > {iou_threshold})")
    print("=" * 74)

    box_df = df[(~df["is_background"]) & (df["class_id"] != -1)]
    flagged = []

    for (split, file_name), group in box_df.groupby(["split", "file_name"]):
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                iou = compute_iou(
                    (a["cx"], a["cy"], a["norm_w"], a["norm_h"]),
                    (b["cx"], b["cy"], b["norm_w"], b["norm_h"]),
                )
                if iou >= iou_threshold:
                    flagged.append({
                        "split": split, "file_name": file_name, "image_path": a["image_path"],
                        "class_a": a["class_name"], "class_b": b["class_name"], "iou": round(iou, 3),
                    })

    flagged_df = pd.DataFrame(flagged)
    n_images_flagged = flagged_df["image_path"].nunique() if not flagged_df.empty else 0
    print(f"Flagged {len(flagged_df)} overlapping box pairs across {n_images_flagged} images.")
    if not flagged_df.empty:
        print("\nBreakdown by class pair:")
        print(flagged_df.groupby(["class_a", "class_b"]).size().sort_values(ascending=False).to_string())
        out_path = os.path.join(output_dir, "flagged_overlapping_boxes.csv")
        flagged_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}  (review manually before deciding what to drop)")
    return flagged_df


def run_ocr_flagging(df, output_dir, lang, max_per_class, min_box_px):
    """
    OCR each box crop (not the whole image) to flag boxes that are likely
    text/watermarks mislabeled as a drone/bird detection. Off by default —
    opt in with --run_ocr since it needs pytesseract + the tesseract binary.
    """
    print("\n" + "=" * 74)
    print("                    OCR-IN-BOX CHECK")
    print("=" * 74)
    try:
        import pytesseract
    except ImportError:
        print("pytesseract not installed — skipping. Install with:")
        print("  pip install pytesseract")
        print("  apt-get install tesseract-ocr tesseract-ocr-ara   # (Debian/Ubuntu; add ara for Arabic)")
        return None

    box_df = df[
        (~df["is_background"]) & (df["class_id"] != -1)
        & df["bbox_w_px"].notna() & (df["bbox_w_px"] >= min_box_px)
    ]

    sampled_parts = []
    for cname, group in box_df.groupby("class_name"):
        n = min(max_per_class, len(group))
        sampled_parts.append(group.sample(n=n, random_state=42))
    sample_df = pd.concat(sampled_parts) if sampled_parts else box_df.iloc[0:0]

    print(f"Running OCR on {len(sample_df)} box crops (lang='{lang}', "
          f"capped at {max_per_class}/class, skipping boxes <{min_box_px}px wide)...")

    flagged = []
    for _, row in sample_df.iterrows():
        img_path = row["image_path"]
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")
                w, h = im.size
                x1 = max(0, int((row["cx"] - row["norm_w"] / 2) * w))
                y1 = max(0, int((row["cy"] - row["norm_h"] / 2) * h))
                x2 = min(w, int((row["cx"] + row["norm_w"] / 2) * w))
                y2 = min(h, int((row["cy"] + row["norm_h"] / 2) * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = im.crop((x1, y1, x2, y2))
                # Upscale small crops — OCR needs a decent pixel size to work.
                scale = max(1, 200 // max(crop.width, 1))
                if scale > 1:
                    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
                text = pytesseract.image_to_string(crop, lang=lang).strip()
        except Exception:
            continue

        if len(text) >= 2:
            flagged.append({
                "split": row["split"], "file_name": row["file_name"], "image_path": img_path,
                "class_name": row["class_name"], "ocr_text": text.replace("\n", " ")[:80],
            })

    flagged_df = pd.DataFrame(flagged)
    print(f"\nFlagged {len(flagged_df)} / {len(sample_df)} sampled boxes with detectable text.")
    if not flagged_df.empty:
        print("\nText-hit rate by class (of sampled boxes):")
        rate = flagged_df.groupby("class_name").size() / sample_df.groupby("class_name").size()
        print((rate.fillna(0) * 100).round(1).astype(str).add(" %").to_string())
        out_path = os.path.join(output_dir, "flagged_text_boxes.csv")
        flagged_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}  (review manually — some hits will be false positives on noise)")
    return flagged_df


def compute_auc(pos_vals, neg_vals):
    """Rank-based AUC (Mann-Whitney U), no sklearn dependency."""
    values = pd.concat([pos_vals.reset_index(drop=True), neg_vals.reset_index(drop=True)], ignore_index=True)
    ranks = values.rank(method="average")
    n_pos, n_neg = len(pos_vals), len(neg_vals)
    sum_ranks_pos = ranks[:n_pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def report_shortcut_risk(df, image_domain_df):
    """
    Can whole-image style features ALONE (no target info) separate thermal-uav
    frames from bird/background frames? High separability = the model could
    learn to key off scene style (e.g. 'sharp cityscape' or 'bright sky')
    rather than the actual drone signature -- a shortcut/spurious-cue risk.
    """
    print("\n" + "=" * 74)
    print("      SHORTCUT-FEATURE RISK CHECK (can the model 'cheat' on style?)")
    print("=" * 74)
    print("AUC ~0.5 = feature alone can't separate classes (good).")
    print("AUC near 0 or 1 = highly separable by that feature alone (shortcut risk).\n")

    img_to_classes = (
        df[df["class_id"] != -1]
        .groupby("image_path")["class_name"]
        .apply(lambda s: set(s))
    )
    stats = image_domain_df.drop_duplicates(subset=["image_path"]).set_index("image_path")

    pos_paths = [p for p, cls in img_to_classes.items() if "thermal-uav" in cls and p in stats.index]
    neg_paths = [p for p, cls in img_to_classes.items() if "thermal-uav" not in cls and p in stats.index]
    bg_paths = df[df["is_background"]]["image_path"].dropna().unique().tolist()
    neg_paths = list(set(neg_paths) | (set(bg_paths) & set(stats.index)))

    if len(pos_paths) < 5 or len(neg_paths) < 5:
        print("Not enough positive/negative images to compute this check.")
        return

    for feature in ["sharpness", "brightness_mean"]:
        if feature not in stats.columns or stats[feature].isna().all():
            continue
        pos_vals = stats.loc[pos_paths, feature].dropna()
        neg_vals = stats.loc[neg_paths, feature].dropna()
        if len(pos_vals) < 5 or len(neg_vals) < 5:
            continue
        auc = compute_auc(pos_vals, neg_vals)
        risk = max(auc, 1 - auc)
        flag = "  ⚠ HIGH — plausible shortcut risk, investigate before training" if risk > 0.75 else ""
        print(f"  {feature:<18} thermal-uav vs bird/background AUC = {auc:.3f}  "
              f"(n_pos={len(pos_vals)}, n_neg={len(neg_vals)}){flag}")


def draw_boxes_and_thumbnail(img_path, boxes, thumb_size):
    """Load image, draw normalized YOLO boxes, return a square thumbnail."""
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (thumb_size, thumb_size), (40, 40, 40))
        return im
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for (cx, cy, bw, bh) in boxes:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 128), width=max(2, w // 200))
    im.thumbnail((thumb_size, thumb_size))
    canvas = Image.new("RGB", (thumb_size, thumb_size), (20, 20, 20))
    canvas.paste(im, ((thumb_size - im.width) // 2, (thumb_size - im.height) // 2))
    return canvas


def build_label_boxes(label_path):
    boxes = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 or not parts[0].isdigit():
                continue
            boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def generate_contact_sheets(df, dataset_dir, output_dir, sample_size, grid_cols, thumb_size):
    print("\n" + "=" * 74)
    print("                    GENERATING CONTACT SHEETS")
    print("=" * 74)
    os.makedirs(output_dir, exist_ok=True)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for class_name in sorted(df["class_name"].unique(), key=str):
        class_df = df[df["class_name"] == class_name]
        unique_rows = class_df.drop_duplicates(subset=["image_path"])
        unique_rows = unique_rows[unique_rows["image_path"].notna()]
        if unique_rows.empty:
            continue
        n = min(sample_size, len(unique_rows))
        sampled = unique_rows.sample(n=n, random_state=42)

        thumbs = []
        for _, row in sampled.iterrows():
            img_path = row["image_path"]
            split = row["split"]
            base_name = os.path.splitext(row["file_name"])[0]
            label_path = os.path.join(dataset_dir, split, "labels", base_name + ".txt")
            boxes = build_label_boxes(label_path)
            thumb = draw_boxes_and_thumbnail(img_path, boxes, thumb_size)
            thumbs.append(thumb)

        rows_needed = (len(thumbs) + grid_cols - 1) // grid_cols
        sheet = Image.new("RGB", (grid_cols * thumb_size, rows_needed * thumb_size), (10, 10, 10))
        for i, thumb in enumerate(thumbs):
            r, c = divmod(i, grid_cols)
            sheet.paste(thumb, (c * thumb_size, r * thumb_size))

        safe_name = str(class_name).replace("/", "_")
        out_path = os.path.join(output_dir, f"contact_sheet_{safe_name}.png")
        sheet.save(out_path)
        print(f"  Saved {out_path}  ({n} samples)")


def main():
    args = parse_args()
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = load_dataset_config(args.dataset_dir)
    df, image_domain_df = parse_yolo_annotations(args.dataset_dir)

    if df.empty:
        print(f"No valid annotations found in: {args.dataset_dir}")
        return

    df["class_name"] = df["class_id"].map(class_names).fillna(df["class_id"].astype(str))
    df.loc[df["is_background"], "class_name"] = "background"

    report_overall_distribution(df)
    report_split_breakdown(df)
    report_target_size_analysis(df)
    report_box_area_ratio(df)
    report_domain_signal(df, image_domain_df)
    report_shortcut_risk(df, image_domain_df)
    find_overlapping_boxes(df, args.output_dir, iou_threshold=args.iou_threshold)

    if args.run_ocr:
        run_ocr_flagging(
            df, args.output_dir, lang=args.ocr_lang,
            max_per_class=args.ocr_max_per_class, min_box_px=args.ocr_min_box_px,
        )
    else:
        print("\n(Skipping OCR-in-box check — pass --run_ocr to flag boxes that are likely text/watermarks.)")

    ann_csv = os.path.join(args.output_dir, "annotations_audit.csv")
    domain_csv = os.path.join(args.output_dir, "image_domain_stats.csv")
    df.to_csv(ann_csv, index=False)
    image_domain_df.to_csv(domain_csv, index=False)
    print(f"\nSaved: {ann_csv}")
    print(f"Saved: {domain_csv}")

    if not args.no_visual:
        generate_contact_sheets(
            df, args.dataset_dir, args.output_dir,
            args.sample_size, args.grid_cols, args.thumb_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
"""
Extract stratified sample crops from the CLEANED RGB detection track
(clean_detections.py's cleaned_track.csv / rejected_jumps.csv), producing a
manifest.csv compatible with label_review.py -- so you go straight into the
existing labeling tool after this runs.

This checks a DIFFERENT thing than show_CNN_detections.py's accepted/rejected:
that tool validates the two-stage classifier's own confirmed/rejected
decisions (stage1_conf/stage2_prob gate). This script validates whether the
MOTION-CONSISTENCY-FILTERED track (what we'd actually use as pseudo-ground-
truth for thermal eval) is trustworthy. It specifically over-samples:
  - the low-confidence static segment around frames 12-2000 flagged earlier
    (not just whatever a uniform random sample happens to catch)
  - a general stratified sample of accepted points by track_status
  - a sample of rejected points by reject_reason, to confirm they really
    are noise and not real fast maneuvers being wrongly dropped

Usage:
    python3 detection/data/extract_track_crops_for_review.py \
        --video data/videos/2026-07-08T11_30_44_day_short.mp4 \
        --cleaned_track data/gt_audit_out/cleaned_track.csv \
        --rejected_jumps data/gt_audit_out/rejected_jumps.csv \
        --out_dir data/track_review \
        --per_bucket 25

Then:
    python3 label_review.py --audit_dir track_review
"""

import os
import ast
import argparse

import cv2
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--cleaned_track", required=True)
    p.add_argument("--rejected_jumps", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--per_bucket", type=int, default=25, help="Max crops sampled per stratification bucket.")
    p.add_argument("--margin", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focus_buckets", default=None,
                    help="Comma-separated review_bucket names to sample this round (e.g. "
                         "'reacquired_after_gap,reacquired_after_outliers,rejected_position_jump,rejected_size_jump'). "
                         "If set, ONLY these buckets are sampled -- use this to top up weak buckets without "
                         "re-touching ones that already have enough labels. Omit to sample all buckets as before.")
    p.add_argument("--exclude_existing", action="store_true",
                    help="If out_dir/manifest.csv already exists, exclude its frame_ids from this round's sampling "
                         "(avoids duplicate crops) and APPEND new rows to it instead of overwriting.")
    return p.parse_args()


def parse_bbox(bbox_str):
    if isinstance(bbox_str, (list, tuple)):
        return [int(v) for v in bbox_str]
    return [int(v) for v in ast.literal_eval(bbox_str)]


def sample_stratified(df, group_col, per_bucket, seed):
    parts = []
    for _, g in df.groupby(group_col):
        n = min(per_bucket, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts) if parts else df.iloc[0:0]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    existing_frame_ids = set()
    existing_manifest_path = os.path.join(args.out_dir, "manifest.csv")
    if args.exclude_existing and os.path.exists(existing_manifest_path):
        existing_df = pd.read_csv(existing_manifest_path)
        existing_frame_ids = set(existing_df["frame_id"].tolist())
        print(f"Excluding {len(existing_frame_ids)} already-sampled frame_ids from {existing_manifest_path}")

    acc = pd.read_csv(args.cleaned_track)
    rej = pd.read_csv(args.rejected_jumps)
    if existing_frame_ids:
        acc = acc[~acc["frame_id"].isin(existing_frame_ids)]
        rej = rej[~rej["frame_id"].isin(existing_frame_ids)]

    focus = set(b.strip() for b in args.focus_buckets.split(",")) if args.focus_buckets else None

    acc_by_status = sample_stratified(acc, "track_status", args.per_bucket, args.seed).copy()
    acc_by_status["review_bucket"] = acc_by_status["track_status"]
    if focus is not None:
        acc_by_status = acc_by_status[acc_by_status["review_bucket"].isin(focus)]

    if focus is None or "flagged_low_conf_segment" in focus:
        low_conf_segment = acc[(acc.frame_id >= 12) & (acc.frame_id < 2000)]
        n_low_conf = min(args.per_bucket, len(low_conf_segment))
        low_conf_sample = (low_conf_segment.sample(n=n_low_conf, random_state=args.seed)
                            if n_low_conf else low_conf_segment).copy()
        low_conf_sample["review_bucket"] = "flagged_low_conf_segment"
    else:
        low_conf_sample = acc.iloc[0:0].copy()
        low_conf_sample["review_bucket"] = []

    accepted_sample = pd.concat([acc_by_status, low_conf_sample]).drop_duplicates(subset=["frame_id"])
    accepted_sample["review_status"] = "accepted"

    rejected_sample = sample_stratified(rej, "reject_reason", args.per_bucket, args.seed).copy()
    rejected_sample["review_bucket"] = "rejected_" + rejected_sample["reject_reason"]
    if focus is not None:
        rejected_sample = rejected_sample[rejected_sample["review_bucket"].isin(focus)]
    rejected_sample["review_status"] = "rejected"

    combined = pd.concat([accepted_sample, rejected_sample], ignore_index=True)
    print(f"Sampled {len(accepted_sample)} accepted + {len(rejected_sample)} rejected = {len(combined)} crops to review.")
    print("\nAccepted breakdown:")
    print(accepted_sample["review_bucket"].value_counts().to_string())
    print("\nRejected breakdown:")
    print(rejected_sample["review_bucket"].value_counts().to_string())

    frame_to_rows = {}
    for _, row in combined.iterrows():
        frame_to_rows.setdefault(int(row["frame_id"]), []).append(row)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    manifest_rows = []
    current_frame_id = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_id += 1
        if current_frame_id not in frame_to_rows:
            continue

        h, w = frame.shape[:2]
        for row in frame_to_rows[current_frame_id]:
            x1, y1, x2, y2 = parse_bbox(row["bbox"])
            cx1, cy1 = max(0, x1 - args.margin), max(0, y1 - args.margin)
            cx2, cy2 = min(w, x2 + args.margin), min(h, y2 + args.margin)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            fname = f"frame_{current_frame_id:06d}_{row['review_status']}_{row['review_bucket']}.png"
            cv2.imwrite(os.path.join(args.out_dir, fname), crop)
            saved += 1

            manifest_rows.append({
                "filename": fname,
                "frame_id": current_frame_id,
                "conf": row.get("stage1_conf", ""),
                "bbox": [x1, y1, x2, y2],
                # Reusing label_review.py's "gate_status" field for our own
                # bucket labels -- its by-gate TP/FP summary breakdown then
                # directly answers "how trustworthy is each bucket" for free.
                "gate_status": f"{row['review_status']}_{row['review_bucket']}",
                "label": "",
            })

    cap.release()

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    new_manifest_df = pd.DataFrame(manifest_rows)
    if args.exclude_existing and os.path.exists(manifest_path):
        combined_manifest = pd.concat([pd.read_csv(manifest_path), new_manifest_df], ignore_index=True)
        combined_manifest.to_csv(manifest_path, index=False)
        print(f"\nAppended {len(new_manifest_df)} new rows to existing {manifest_path} "
              f"({len(combined_manifest)} total).")
    else:
        new_manifest_df.to_csv(manifest_path, index=False)
        print(f"\nSaved {saved} crops to {args.out_dir}")
        print(f"Manifest written to {manifest_path}")
    print(f"\nNext: python3 label_review.py --audit_dir {args.out_dir}")


if __name__ == "__main__":
    main()
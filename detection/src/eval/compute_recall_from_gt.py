"""
Compute recall for one or more models against a shared, model-independent
ground-truth sample (from sample_frames_for_recall.py + label_review.py).
Each model is checked separately against the SAME labeled frames -- a fair
paired comparison, and valid because the sample wasn't chosen based on
either model's output.

A detection counts as "catching" a ground-truth positive frame if the model
fired within +/- --tolerance_frames of it (small tolerance for detection
latency / off-by-one frame alignment, not for genuinely different events).

Usage:
    python3 compute_recall_from_ground_truth.py \
        --manifest recall_ground_truth/manifest.csv \
        --json thermal_dets_coco.json:coco \
        --json thermal_dets_transfer.json:rgb_transfer \
        --tolerance_frames 2

    # score only an unseen temporal holdout (frames the training set never covered)
    python3 compute_recall_from_ground_truth.py ... --min_frame 20100
"""

import json
import math
import argparse

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Labeled manifest.csv from the ground-truth sampling round.")
    p.add_argument("--json", action="append", required=True,
                    help="path:name, repeatable -- one per model, e.g. --json dets_coco.json:coco")
    p.add_argument("--tolerance_frames", type=int, default=2,
                    help="A model 'catches' a ground-truth frame if it fired within this many frames of it "
                         "(default: 2 -- small latency tolerance, not for matching different events).")
    p.add_argument("--conf_floor", type=float, default=0.0,
                    help="Minimum confidence for a detection to count (default: 0.0, any logged detection).")
    p.add_argument("--min_frame", type=int, default=None,
                    help="Only score ground-truth frames with frame_id >= this (e.g. a temporal holdout).")
    p.add_argument("--max_frame", type=int, default=None,
                    help="Only score ground-truth frames with frame_id <= this.")
    return p.parse_args()


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval -- stays sensible for small n and rates near 0/1."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load_detected_frames(json_path, conf_floor):
    with open(json_path) as f:
        data = json.load(f)
    frames = set()
    for entry in data:
        for det in entry["detections"]:
            if det["conf"] >= conf_floor:
                frames.add(entry["frame_id"])
                break
    return frames


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.min_frame is not None:
        manifest = manifest[manifest["frame_id"] >= args.min_frame]
    if args.max_frame is not None:
        manifest = manifest[manifest["frame_id"] <= args.max_frame]
    if args.min_frame is not None or args.max_frame is not None:
        print(f"Restricted to frames [{args.min_frame}, {args.max_frame}]: {len(manifest)} ground-truth frames.")

    gt_positive = manifest[manifest["label"].isin(["TP", "TP_loose"])]
    gt_negative = manifest[manifest["label"] == "FP"]
    n_unlabeled = manifest["label"].isna().sum() + (manifest["label"] == "").sum()

    print(f"Ground truth: {len(gt_positive)} positive frames, {len(gt_negative)} negative frames, "
          f"{n_unlabeled} unlabeled (finish labeling before trusting these numbers).")

    if gt_positive.empty:
        print("No positive ground-truth frames labeled yet -- nothing to compute recall against.")
        return

    print(f"\n{'Model':<20} {'Recall':<10} {'95% CI':<18} {'Caught / Total Positive'}")
    print("-" * 70)
    for spec in args.json:
        json_path, name = spec.rsplit(":", 1)
        detected = load_detected_frames(json_path, args.conf_floor)

        caught = 0
        for fid in gt_positive["frame_id"]:
            window = range(fid - args.tolerance_frames, fid + args.tolerance_frames + 1)
            if any(w in detected for w in window):
                caught += 1

        recall = caught / len(gt_positive)
        lo, hi = wilson_ci(caught, len(gt_positive))
        print(f"{name:<20} {recall:<10.2%} {f'[{lo:.1%}, {hi:.1%}]':<18} {caught}/{len(gt_positive)}")

    # False-positive-rate sanity check on the same ground truth, for context
    # (not a substitute for the earlier confidence-bucketed precision work).
    if not gt_negative.empty:
        print(f"\n(For context) false-fire rate on the {len(gt_negative)} confirmed-empty ground-truth frames:")
        for spec in args.json:
            json_path, name = spec.rsplit(":", 1)
            detected = load_detected_frames(json_path, args.conf_floor)
            false_fires = sum(1 for fid in gt_negative["frame_id"] if fid in detected)
            lo, hi = wilson_ci(false_fires, len(gt_negative))
            print(f"  {name:<20} {false_fires}/{len(gt_negative)} empty frames had a detection "
                  f"({false_fires / len(gt_negative):.1%}, 95% CI [{lo:.1%}, {hi:.1%}])")


if __name__ == "__main__":
    main()
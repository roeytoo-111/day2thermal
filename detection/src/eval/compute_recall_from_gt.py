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
    python3 src/eval/compute_recall_from_gt.py \
        --manifest data/recall_ground_truth/manifest.csv \
        --json results/detections/thermal_dets_rgb_transfer_noleak_p2.json:p2_noleak \
        --json results/detections/thermal_dets_rgb_transfer_noleak_p2_synthA.json:p2_noleak_synthA \
        --tolerance_frames 2 --conf_floor 0.1

    # location-aware, with the gt_boxes.py review applied, over a threshold sweep
    python3 src/eval/compute_recall_from_gt.py ... --boxes data/recall_ground_truth/boxes.csv \
        --sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7

    # score only an unseen temporal holdout (frames the training set never covered)
    python3 src/eval/compute_recall_from_gt.py ... --min_frame 20100
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
    p.add_argument("--boxes", default=None,
                    help="boxes.csv from gt_boxes.py review -> LOCATION-AWARE scoring: a positive frame is caught "
                         "only if a detection lands on the drone box; the audit's corrections are applied.")
    p.add_argument("--det_index_base", type=int, default=1,
                    help="frame_id of the first frame in the detection JSON (run_yolo_inference.py: 1)")
    p.add_argument("--sweep", default=None,
                    help="Comma-separated conf floors to report (with --boxes), e.g. 0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    return p.parse_args()


DET_INDEX_BASE = 1   # run_yolo_inference.py numbers frames from 1; GT frame_ids are 0-based video indices


def load_dets(json_path):
    """Detections keyed by 0-based video frame index (the GT convention)."""
    with open(json_path) as f:
        return {e["frame_id"] - DET_INDEX_BASE: e["detections"] for e in json.load(f)}


def on_target(det_box, gt, min_px=8.0):
    """Hit if the detection centre is within max(min_px, 0.75 * gt long side) of the GT centre, or IoU >= 0.1.
    Tiny targets make IoU unstable, hence the centre-distance rule."""
    dx = (det_box[0] + det_box[2]) / 2 - (gt[0] + gt[2]) / 2
    dy = (det_box[1] + det_box[3]) / 2 - (gt[1] + gt[3]) / 2
    if math.hypot(dx, dy) <= max(min_px, 0.75 * max(gt[2] - gt[0], gt[3] - gt[1])):
        return True
    ix = max(0, min(det_box[2], gt[2]) - max(det_box[0], gt[0]))
    iy = max(0, min(det_box[3], gt[3]) - max(det_box[1], gt[1]))
    inter = ix * iy
    union = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    return union > 0 and inter / union >= 0.1


def located_report(args):
    global DET_INDEX_BASE
    DET_INDEX_BASE = args.det_index_base
    """Location-aware recall / false-fire with the gt_boxes.py review applied.
    Positives: frames reviewed as 'drone' (incl. frames originally labelled empty that the audit found a drone in).
    Negatives: frames reviewed as 'nothing' or 'bird_or_other' (a hit on a bird is a false fire).
    Excluded: 'unsure', and 'no_drone_visible' on frames labelled positive (ambiguous; counted and reported)."""
    b = pd.read_csv(args.boxes)
    if args.min_frame is not None:
        b = b[b.frame_id >= args.min_frame]
    if args.max_frame is not None:
        b = b[b.frame_id <= args.max_frame]
    pos = b[b.verdict == "drone"]
    neg = b[b.verdict.isin(["nothing", "bird_or_other"])]
    moved = int(((b.verdict == "drone") & (b.orig_label == "FP")).sum())
    print(f"Box GT ({args.boxes}): {len(pos)} positive frames ({moved} of them originally labelled empty), "
          f"{len(neg)} negative frames ({int((neg.verdict == 'bird_or_other').sum())} with a bird/other object); "
          f"excluded: {int((b.verdict == 'unsure').sum())} unsure, "
          f"{int((b.verdict == 'no_drone_visible').sum())} labelled-positive with no drone visible.")
    floors = [float(v) for v in args.sweep.split(",")] if args.sweep else [args.conf_floor]
    print(f"\n{'model':<22}{'conf':>5}  {'located recall':>24}  {'frame recall':>13}  {'false-fire (neg frames)':>28}  "
          f"{'off-target dets/pos frame':>26}")
    for spec in args.json:
        json_path, name = spec.rsplit(":", 1)
        dets = load_dets(json_path)
        for conf in floors:
            hit = frame_hit = off = 0
            for r in pos.itertuples():
                ds = [d for d in dets.get(int(r.frame_id), []) if d["conf"] >= conf]
                gt = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                on = [d for d in ds if on_target(d["bbox"], gt)]
                hit += bool(on)
                frame_hit += bool(ds)
                off += len(ds) - len(on)
            ff = sum(any(d["conf"] >= conf for d in dets.get(int(f), [])) for f in neg.frame_id)
            lo, hi = wilson_ci(hit, len(pos))
            flo, fhi = wilson_ci(ff, len(neg))
            print(f"{name:<22}{conf:>5}  {hit:>4}/{len(pos):<4} {hit / len(pos):6.1%} [{lo:.0%}-{hi:.0%}]  "
                  f"{frame_hit / len(pos):>12.1%}  {ff:>4}/{len(neg):<4} {ff / len(neg):6.1%} [{flo:.0%}-{fhi:.0%}]  "
                  f"{off / len(pos):>26.2f}")
    print("\nlocated recall: a detection on the drone box in that exact frame. frame recall: any detection in the "
          "frame (the old metric, same frames, no latency window).")


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
    if args.boxes:
        located_report(args)
        return
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
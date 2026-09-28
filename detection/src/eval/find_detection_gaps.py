"""
Find contiguous stretches where NEITHER detection JSON fired at all, and
sample frames from the longest gaps for manual review. This is where recall
misses concentrate -- if a real drone sat through one of these gaps
undetected by both models, that's the failure mode recall needs to catch,
and it's a much smaller search space than reviewing all frames blind.

Produces a manifest.csv compatible with label_review.py. Since there's no
candidate bbox here (that's the whole point -- nothing was detected), the
"crop" is the full frame. Re-interpret the keys when reviewing:
  't' -> a drone IS visible here, both models missed it (a real recall miss)
  'f' -> no drone, this gap is a legitimate absence

Usage:
    python3 find_detection_gaps.py \
        --json1 thermal_dets_coco.json --json2 thermal_dets_transfer.json \
        --video data/videos/thermal.mp4 \
        --out_dir gap_review \
        --min_gap_frames 30 --sample_per_gap 3 --max_gaps 40
"""

import os
import json
import argparse

import cv2
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json1", required=True)
    p.add_argument("--json2", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--conf_floor", type=float, default=0.0,
                    help="Minimum confidence for a detection to count as 'covering' a frame (default: 0.0, "
                         "any logged detection counts -- raise this if you want to ignore very low-confidence "
                         "noise when deciding what counts as covered).")
    p.add_argument("--min_gap_frames", type=int, default=30,
                    help="Only consider gaps at least this long (default: 30, ~1s at 30fps). Short 1-2 frame "
                         "gaps are likely normal flicker/occlusion, not worth reviewing.")
    p.add_argument("--sample_per_gap", type=int, default=3, help="Frames sampled per qualifying gap.")
    p.add_argument("--max_gaps", type=int, default=40, help="Cap on number of gaps reviewed, longest first.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_covered_frames(json_path, conf_floor):
    with open(json_path) as f:
        data = json.load(f)
    covered = set()
    for entry in data:
        for det in entry["detections"]:
            if det["conf"] >= conf_floor:
                covered.add(entry["frame_id"])
                break
    return covered


def find_gaps(covered, max_frame_id, min_gap_frames):
    gaps = []
    gap_start = None
    for f in range(1, max_frame_id + 1):
        if f not in covered:
            if gap_start is None:
                gap_start = f
        else:
            if gap_start is not None:
                gap_len = f - gap_start
                if gap_len >= min_gap_frames:
                    gaps.append((gap_start, f - 1, gap_len))
                gap_start = None
    if gap_start is not None:
        gap_len = max_frame_id - gap_start + 1
        if gap_len >= min_gap_frames:
            gaps.append((gap_start, max_frame_id, gap_len))
    return gaps


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json1) as f:
        max_frame_id = max(e["frame_id"] for e in json.load(f))

    covered1 = load_covered_frames(args.json1, args.conf_floor)
    covered2 = load_covered_frames(args.json2, args.conf_floor)
    covered_union = covered1 | covered2

    print(f"Model 1 covered {len(covered1)}/{max_frame_id} frames.")
    print(f"Model 2 covered {len(covered2)}/{max_frame_id} frames.")
    print(f"Union covered {len(covered_union)}/{max_frame_id} frames "
          f"({max_frame_id - len(covered_union)} frames neither model fired on).")

    gaps = find_gaps(covered_union, max_frame_id, args.min_gap_frames)
    gaps.sort(key=lambda g: -g[2])  # longest first
    print(f"\nFound {len(gaps)} gaps >= {args.min_gap_frames} frames. "
          f"Reviewing the {min(args.max_gaps, len(gaps))} longest.")

    gaps = gaps[:args.max_gaps]
    for start, end, length in gaps[:10]:
        print(f"  frames {start}-{end}  (length {length})")
    if len(gaps) > 10:
        print(f"  ... and {len(gaps) - 10} more")

    frames_to_pull = []
    for start, end, length in gaps:
        n = min(args.sample_per_gap, length)
        step = max(1, length // n)
        sampled = [start + i * step for i in range(n)]
        for fid in sampled:
            frames_to_pull.append((fid, start, end, length))

    frames_to_pull.sort()
    frame_lookup = {fid: (start, end, length) for fid, start, end, length in frames_to_pull}

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
        if current_frame_id not in frame_lookup:
            continue

        start, end, length = frame_lookup[current_frame_id]
        fname = f"frame_{current_frame_id:06d}_gap{start}-{end}_len{length}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname), frame)
        saved += 1

        manifest_rows.append({
            "filename": fname,
            "frame_id": current_frame_id,
            "conf": "",
            "bbox": "",
            "gate_status": f"gap_len_{length}",
            "label": "",
        })

    cap.release()

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nSaved {saved} full frames to {args.out_dir}")
    print(f"Manifest written to {manifest_path}")
    print(f"\nReview with: python3 label_review.py --audit_dir {args.out_dir}")
    print("Remember: 't' here means 'drone visible, both models missed it' (a real recall miss),")
    print("'f' means 'no drone, legitimate empty frame'.")


if __name__ == "__main__":
    main()
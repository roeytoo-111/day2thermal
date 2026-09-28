"""
Sample frames systematically across the whole video, completely independent
of either detection model's output, for ground-truth recall labeling. This
is the piece the gap-based approach couldn't give you: an unbiased sample,
since selection doesn't depend on what either model did or didn't detect.

Produces a manifest.csv compatible with label_review.py. Full frames (no
crop -- there's no candidate box to crop around, and none should be implied
by the sampling).

Usage:
    python3 sample_frames_for_recall.py \
        --video data/videos/thermal.mp4 \
        --out_dir recall_ground_truth \
        --sample_every 40
"""

import os
import argparse

import cv2
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_every", type=int, default=40,
                    help="Keep every Nth frame (default: 40 -> ~613 frames from a 24524-frame video). "
                         "Purely systematic, no dependency on any detection output.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
        if current_frame_id % args.sample_every != 0:
            continue

        fname = f"frame_{current_frame_id:06d}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname), frame)
        saved += 1
        manifest_rows.append({
            "filename": fname,
            "frame_id": current_frame_id,
            "conf": "",
            "bbox": "",
            "gate_status": "ground_truth_sample",
            "label": "",
        })

    cap.release()

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved {saved} frames (every {args.sample_every}) to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nReview with: python3 label_review.py --audit_dir {args.out_dir}")
    print("'t' = drone visible in this frame, 'f' = no drone. This is ground truth, independent of either model.")


if __name__ == "__main__":
    main()
"""
Extract confidence-stratified sample crops from a single-stage detection JSON
(the schema run_yolo_video_inference.py writes: [{"frame_id":.., "detections":
[{"bbox":.., "conf":.., "class_id":.., "class_name":.., "final":"detected"}]}]),
producing a manifest.csv compatible with label_review.py.

Stratifies per class_name AND by 0.1-wide confidence bucket, so low-confidence
candidates are represented, not just the easy high-confidence ones. Uses
class_name as label_review.py's "gate_status" field, so its by-gate summary
gives you a separate precision readout per class for free -- e.g. is the
model's "bird" bucket actually mostly false triggers vs. "thermal-uav".

Usage:
    python3 extract_detection_json_crops.py \
        --json thermal_dets_coco.json \
        --video data/videos/thermal.mp4 \
        --out_dir thermal_review_coco \
        --per_bucket 20
"""

import os
import json
import argparse
from collections import defaultdict

import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--per_bucket", type=int, default=20,
                    help="Max crops sampled per (class_name, 0.1-wide conf bucket) combination.")
    p.add_argument("--margin", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import random
    random.seed(args.seed)

    with open(args.json) as f:
        data = json.load(f)

    # Bucket every detection by (class_name, conf_bucket)
    buckets = defaultdict(list)
    for entry in data:
        frame_id = entry["frame_id"]
        for det in entry["detections"]:
            conf_bucket = min(9, int(det["conf"] * 10)) / 10
            key = (det["class_name"], conf_bucket)
            buckets[key].append((frame_id, det))

    sampled = []
    print("Sampling per (class, confidence bucket):")
    for key, items in sorted(buckets.items()):
        n = min(args.per_bucket, len(items))
        chosen = random.sample(items, n)
        sampled.extend(chosen)
        print(f"  {key[0]:<15} conf {key[1]:.1f}-{key[1]+0.1:.1f}: {len(items)} available -> {n} sampled")

    frame_to_dets = defaultdict(list)
    for frame_id, det in sampled:
        frame_to_dets[frame_id].append(det)

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
        if current_frame_id not in frame_to_dets:
            continue

        h, w = frame.shape[:2]
        for i, det in enumerate(frame_to_dets[current_frame_id]):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cx1, cy1 = max(0, x1 - args.margin), max(0, y1 - args.margin)
            cx2, cy2 = min(w, x2 + args.margin), min(h, y2 + args.margin)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            fname = f"frame_{current_frame_id:06d}_det{i}_{det['class_name']}_conf{det['conf']:.2f}.png"
            cv2.imwrite(os.path.join(args.out_dir, fname), crop)
            saved += 1

            manifest_rows.append({
                "filename": fname,
                "frame_id": current_frame_id,
                "conf": det["conf"],
                "bbox": [x1, y1, x2, y2],
                "gate_status": det["class_name"],  # -> per-class precision breakdown in label_review.py
                "label": "",
            })

    cap.release()

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    import pandas as pd
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nSaved {saved} crops to {args.out_dir}")
    print(f"Manifest written to {manifest_path}")
    print(f"\nNext: python3 label_review.py --audit_dir {args.out_dir}")


if __name__ == "__main__":
    main()
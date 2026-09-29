"""
Run a trained YOLO checkpoint over a video, frame by frame, logging detections
to JSON. Schema intentionally mirrors the RGB pipeline's detection JSON
(frame_id, detections[].bbox/conf/final) so the existing tooling
(clean_detections.py, show_CNN_detections.py-style crop extraction) works on
this output largely unmodified -- this is a single-stage detector (no
stage1_conf/stage2_prob split), so "final" is just "detected" for anything
above --conf, there's no separate accept/reject gate to log.

Usage:
    python3 src/deploy/run_yolo_inference.py \
        --model models/<run>/weights/best.pt \
        --video data/videos/thermal.mp4 \
        --out_json results/detections/thermal_dets_<run>.json

Log at the default conf 0.05 and score downstream with
compute_recall_from_gt.py --conf_floor 0.1 (plus a sweep): runs logged at
different floors are not comparable.
"""

import json
import argparse
import time

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO inference over a video, log detections to JSON.")
    p.add_argument("--model", required=True, help="Path to trained .pt checkpoint.")
    p.add_argument("--video", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--conf", type=float, default=0.05, help="Confidence threshold to LOG (default: 0.05). "
                    "Kept low deliberately -- filter harder downstream if needed, but you can't recover "
                    "detections you never logged.")
    p.add_argument("--imgsz", type=int, default=512, help="Must match training imgsz (default: 512).")
    p.add_argument("--device", default="0")
    p.add_argument("--max_det", type=int, default=10, help="Max detections logged per frame (default: 10).")
    p.add_argument("--progress_every", type=int, default=500, help="Print progress every N frames.")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    names = model.names  # class_id -> class_name, from the model itself

    results_out = []
    start = time.time()
    frame_id = 0

    # stream=True keeps memory flat across a long video instead of buffering
    # every frame's results in memory at once.
    for result in model.predict(
        source=args.video,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    ):
        frame_id += 1
        detections = []
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                detections.append({
                    "bbox": [round(v, 1) for v in xyxy],
                    "conf": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "final": "detected",  # single-stage model, no accept/reject gate to log
                })

        results_out.append({"frame_id": frame_id, "detections": detections})

        if frame_id % args.progress_every == 0:
            elapsed = time.time() - start
            fps = frame_id / elapsed if elapsed > 0 else 0
            n_with_det = sum(1 for r in results_out if r["detections"])
            print(f"  frame {frame_id}  ({fps:.1f} fps)  "
                  f"{n_with_det}/{frame_id} frames with >=1 detection so far")

    with open(args.out_json, "w") as f:
        json.dump(results_out, f)

    n_with_det = sum(1 for r in results_out if r["detections"])
    n_total_boxes = sum(len(r["detections"]) for r in results_out)
    elapsed = time.time() - start
    print(f"\nDone. {frame_id} frames in {elapsed:.1f}s ({frame_id/elapsed:.1f} fps).")
    print(f"{n_with_det}/{frame_id} frames had >=1 detection, {n_total_boxes} total boxes logged.")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
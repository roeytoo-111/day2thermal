import argparse
import json
import cv2

# Color scheme (BGR format for OpenCV)
COLORS = {
    "detected": (0, 255, 0),             # Green for single-stage/general detections
    "accepted_high_conf": (0, 255, 0),    # Green
    "confirmed": (255, 165, 0),           # Orange
    "rejected": (0, 0, 255),              # Red
}


def draw_detections(frame, dets, show_rejected):
    out = frame.copy()
    for d in dets:
        final_status = d.get("final", "detected")
        if final_status == "rejected" and not show_rejected:
            continue

        # Convert bounding box float coordinates to integers for OpenCV
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = COLORS.get(final_status, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Build overlay text label
        label_parts = []
        if "class_name" in d:
            label_parts.append(d["class_name"])

        if "conf" in d:
            label_parts.append(f"{d['conf']:.2f}")
        elif "stage1_conf" in d:
            label_parts.append(f"s1={d['stage1_conf']:.2f}")

        if d.get("stage2_prob") is not None:
            label_parts.append(f"s2={d['stage2_prob']:.2f}")

        label = " ".join(label_parts)

        # Draw label background box for improved readability
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = max(y1 - 6, text_h + 4)
        cv2.rectangle(
            out,
            (x1, text_y - text_h - 2),
            (x1 + text_w, text_y + baseline - 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Overdetection lay detection JSON onto its source video for review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input video.")
    p.add_argument("--json", required=True, help="Path to detections JSON log.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write annotated output video (mp4).",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show frames in a live display window.",
    )
    p.add_argument(
        "--show-rejected",
        action="store_true",
        help="Also draw rejected detections.",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="First frame to process (1-indexed).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="Stop after this many frames."
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.json) as f:
        log = json.load(f)

    frame_dets = {entry["frame_id"]: entry["detections"] for entry in log}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video file {args.video}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"Video: {frame_w}x{frame_h} @ {fps:.1f} FPS -- {total} frames total -- {len(frame_dets)} logged frames"
    )

    writer = None
    if args.out:
        writer = cv2.VideoWriter(
            args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
        )

    frame_id = 0
    shown = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id < args.start_frame:
            continue

        dets = frame_dets.get(frame_id, [])
        drawn = draw_detections(frame, dets, args.show_rejected)

        if args.display:
            cv2.imshow("Detections", drawn)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrupted by user.")
                break

        if writer is not None:
            writer.write(drawn)

        shown += 1
        if args.max_frames and shown >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video saved to: {args.out}")
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
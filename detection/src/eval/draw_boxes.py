"""
Check the ground truth video with bounding boxes from the RGB detector.
Check data/extract_gt_frames.py to see implementation. Implemented
a jump detector to filter out jumps (assumed to be FPs)

Usage:
python detection/eval/draw_boxes.py -v daata/videos/2026-07-08T11_30_44_day_short.mp4 -d data/gt_audit_out/cleaned_track.csv -o data/video/dayshort_detected.mp4
"""
import argparse
import ast
import os
import cv2
import pandas as pd


def parse_bbox(bbox_val):
    """Parses bbox whether it's a string, list, or numpy array."""
    if pd.isna(bbox_val):
        return None
    if isinstance(bbox_val, (list, tuple)):
        return list(bbox_val)
    if isinstance(bbox_val, str):
        try:
            return ast.literal_eval(bbox_val.strip())
        except (ValueError, SyntaxError):
            return None
    return None


def draw_bounding_boxes(video_path, detections_path, output_path):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Auto-detect separator (comma vs tab/whitespace)
    with open(detections_path, "r") as f:
        first_line = f.readline()
    sep = "," if "," in first_line else r"\s+"

    # Read CSV and strip potential leading/trailing spaces from column names
    df = pd.read_csv(detections_path, sep=sep, engine="python")
    df.columns = df.columns.str.strip()

    if "frame_id" not in df.columns:
        raise KeyError(
            f"Could not find 'frame_id' in CSV headers. Found columns: {list(df.columns)}"
        )

    # Ensure frame_id is integer type for accurate indexing
    df["frame_id"] = df["frame_id"].astype(int)
    detections_by_frame = df.groupby("frame_id")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 1
    print(f"Processing video: {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detections_by_frame.groups:
            frame_detections = detections_by_frame.get_group(frame_idx)

            for _, row in frame_detections.iterrows():
                bbox = parse_bbox(row.get("bbox"))
                if bbox and len(bbox) == 4:
                    xmin, ymin, xmax, ymax = map(int, bbox)
                else:
                    cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
                    xmin = int(cx - w / 2)
                    ymin = int(cy - h / 2)
                    xmax = int(cx + w / 2)
                    ymax = int(cy + h / 2)

                status = str(row.get("track_status", ""))
                final = str(row.get("final", ""))

                # BGR Color scheme: Green for accepted/confirmed, Red otherwise
                color = (
                    (0, 255, 0)
                    if "accepted" in final or "confirmed" in final
                    else (0, 0, 255)
                )

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f"{status} | {final}"
                cv2.putText(
                    frame,
                    label,
                    (xmin, max(ymin - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished! Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay bounding boxes onto video."
    )
    parser.add_argument(
        "--video", "-v", required=True, help="Path to input video"
    )
    parser.add_argument(
        "--detections", "-d", required=True, help="Path to detections CSV"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="annotated_output.mp4",
        help="Output video file path",
    )

    args = parser.parse_args()
    draw_bounding_boxes(args.video, args.detections, args.output)
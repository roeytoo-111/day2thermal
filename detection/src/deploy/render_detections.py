"""
Watch a model instead of just scoring it: burn detections (and GT, if labelled) onto the video.

Why: score_models.py gives a number; it doesn't tell you WHERE it fails (a specific miss, a recurring
false-fire source, a box that's on the drone but the wrong size). This renders the video so you can look.

Colours: detection >= --conf -> green if it lands on a GT box (on_target, same rule as scoring), else
yellow (a false fire, or just unscored if no GT given). A GT positive frame with no on-target detection
draws its box in red (a miss). Conf is printed by each box.

By default only "events" are kept (a detection fires, or the frame is a labelled positive), each padded by
--pad-s of context, so a 10-minute mostly-empty video becomes a short reel instead of something you'd have
to scrub through. --events-only=false renders every frame (every --every-th one) for a full fly-through.

Frame indexing: sequential decode, 0-based, matching the detection JSON's frame_id - 1 (run_yolo_inference.py
and ultralytics' own video loader both decode mp4s in order; cv2 .set(POS_FRAMES) seeking is NOT used here,
same reason as FrameReader in gt_boxes.py -- it disagrees with sequential decode on these variable-rate files).

    python3 src/deploy/render_detections.py --video ../data/new_videos/clipped/2026-07-30T09_27_04_thermal_clipped.mp4 \\
        --json results/detections/pasteB/0730.json --conf 0.25 --out /tmp/pasteB_0730.mp4 \\
        --boxes data/new_sessions/0730_boxes.csv
    # a full fly-through, no GT, every 2nd frame, 2x playback speed:
    python3 src/deploy/render_detections.py --video data/videos/thermal.mp4 --json results/detections/pasteA/0708.json \\
        --conf 0.25 --out /tmp/pasteA_0708_full.mp4 --events-only false --every 2 --speed 2
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from compute_recall_from_gt import load_dets, on_target   # noqa: E402

GREEN, YELLOW, RED, WHITE = (60, 220, 60), (0, 210, 255), (40, 40, 230), (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--json", required=True, help="run_yolo_inference.py output")
    p.add_argument("--out", required=True, help="output .mp4")
    p.add_argument("--conf", type=float, default=0.25, help="draw detections at/above this confidence")
    p.add_argument("--boxes", default=None, help="GT boxes.csv (gt_boxes.py), optional")
    p.add_argument("--manifest", default=None, help="restrict --boxes to one split (needs --split)")
    p.add_argument("--split", default="val")
    p.add_argument("--events-only", type=lambda s: s.lower() != "false", default=True,
                   help="only frames with a detection or a labelled positive, +-pad-s context (default true)")
    p.add_argument("--pad-s", type=float, default=1.5, help="context seconds around each event")
    p.add_argument("--every", type=int, default=1, help="render every N-th source frame (speed/size)")
    p.add_argument("--speed", type=float, default=1.0, help="output fps multiplier (e.g. 2 = 2x playback)")
    p.add_argument("--max-s", type=float, default=None, help="stop after this many source seconds")
    return p.parse_args()


def load_gt(a):
    if not a.boxes:
        return {}
    b = pd.read_csv(a.boxes)
    if a.manifest:
        m = pd.read_csv(a.manifest)
        b = b[b.frame_id.isin(m.loc[m.split == a.split, "frame_id"])]
    b = b[b.verdict.isin(["drone", "bird_or_other"])]
    gt = {}
    for r in b.itertuples():
        gt.setdefault(int(r.frame_id), []).append([r.x0, r.y0, r.x1, r.y1])
    return gt


def main():
    a = parse_args()
    dets = load_dets(a.json)                 # 0-based frame index -> detections
    gt = load_gt(a)                          # 0-based frame index -> GT boxes (boxes.csv convention)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(n, int(a.max_s * fps)) if a.max_s else n

    # pass 1: which frames are "events" (cheap, no decoding)
    pad = int(round(a.pad_s * fps))
    is_event = np.zeros(last, bool)
    for i in range(last):
        if (i in dets and any(d["conf"] >= a.conf for d in dets[i])) or i in gt:
            is_event[max(i - pad, 0):min(i + pad + 1, last)] = True
    keep = is_event if a.events_only else np.ones(last, bool)
    keep[::] &= (np.arange(last) % a.every == 0)
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise SystemExit("nothing to render (no detections >= --conf and no GT positives found)")

    writer = None
    n_hit = n_miss = n_ff = 0
    for i in range(last):
        ok, frame = cap.read()
        if not ok:
            break
        if not keep[i]:
            continue
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps * a.speed / a.every,
                                     (frame.shape[1], frame.shape[0]))
        boxes = gt.get(i, [])
        hit_any = [False] * len(boxes)
        for d in dets.get(i, []):
            if d["conf"] < a.conf:
                continue
            on = [j for j, b in enumerate(boxes) if on_target(d["bbox"], b)]
            for j in on:
                hit_any[j] = True
            color = GREEN if on else YELLOW
            if not on and boxes:
                n_ff += 1
            x0, y0, x1, y1 = (int(v) for v in d["bbox"])
            cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), color, 1)
            cv2.putText(frame, f"{d['conf']:.2f}", (x0 - 2, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        color, 1, cv2.LINE_AA)
        for j, b in enumerate(boxes):
            if not hit_any[j]:
                x0, y0, x1, y1 = (int(v) for v in b)
                cv2.rectangle(frame, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), RED, 1)
                n_miss += 1
            else:
                n_hit += 1
        cv2.putText(frame, f"frame {i}  t={i / fps:.1f}s", (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, WHITE, 1, cv2.LINE_AA)
        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"wrote {a.out}: {n_keep}/{last} source frames kept "
          f"({'events only, +-' + str(a.pad_s) + 's' if a.events_only else 'full'})")
    print(f"  green=hit {n_hit}  red=missed GT {n_miss}  yellow=false-fire-looking box {n_ff}  "
          f"(yellow with no GT given is just an unscored detection, not necessarily wrong)")


if __name__ == "__main__":
    main()

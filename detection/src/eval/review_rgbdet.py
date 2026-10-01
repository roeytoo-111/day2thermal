r"""
Confirm/reject the RGB-derived drone boxes from rgb_to_thermal_labels.py: one pre-drawn box per frame
(not a 6-candidate list), so gt_boxes.py review doesn't fit. This is a smaller, purpose-built reviewer.

Why review at all: the RGB cascade's "accepted"/"confirmed" boxes are proposals, not ground truth -- some
land on real drones, many don't (notebook 2026-10-01/02). Only reviewed rows are usable as training data.

Keys:
  y        confirm the box as-is
  n        no drone here (becomes "nothing")
  drag     redraw the box (then y to accept the new one)
  space    skip for now (re-asked next run)
  q        save and quit (resumable; already-reviewed rows are never re-shown)

Safety: --exclude-frame-min/max drops frames in that range before they're ever shown (e.g. a scored val
range) -- impossible to accidentally confirm a label there.

    python3 src/eval/review_rgbdet.py \
        --video ../data/all_video_pairs/clipped/06-23_scenario1_thermal_clipped.mp4 \
        --boxes data/new_sessions/0623s1_rgbdet_boxes.csv --sample 300
    python3 src/eval/review_rgbdet.py \
        --video ../data/new_videos/clipped/2026-07-15_thermal_clipped.mp4 \
        --boxes data/new_sessions/0715_rgbdet_boxes.csv --sample 300 --exclude-frame-min 9775
"""
import os
import sys
import csv
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_boxes import FrameReader   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--boxes", required=True, help="rgb_to_thermal_labels.py output; reviewed in place")
    p.add_argument("--sample", type=int, default=None, help="review at most this many 'drone' rows (random)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-frame-min", type=int, default=None, help="drop frame_id >= this before reviewing")
    p.add_argument("--exclude-frame-max", type=int, default=None, help="drop frame_id < this before reviewing")
    p.add_argument("--zoom", type=int, default=5, help="magnification of the box crop")
    return p.parse_args()


def main():
    a = parse_args()
    b = pd.read_csv(a.boxes)
    if "reviewed" not in b.columns:
        b["reviewed"] = ""
    b.reviewed = b.reviewed.fillna("")
    if a.exclude_frame_min is not None:
        excl = b.frame_id >= a.exclude_frame_min
        print(f"excluding {excl.sum()} frames >= {a.exclude_frame_min} (never shown, never reviewed)")
        b = b[~excl]
    if a.exclude_frame_max is not None:
        excl = b.frame_id < a.exclude_frame_max
        print(f"excluding {excl.sum()} frames < {a.exclude_frame_max} (never shown, never reviewed)")
        b = b[~excl]
    todo = b[(b.verdict == "drone") & (b.reviewed == "")]
    if a.sample and len(todo) > a.sample:
        todo = todo.sample(a.sample, random_state=a.seed)
    todo = todo.sort_values("frame_id")
    print(f"{len(todo)} boxes to review ({(b.verdict == 'drone').sum()} total, "
          f"{(b.reviewed != '').sum()} already done)")
    if len(todo) == 0:
        return
    fr = FrameReader(a.video, needed=todo.frame_id.astype(int).tolist())
    full = pd.read_csv(a.boxes)   # re-read: write back into the UNFILTERED file, excluded rows untouched
    if "reviewed" not in full.columns:
        full["reviewed"] = ""
    full.reviewed = full.reviewed.fillna("")
    idx_by_frame = {int(r.frame_id): i for i, r in full.iterrows()}

    def save():
        full.to_csv(a.boxes, index=False)

    drag = {"down": None, "cur": None, "box": None}

    def on_mouse(ev, x, y, flags, param):
        z = a.zoom
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag["down"] = (x // z, y // z)
        elif ev == cv2.EVENT_MOUSEMOVE and drag["down"] is not None:
            drag["cur"] = (x // z, y // z)
        elif ev == cv2.EVENT_LBUTTONUP and drag["down"] is not None:
            (x0, y0), (x1, y1) = drag["down"], (x // z, y // z)
            if abs(x1 - x0) > 1 and abs(y1 - y0) > 1:
                drag["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            drag["down"] = drag["cur"] = None

    cv2.namedWindow("confirm RGB-derived box")
    cv2.setMouseCallback("confirm RGB-derived box", on_mouse)
    n_y = n_n = n_skip = 0
    for r in todo.itertuples():
        fi = int(r.frame_id)
        im = fr.read(fi)
        if im is None:
            continue
        box = [r.x0, r.y0, r.x1, r.y1]
        H, W = im.shape[:2]
        half = max(40, int(max(box[2] - box[0], box[3] - box[1]) * 1.5))
        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        x0c, y0c = max(cx - half, 0), max(cy - half, 0)
        x1c, y1c = min(cx + half, W), min(cy + half, H)
        crop0 = im[y0c:y1c, x0c:x1c]
        drag["box"] = [box[0] - x0c, box[1] - y0c, box[2] - x0c, box[3] - y0c]
        while True:
            crop = cv2.resize(crop0, (crop0.shape[1] * a.zoom, crop0.shape[0] * a.zoom),
                              interpolation=cv2.INTER_NEAREST)
            bx = drag["box"]
            cv2.rectangle(crop, (bx[0] * a.zoom, bx[1] * a.zoom), (bx[2] * a.zoom, bx[3] * a.zoom), (0, 255, 255), 1)
            cv2.putText(crop, f"frame {fi}  y=confirm n=nothing drag=redraw space=skip q=quit  "
                              f"[{n_y} y / {n_n} n / {len(todo) - n_y - n_n - n_skip} left]",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("confirm RGB-derived box", crop)
            k = cv2.waitKey(30) & 0xFF
            if drag["down"] is not None or k != 255:
                if k == ord("y"):
                    bx = drag["box"]
                    full.loc[idx_by_frame[fi], ["x0", "y0", "x1", "y1", "verdict", "reviewed"]] = \
                        [bx[0] + x0c, bx[1] + y0c, bx[2] + x0c, bx[3] + y0c, "drone", "y"]
                    n_y += 1
                    break
                if k == ord("n"):
                    full.loc[idx_by_frame[fi], ["verdict", "reviewed"]] = ["nothing", "y"]
                    n_n += 1
                    break
                if k == ord(" "):
                    n_skip += 1
                    break
                if k == ord("q"):
                    save()
                    print(f"saved {a.boxes}: {n_y} confirmed, {n_n} rejected, {n_skip} skipped this session")
                    return
    save()
    print(f"saved {a.boxes}: {n_y} confirmed, {n_n} rejected, {n_skip} skipped this session")


if __name__ == "__main__":
    main()

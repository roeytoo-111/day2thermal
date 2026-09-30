"""
Thermal drone labels from the production RGB detector, via the synced + registered day camera.

Every paired session has a verified time offset (verify_stream_sync.py, incl. clock drift) and per-segment
day->thermal geometry (video_pairs.py calibrate: homography to the undistorted thermal + lens model). So a
drone box the RGB detector finds in day frame d maps into thermal frame t with day_index(t) == d. Boxes
become proposals for human confirmation (gt_boxes.py review --verify: Enter keeps), not ground truth.

Only thermal frames inside a VERIFIED registration segment are labelled, and only boxes fully inside the
day camera's field of view. Frames where the RGB detector found nothing are written as "nothing" ONLY if
--negatives is given (the day camera sees a smaller area than the thermal one: a drone outside it is
invisible to the RGB detector, so "nothing" is weak evidence).

Input: RGB detections as written by run_yolo_inference.py (frame_id 1-based, bbox xyxy in day pixels) for
the SAME clipped day video the registration was computed on.

    python3 src/data/rgb_to_thermal_labels.py --registration ../data/vid_pairs_0715/registration.json \\
        --rgb-json rgb_dets_0715.json --every 25 --min-conf 0.4 \\
        --out-manifest data/new_sessions/0715_rgb_manifest.csv --out-boxes data/new_sessions/0715_rgb_boxes.csv
    python3 src/eval/gt_boxes.py review --verify --only drone,nothing ...   # confirm
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from day2thermal.video_pairs import dist_from_und, day_index   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registration", required=True)
    p.add_argument("--rgb-json", required=True)
    p.add_argument("--rgb-index-base", type=int, default=1, help="frame_id of the first frame in the RGB JSON")
    p.add_argument("--every", type=int, default=25, help="label every N-th thermal frame")
    p.add_argument("--min-conf", type=float, default=0.4)
    p.add_argument("--negatives", action="store_true", help="also write 'nothing' for frames without RGB detections")
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--out-boxes", required=True)
    return p.parse_args()


def day_box_to_thermal(box, H, lam, th_size):
    """Map a day-pixel xyxy box to a thermal-pixel xyxy box (corners + edge midpoints through H and the lens)."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [(x0 + x1) / 2, y0], [x1, (y0 + y1) / 2],
                    [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2]], np.float64)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    xu, yu = ph[:, 0] / ph[:, 2], ph[:, 1] / ph[:, 2]
    xd, yd = dist_from_und(xu, yu, lam, th_size)
    return [float(xd.min()), float(yd.min()), float(xd.max()), float(yd.max())]


def main():
    a = parse_args()
    reg = json.load(open(a.registration))
    lam, th_size = reg["lens"]["lambda"], tuple(reg["thermal_size"])
    W, Hh = reg["rgb_size"]
    fps_t, fps_d = reg["fps_thermal"], reg["fps_day"]
    off, drift = reg["offset_ms"] / 1000, reg.get("drift_ms_per_s", 0.0) / 1000
    segs = [g for g in reg["segments"] if g["verified"]]
    rgb = {e["frame_id"] - a.rgb_index_base: e["detections"] for e in json.load(open(a.rgb_json))}
    rows, man = [], []
    for g in segs:
        H = np.array(g["H_day_to_undistorted_thermal"])
        for ti in range(g["span"][0], g["span"][1] + 1, a.every):
            di = day_index(ti, fps_t, fps_d, off, drift)
            dets = [d for d in rgb.get(di, []) if d["conf"] >= a.min_conf]
            boxes = []
            for d in dets:
                x0, y0, x1, y1 = d["bbox"]
                if x0 < 0 or y0 < 0 or x1 > W or y1 > Hh:
                    continue
                tb = day_box_to_thermal(d["bbox"], H, lam, th_size)
                tb = [max(tb[0] - 1, 0), max(tb[1] - 1, 0), min(tb[2] + 1, th_size[0]), min(tb[3] + 1, th_size[1])]
                boxes.append((tb, d["conf"]))
            if not boxes and not a.negatives:
                continue
            man.append({"filename": f"frame_{ti:06d}.png", "frame_id": ti, "label": "", "split": "train"})
            if boxes:
                for tb, c in boxes:
                    rows.append({"frame_id": ti, "orig_label": "", "verdict": "drone", "x0": int(round(tb[0])),
                                 "y0": int(round(tb[1])), "x1": int(round(tb[2])), "y1": int(round(tb[3])),
                                 "source": f"rgb_det:{c:.2f}:seg{g['id']}", "verified": ""})
            else:
                rows.append({"frame_id": ti, "orig_label": "", "verdict": "nothing", "x0": "", "y0": "", "x1": "",
                             "y1": "", "source": f"rgb_none:seg{g['id']}", "verified": ""})
    pd.DataFrame(man).drop_duplicates("frame_id").to_csv(a.out_manifest, index=False)
    b = pd.DataFrame(rows)
    # gt_boxes keeps one row per frame: keep the most confident RGB box per frame (multi-drone frames: review)
    if len(b):
        b["_c"] = b.source.str.extract(r"rgb_det:([0-9.]+)").astype(float).fillna(0)
        b = b.sort_values("_c", ascending=False).drop_duplicates("frame_id").drop(columns="_c").sort_values("frame_id")
    b.to_csv(a.out_boxes, index=False)
    nd = int((b.verdict == "drone").sum()) if len(b) else 0
    print(f"{len(man)} thermal frames in verified segments -> {nd} with an RGB-derived drone box"
          f"{', ' + str(len(b) - nd) + ' nothing' if a.negatives else ''}. Confirm with gt_boxes.py review --verify.")


if __name__ == "__main__":
    main()

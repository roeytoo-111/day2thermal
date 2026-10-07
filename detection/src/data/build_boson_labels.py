r"""
Per-session thermal training labels from two independent detectors on a registered day/thermal pair:

  RGB   = production day model on the 4K day video, tiled (rgb_label_videos.py), boxes mapped into the thermal view
          with the session's registration (segment homography + thermal lens + clock offset/drift);
  THERM = current thermal model on the thermal video (run_yolo_inference.py, conf >= 0.05).

For every thermal frame inside a verified registration segment (and not carrying a green overlay mark):
  pos     an RGB candidate (class uav/airplane, conf >= --rgb-conf) lands in the thermal frame AND the thermal model
          has a detection on it (conf >= --therm-conf). Label box = the THERMAL box (native alignment, no
          registration error). Measured precision of this rule on hand-reviewed 06-23 s1 boxes: 99.5%.
  hard    RGB candidate (conf >= --hard-conf) inside the thermal frame, thermal model silent (< 0.05) there.
          These are the thermal model's misses *or* RGB false alarms: for human review, never auto-labelled.
  neg     NO confident RGB detection (any class, conf >= --neg-rgb-conf 0.25) and NO thermal detection
          (>= --neg-therm-conf 0.10) in the frame. (The first version used 0.05 for both and left 5 negatives in
          1,223 frames: the RGB model fires faintly on almost every frame.)
          Both models are blind to a drone here, which is the best background evidence available.
  skip    anything else (bird candidates, one-sided weak detections, ...): neither positive nor negative.

Indexing: thermal frame = 0-based sequential decode index of the (remuxed) thermal mp4. Thermal JSON frame_id is
1-based; RGB JSON frame_id k+1 <-> day frame src_frame (every N-th day frame). Day frame for a thermal frame is the
nearest PROCESSED day frame to day_index(...).

    python3 src/data/build_boson_labels.py --session 10_14_01 --registration data/boson_work/reg/10_14_01/registration.json \
        --rgb-json data/boson_work/rgbdets/10_14_01.json --therm-json data/boson_work/thermdets/10_14_01.json \
        --overlay data/boson_work/overlay_flags.json --out data/boson_work/labels/10_14_01.csv
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from day2thermal.video_pairs import day_index                      # noqa: E402
from rgb_to_thermal_labels import day_box_to_thermal               # noqa: E402
from compute_recall_from_gt import on_target                       # noqa: E402

DRONE_CLASSES = {"uav", "airplane"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True)
    p.add_argument("--registration", required=True)
    p.add_argument("--rgb-json", required=True)
    p.add_argument("--therm-json", required=True)
    p.add_argument("--overlay", required=True, help="thermal_overlay_flags.json")
    p.add_argument("--out", required=True)
    p.add_argument("--rgb-conf", type=float, default=0.10)
    p.add_argument("--hard-conf", type=float, default=0.50)
    p.add_argument("--therm-conf", type=float, default=0.20)
    p.add_argument("--t-min", type=float, default=None, help="usable window start, thermal seconds")
    p.add_argument("--t-max", type=float, default=None, help="usable window end, thermal seconds")
    p.add_argument("--neg-rgb-conf", type=float, default=0.25, help="a clean negative has no RGB detection >= this")
    p.add_argument("--neg-therm-conf", type=float, default=0.10, help="... and no thermal detection >= this")
    p.add_argument("--margin", type=int, default=8, help="mapped box must lie this far inside the thermal frame")
    return p.parse_args()


def main():
    a = parse_args()
    reg = json.load(open(a.registration))
    lam, th_size = reg["lens"]["lambda"], tuple(reg["thermal_size"])
    W, H = th_size
    fps_t, fps_d = reg["fps_thermal"], reg["fps_day"]
    off, drift = reg["offset_ms"] / 1000, reg.get("drift_ms_per_s", 0.0) / 1000
    segs = [g for g in reg["segments"] if g["verified"]]
    rgb_list = json.load(open(a.rgb_json))
    step = int(round(np.median(np.diff([e["src_frame"] for e in rgb_list])))) if len(rgb_list) > 1 else 1
    rgb = {e["src_frame"]: e["detections"] for e in rgb_list}
    therm = {e["frame_id"] - 1: e["detections"] for e in json.load(open(a.therm_json))}
    flagged = set(json.load(open(a.overlay)).get(a.session, {}).get("flag", []))
    day_frames = sorted(rgb)
    rows = []
    for g in segs:
        Hm = np.array(g["H_day_to_undistorted_thermal"])
        for ti in range(g["span"][0], g["span"][1] + 1):
            if ti in flagged or (a.t_min is not None and ti / fps_t < a.t_min) or (a.t_max is not None and ti / fps_t > a.t_max):
                continue
            di = day_index(ti, fps_t, fps_d, off, drift)
            k = int(round(di / step)) * step
            if k not in rgb:
                continue
            td = [d for d in therm.get(ti, []) if d["conf"] >= 0.05]
            rd = rgb[k]
            cands, any_rgb = [], False
            for d in rd:
                if d["conf"] >= a.neg_rgb_conf:
                    any_rgb = True
                if d["cls"] not in DRONE_CLASSES or d["conf"] < a.rgb_conf:
                    continue
                tb = day_box_to_thermal(d["bbox"], Hm, lam, th_size)
                if tb[0] < a.margin or tb[1] < a.margin or tb[2] > W - a.margin or tb[3] > H - a.margin:
                    continue
                cands.append((tb, d["conf"], d["cls"], d["bbox"]))
            status, box, rc, tc, dbox = None, None, None, None, None
            best = None
            for tb, c, cls, db in cands:
                for t in td:
                    if on_target(t["bbox"], tb) and t["conf"] >= a.therm_conf and (best is None or t["conf"] > best[1]):
                        best = (t["bbox"], t["conf"], c, db)
            if best is not None:
                status, box, tc, rc, dbox = "pos", best[0], best[1], best[2], best[3]
            else:
                hard = [(tb, c, db) for tb, c, _, db in cands if c >= a.hard_conf
                        and not any(on_target(t["bbox"], tb) for t in td)]
                if hard:
                    tb, c, dbox = max(hard, key=lambda x: x[1])
                    status, box, rc = "hard", tb, c
                elif not any_rgb and not any(t["conf"] >= a.neg_therm_conf for t in td):
                    status = "neg"
                else:
                    status = "skip"
            rows.append({"session": a.session, "frame": ti, "seg": g["id"], "status": status,
                         "x0": "" if box is None else round(box[0], 1), "y0": "" if box is None else round(box[1], 1),
                         "x1": "" if box is None else round(box[2], 1), "y1": "" if box is None else round(box[3], 1),
                         "rgb_conf": "" if rc is None else round(rc, 3), "therm_conf": "" if tc is None else round(tc, 3),
                         "day_frame": k, "dx0": "" if dbox is None else round(dbox[0], 1), "dy0": "" if dbox is None else round(dbox[1], 1),
                         "dx1": "" if dbox is None else round(dbox[2], 1), "dy1": "" if dbox is None else round(dbox[3], 1)})
    df = pd.DataFrame(rows, columns=["session", "frame", "seg", "status", "x0", "y0", "x1", "y1", "rgb_conf", "therm_conf",
                                      "day_frame", "dx0", "dy0", "dx1", "dy1"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    n_all = sum(g["span"][1] - g["span"][0] + 1 for g in segs)
    print(f"{a.session}: {len(df)} frames labelled of {n_all} in verified segments "
          f"({len(flagged)} overlay-flagged): {df.status.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

r"""
Offline seed / model ensemble of uav detections (weighted-box fusion), for the two detection formats we log:
  --fmt yolo  : run_yolo_inference.py JSON (list of {frame_id, detections: [{bbox, conf, class_name}]})
  --fmt ts    : detect_ts.py JSON ({names, frames: [{t, dets: [[x0, y0, x1, y1, conf, cls]]}]})
Only uav boxes are kept (class name containing "uav"; the models differ in class order). Per frame, boxes of all
models are clustered greedily by confidence: a box joins a cluster if IoU >= --iou with the cluster box OR its
centre is within max(--center-px, 0.5 x the cluster box size) (tiny targets barely overlap). Fused box = confidence-
weighted mean; fused score = sum of the best confidence of each model in the cluster / K (a model that did not fire
counts 0), so a box only one of K models sees gets at most 1/K of its score.
Output: same format as the input, uav only, class 0 "uav".

    python3 src/eval/ensemble_dets.py --fmt yolo --out results/detections/ens_A/1015.json \
        results/detections/bosonM0u_s1/1015.json results/detections/bosonN2_s0/1015.json ...
"""
import os
import json
import argparse

import numpy as np


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def fuse(per_model, K, iou_thr, cpx):
    """per_model: list (len K) of lists of [x0,y0,x1,y1,conf]. Returns fused [x0,y0,x1,y1,score]."""
    allb = sorted([(b, m) for m, bs in enumerate(per_model) for b in bs], key=lambda x: -x[0][4])
    clusters = []
    for b, m in allb:
        for c in clusters:
            cb = c["box"]; size = max(cb[2] - cb[0], cb[3] - cb[1])
            d = np.hypot((b[0] + b[2] - cb[0] - cb[2]) / 2, (b[1] + b[3] - cb[1] - cb[3]) / 2)
            if iou(b, cb) >= iou_thr or d <= max(cpx, 0.5 * size):
                c["members"].append((b, m)); break
        else:
            clusters.append({"box": b[:4], "members": [(b, m)]})
    out = []
    for c in clusters:
        best = {}
        for b, m in c["members"]:
            if m not in best or b[4] > best[m][4]:
                best[m] = b
        w = np.array([b[4] for b in best.values()]); B = np.array([b[:4] for b in best.values()])
        box = (B * w[:, None]).sum(0) / w.sum()
        out.append([*box.tolist(), float(w.sum() / K)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fmt", choices=["yolo", "ts"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+"); ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--center-px", type=float, default=6.0); ap.add_argument("--min-score", type=float, default=0.02)
    a = ap.parse_args()
    K = len(a.inputs); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.fmt == "yolo":
        runs = []
        for p in a.inputs:
            d = {e["frame_id"]: [[*x["bbox"], x["conf"]] for x in e["detections"] if "uav" in str(x.get("class_name", "uav")).lower()]
                 for e in json.load(open(p))}
            runs.append(d)
        ids = sorted(set().union(*[set(r) for r in runs]))
        res = []
        for f in ids:
            fused = [b for b in fuse([r.get(f, []) for r in runs], K, a.iou, a.center_px) if b[4] >= a.min_score]
            res.append({"frame_id": f, "detections": [{"bbox": [round(v, 1) for v in b[:4]], "conf": round(b[4], 4),
                                                       "class_id": 0, "class_name": "uav"} for b in fused]})
        json.dump(res, open(a.out, "w"))
    else:
        runs, ts = [], None
        for p in a.inputs:
            D = json.load(open(p)); ids = {int(k) for k, v in D["names"].items() if "uav" in str(v).lower()}
            runs.append({round(f["t"], 4): [x[:5] for x in f["dets"] if x[5] in ids] for f in D["frames"]})
            ts = sorted(runs[-1]) if ts is None else ts
        frames = []
        for t in ts:
            fused = [b for b in fuse([r.get(t, []) for r in runs], K, a.iou, a.center_px) if b[4] >= a.min_score]
            frames.append({"t": t, "dets": [[round(v, 1) for v in b[:4]] + [round(b[4], 4), 0] for b in fused]})
        json.dump({"names": {"0": "uav"}, "frames": frames, "ensemble_of": [os.path.basename(p) for p in a.inputs]}, open(a.out, "w"))
    print(f"{a.out}: fused {K} models")


if __name__ == "__main__":
    main()

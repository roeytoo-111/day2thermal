r"""
Pre-fill a gt_boxes.py boxes.csv from a model's every-frame detections (detect_ts.py JSON), so the human review
(`gt_boxes.py review --verify --only all --gate`) is mostly "Enter = keep". The human verdict is what makes it
ground truth; the pre-fill only saves keystrokes.

1. Link uav detections (conf >= --conf) frame to frame into tracks: constant-velocity prediction, match radius
   max(--radius, 2 x box size), gaps up to --max-gap frames.
2. Keep tracks with >= --min-len detections and median conf >= --min-med.
3. For every manifest frame covered by a kept track: the track's box (detected, or linearly interpolated inside a
   gap) -> verdict "drone", source "prefill_det" / "prefill_interp". One box per frame (gt_boxes stores one):
   the track with the highest median confidence wins.
Frames with no kept track are NOT written (they stay unlabelled, so the reviewer must look at them).
Also merges the model's boxes into proposals.json as numbered candidates (marked "model") ahead of the model-free
ones, keeping the model-free candidates so drones the model misses still show up.

    python3 src/eval/prefill_from_tracks.py --dets ../data/arsuf_1005/th_M0s1.json \
        --manifest ../data/arsuf_1005/gt_thermal/manifest.csv --proposals ../data/arsuf_1005/gt_thermal/proposals.json \
        --out ../data/arsuf_1005/gt_thermal/boxes.csv
"""
import os
import csv
import json
import argparse

import numpy as np
import pandas as pd


def link(frames, conf, radius, max_gap, cls):
    tracks, active = [], []
    for k, f in enumerate(frames):
        dets = [d for d in f["dets"] if d[5] == cls and d[4] >= conf]
        used = set()
        for tr in sorted(active, key=lambda t: -len(t["k"])):
            lk, lb = tr["k"][-1], tr["b"][-1]
            if len(tr["k"]) >= 2:
                v = (np.array(tr["b"][-1]) - np.array(tr["b"][-2])) / max(tr["k"][-1] - tr["k"][-2], 1)
            else:
                v = np.zeros(4)
            pred = np.array(lb) + v * (k - lk)
            pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
            size = max(lb[2] - lb[0], lb[3] - lb[1])
            r = max(radius, 2 * size) * (1 + 0.25 * (k - lk - 1))
            best, bd = None, r
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dd = np.hypot((d[0] + d[2]) / 2 - pc[0], (d[1] + d[3]) / 2 - pc[1])
                if dd < bd:
                    best, bd = j, dd
            if best is not None:
                used.add(best)
                tr["k"].append(k); tr["b"].append(dets[best][:4]); tr["c"].append(dets[best][4])
        for j, d in enumerate(dets):
            if j not in used:
                t = {"k": [k], "b": [d[:4]], "c": [d[4]]}
                tracks.append(t); active.append(t)
        active = [t for t in active if k - t["k"][-1] <= max_gap]
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--proposals", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cls", type=int, default=0); ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--radius", type=float, default=15); ap.add_argument("--max-gap", type=int, default=10)
    ap.add_argument("--min-len", type=int, default=12); ap.add_argument("--min-med", type=float, default=0.35)
    ap.add_argument("--prop-conf", type=float, default=0.1)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: refusing to overwrite review work")
    d = json.load(open(a.dets)); F = d["frames"]
    tracks = link(F, a.conf, a.radius, a.max_gap, a.cls)
    kept = [t for t in tracks if len(t["k"]) >= a.min_len and np.median(t["c"]) >= a.min_med]
    print(f"{len(tracks)} tracks, {len(kept)} kept (>= {a.min_len} dets, median conf >= {a.min_med}); "
          f"kept cover {sum(t['k'][-1] - t['k'][0] + 1 for t in kept)} frames")
    want = sorted(int(x) for x in pd.read_csv(a.manifest).frame_id)
    best = {}
    for t in kept:
        med = float(np.median(t["c"])); ks = np.array(t["k"]); bs = np.array(t["b"])
        for f in want:
            if f < ks[0] or f > ks[-1]:
                continue
            i = np.searchsorted(ks, f)
            if ks[i] == f:
                box, src = bs[i], "prefill_det"
            else:
                w = (f - ks[i - 1]) / (ks[i] - ks[i - 1]); box, src = bs[i - 1] * (1 - w) + bs[i] * w, "prefill_interp"
            if f not in best or med > best[f][0]:
                best[f] = (med, box, src)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame_id", "orig_label", "verdict", "x0", "y0", "x1", "y1", "source", "verified"])
        for f in sorted(best):
            _, b, src = best[f]
            w.writerow([f, "", "drone", int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3])), src, ""])
    P = json.load(open(a.proposals))
    for f in want:
        mb = [{"x": int((x[0] + x[2]) / 2), "y": int((x[1] + x[3]) / 2), "score": round(x[4], 3), "model": True,
               "box": [int(x[0]), int(x[1]), int(x[2]), int(x[3])]}
              for x in sorted(F[f]["dets"], key=lambda x: -x[4]) if x[5] == a.cls and x[4] >= a.prop_conf][:3]
        P["proposals"][str(f)] = mb + P["proposals"].get(str(f), [])
    P["method"] += f" + model candidates ({os.path.basename(a.dets)}, conf >= {a.prop_conf}, listed first)"
    json.dump(P, open(a.proposals, "w"))
    src = pd.Series([v[2] for v in best.values()]).value_counts().to_dict()
    print(f"pre-filled {len(best)}/{len(want)} manifest frames as drone {src}; {len(want) - len(best)} left unlabelled")


if __name__ == "__main__":
    main()

r"""
Score detect_ts.py JSONs against a pipeline_eval ground-truth set (labels.csv from export_pipeline_gt.py).

Frames are matched by timestamp (nearest detection frame within --tol seconds). Per camera:
  positive frame  = >= 1 non-difficult uav box;  negative frame = a labelled EMPTY frame;
  frames with only difficult boxes are ignored; unlabelled frames never enter.
Hit rule (the same as score_models.py): a detection of the uav class hits a GT box if its centre is within
max(8 px, 0.75 x the GT box's long side) of the GT centre, or IoU >= 0.1.
Recall = positive frames with a hit at confidence >= thr; false-fire (FF) = negative frames with any uav
detection >= thr. Reported: recall @ FF <= 5 / 10 / 25 %, mean recall over FF 1..25 % (exact thresholds), and
recall at the model's own 0.25 / 0.5 thresholds with their FF. Size buckets (long side of the GT box in the
video's pixels) are reported for recall at the FF <= 10 % threshold.

    python3 src/eval/score_arsuf_gt.py --gt ../data/arsuf_1005/pipeline_gt/arsuf_1005 --camera thermal \
        --model M0 results/arsuf/th_bosonM0u_s1.json --model L_s0 results/arsuf/th_pasteG_y26sp2.json
"""
import json
import argparse

import numpy as np
import pandas as pd


def uav_class_ids(names):
    return {int(k) for k, v in names.items() if "uav" in str(v).lower()}


def load_dets(path):
    d = json.load(open(path)); ids = uav_class_ids(d["names"])
    ts = np.array([f["t"] for f in d["frames"]])
    dets = [[x for x in f["dets"] if x[5] in ids] for f in d["frames"]]
    return ts, dets


def hit(det, g):
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    gx, gy = g["left"] + g["width"] / 2, g["top"] + g["height"] / 2
    r = max(8.0, 0.75 * max(g["width"], g["height"]))
    if np.hypot(cx - gx, cy - gy) <= r:
        return True
    ix = max(0.0, min(det[2], g["left"] + g["width"]) - max(det[0], g["left"]))
    iy = max(0.0, min(det[3], g["top"] + g["height"]) - max(det[1], g["top"]))
    inter = ix * iy
    u = (det[2] - det[0]) * (det[3] - det[1]) + g["width"] * g["height"] - inter
    return u > 0 and inter / u >= 0.1


def frame_table(lab, ts, dets, tol):
    """One row per labelled frame: kind, best-hit conf (pos), best any-uav conf (neg), GT long side."""
    rows = []
    for t, g in lab.groupby("t"):
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) > tol:
            continue                                      # this frame was not run: skip, not a miss
        obj = g[g.cls == "uav"]
        if len(obj) == 0:
            rows.append(("neg", max([d[4] for d in dets[j]], default=0.0), np.nan))
        elif (obj.difficult == 0).any():
            gs = obj[obj.difficult == 0].to_dict("records")
            best = max([d[4] for d in dets[j] for gg in gs if hit(d, gg)], default=0.0)
            rows.append(("pos", best, max(max(x["width"], x["height"]) for x in gs)))
    return pd.DataFrame(rows, columns=["kind", "conf", "size"])


def curve(df):
    pos, neg = df[df.kind == "pos"].conf.values, df[df.kind == "neg"].conf.values
    thr = np.unique(np.r_[pos, neg, 0.05])
    thr = thr[thr >= 0.05]
    R = np.array([(pos >= t).mean() for t in thr]); F = np.array([(neg >= t).mean() if len(neg) else 0 for t in thr])
    return thr, R, F


def at_ff(thr, R, F, ff):
    ok = F <= ff + 1e-9
    return (R[ok].max(), thr[ok][R[ok].argmax()]) if ok.any() else (0.0, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--camera", required=True, choices=["thermal", "day"])
    ap.add_argument("--model", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--tol", type=float, default=0.021); ap.add_argument("--out")
    ap.add_argument("--t-max", type=float, default=None, help="ignore GT frames after this time")
    a = ap.parse_args()
    lab = pd.read_csv(f"{a.gt}/labels.csv", dtype={"class": str}).rename(columns={"class": "cls"})
    lab = lab[lab.camera == a.camera].copy(); lab["cls"] = lab.cls.fillna("")
    lab["difficult"] = lab.difficult.fillna(0).astype(int)
    if a.t_max:
        lab = lab[lab.t <= a.t_max]
    out = []
    for name, path in a.model:
        ts, dets = load_dets(path); df = frame_table(lab, ts, dets, a.tol)
        thr, R, F = curve(df); npos, nneg = (df.kind == "pos").sum(), (df.kind == "neg").sum()
        r5, r10, r25 = (at_ff(thr, R, F, x) for x in (0.05, 0.10, 0.25))
        mean = float(np.mean([at_ff(thr, R, F, x / 100)[0] for x in range(1, 26)]))
        row = {"model": name, "pos": int(npos), "neg": int(nneg), "recall@FF<=5%": 100 * r5[0], "recall@FF<=10%": 100 * r10[0],
               "recall@FF<=25%": 100 * r25[0], "mean recall FF1-25%": 100 * mean}
        for t in (0.25, 0.5):
            row[f"recall@{t}"] = 100 * (df[df.kind == "pos"].conf >= t).mean()
            row[f"FF@{t}"] = 100 * (df[df.kind == "neg"].conf >= t).mean()
        thr10 = r10[1]
        p = df[df.kind == "pos"]
        for lo, hi, nm in ((0, 10, "<10px"), (10, 25, "10-25px"), (25, 1e9, ">=25px")):
            m = (p["size"] >= lo) & (p["size"] < hi)
            row[f"R10 {nm} (n={int(m.sum())})"] = 100 * (p[m].conf >= thr10).mean() if m.sum() and thr10 == thr10 else np.nan
        out.append(row)
    res = pd.DataFrame(out).set_index("model").round(1)
    txt = f"### {a.camera}: {int(res['pos'].iloc[0])} positive / {int(res['neg'].iloc[0])} empty frames\n\n" + res.drop(columns=["pos", "neg"]).T.to_string()
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()

r"""
Empty-frame false fire vs recall over confidence thresholds (the 2026-10-12 sweep), for detection JSONs against a
ground truth. --gt-kind arsuf: pipeline_eval labels.csv (thermal camera, timestamp matched, detect_ts JSON);
--gt-kind boson: gt_boxes boxes.csv (frame index) + run_yolo_inference JSON (frame_id = index + 1).
Also prints, per model, the false fire at the threshold where it reaches a reference recall (--ref-recall).

    python3 src/eval/ff_sweep.py --gt-kind arsuf --gt ../data/arsuf_1005/pipeline_gt_cut/arsuf_1005_cut/labels.csv \
        --model M0 ../data/arsuf_1005/dets/th_bosonM.json --ref-recall 53.4
"""
import json, argparse
import numpy as np, pandas as pd
from score_arsuf_gt import hit


def frames_arsuf(gt, path, tol=0.021):
    lab = pd.read_csv(gt, dtype={"class": str}).rename(columns={"class": "cls"})
    lab = lab[lab.camera == "thermal"]; lab["cls"] = lab.cls.fillna(""); lab["difficult"] = lab.difficult.fillna(0)
    D = json.load(open(path)); ids = {int(k) for k, v in D["names"].items() if "uav" in v}; T = np.array([q["t"] for q in D["frames"]])
    for t, g in lab.groupby("t"):
        j = int(np.argmin(np.abs(T - t)))
        if abs(T[j] - t) > tol:
            continue
        gs = g[(g.cls == "uav") & (g.difficult == 0)].to_dict("records")
        kind = "pos" if gs else ("neg" if (g.cls == "").all() else None)
        if kind:
            yield kind, gs, [q for q in D["frames"][j]["dets"] if q[5] in ids]


def frames_boson(gt, path):
    b = pd.read_csv(gt); D = json.load(open(path))
    dets = {e["frame_id"] - 1: e["detections"] for e in D}
    for r in b.itertuples():
        if r.verdict == "drone":
            gs = [{"left": r.x0, "top": r.y0, "width": r.x1 - r.x0, "height": r.y1 - r.y0}]; kind = "pos"
        elif r.verdict == "nothing":
            gs, kind = [], "neg"
        else:
            continue
        ds = [[*d["bbox"], d["conf"], 0] for d in dets.get(int(r.frame_id), []) if "uav" in str(d.get("class_name", "uav"))]
        yield kind, gs, ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-kind", choices=["arsuf", "boson"], required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--model", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--ref-recall", type=float, default=None)
    ap.add_argument("--thr", default="0.25,0.35,0.45,0.5,0.6")
    a = ap.parse_args()
    thrs = [float(x) for x in a.thr.split(",")]
    rows = []
    for name, path in a.model:
        F = list(frames_arsuf(a.gt, path) if a.gt_kind == "arsuf" else frames_boson(a.gt, path))
        pos = np.array([max([q[4] for q in ds if any(hit(q, g) for g in gs)], default=0.0) for k, gs, ds in F if k == "pos"])
        neg = np.array([max([q[4] for q in ds], default=0.0) for k, gs, ds in F if k == "neg"])
        r = {"model": name, "pos": len(pos), "neg": len(neg)}
        for t in thrs:
            r[f"{t:.2f}"] = f"{100 * (neg >= t).mean():.1f} / {100 * (pos >= t).mean():.1f}"
        if a.ref_recall is not None:
            ts = np.unique(np.r_[pos, neg]); ts = ts[ts >= 0.05]
            ok = [t for t in ts if 100 * (pos >= t).mean() >= a.ref_recall]
            tt = max(ok) if ok else np.nan
            r[f"FF@recall>={a.ref_recall:.0f}"] = f"{100 * (neg >= tt).mean():.1f} (thr {tt:.2f})" if ok else "n/a"
        rows.append(r)
    print("empty-frame false fire % / recall % at each threshold")
    print(pd.DataFrame(rows).set_index("model").to_string())


if __name__ == "__main__":
    main()

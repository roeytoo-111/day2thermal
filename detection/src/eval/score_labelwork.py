r"""
Score thermal (or day) detection JSONs (detect_ts.py format: frames [{t, dets}], t = pts_time) against a tools/labeling work
dir (labels.json + frame_times.json), the labels of the tuning flights (scenario1, 13_18, 13_21).

Frame identity: pts = file_start_pts + t[index - first_index] (frame_times.json of that camera); a label frame is scored if a
detection frame lies within --tol s of it.

Three label sets, because almost all labels are auto labels made by the deployed models (which biases a comparison toward
them):
  reviewed   only frames a human labelled or accepted (src != auto or reviewed)           -> fair, small
  audit      only the random audit frames (audit.json) that a human has done             -> fair, unbiased sample
  trusted    reviewed + unflagged auto frames (flagged-unreviewed and bulk "not airborne" frames dropped) -> big, BIASED
Per set: drone / empty frames, mean recall over FF 1-25 %, recall at FF <= 5 / 10 %, and recall / FF at fixed thresholds.

    python3 src/eval/score_labelwork.py --work ~/label_work/f1318_1026 --cam thermal \
        --model M0n_s1 results/tuning/f1318_bosonM0n_s1.json --model M0f_s0 results/tuning/f1318_bosonM0f_s0.json
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_arsuf_gt import load_dets, hit   # noqa: E402


def frame_sets(work, cam):
    L = json.load(open(os.path.join(work, "labels.json"))); T = L[cam]
    F = json.load(open(os.path.join(work, "frame_times.json")))[cam]
    t, first, start = np.array(F["t"], float), int(F["first_index"]), float(F["file_start_pts"])
    bulk = [b["frames"] for b in L["meta"].get("bulk_none", []) if b["cam"] == cam]
    ap = os.path.join(work, "audit.json")
    audit = set(json.load(open(ap))["frames"]) if os.path.exists(ap) else set()
    out = {"reviewed": [], "audit": [], "trusted": []}
    for k, v in T.items():
        f = int(k)
        if v is None or not (0 <= f - first < len(t)):
            continue
        reviewed = v.get("src") != "auto" or v.get("reviewed")
        in_bulk = any(b0 <= f <= b1 for b0, b1 in bulk)
        if reviewed and v.get("state") != "box" and in_bulk:
            continue                                    # the bulk mark, not a review
        if v.get("unsure"):
            continue
        g = None
        if v.get("state") == "box" and v.get("box"):
            x, y, w, h = v["box"]; g = {"left": x, "top": y, "width": w, "height": h}
        rec = (start + t[f - first], g)
        if reviewed:
            out["reviewed"].append(rec)
            if f in audit:
                out["audit"].append(rec)
        if reviewed or (not v.get("flags") and not in_bulk):
            out["trusted"].append(rec)
    return out


def table(recs, dets_path, tol):
    ts, dets = load_dets(dets_path)
    pos, neg = [], []
    for tt, g in recs:
        j = int(np.argmin(np.abs(ts - tt)))
        if abs(ts[j] - tt) > tol:
            continue
        if g is None:
            neg.append(max([d[4] for d in dets[j]], default=0.0))
        else:
            pos.append(max([d[4] for d in dets[j] if hit(d, g)], default=0.0))
    return np.array(pos), np.array(neg)


def summary(pos, neg, thrs):
    if len(pos) == 0:
        return None
    grid = np.unique(np.r_[pos, neg, 0.05]); grid = grid[grid >= 0.05]
    R = np.array([(pos >= x).mean() for x in grid]); F = np.array([(neg >= x).mean() if len(neg) else 0.0 for x in grid])
    at = lambda ff: R[F <= ff + 1e-9].max() * 100 if (F <= ff + 1e-9).any() else 0.0
    out = {"drone": len(pos), "empty": len(neg), "mean FF1-25": np.mean([at(f / 100) for f in range(1, 26)]) if len(neg) else np.nan,
           "R@FF5": at(0.05) if len(neg) else np.nan, "R@FF10": at(0.10) if len(neg) else np.nan}
    for x in thrs:
        out[f"@{x} R/FF"] = f"{100 * (pos >= x).mean():.1f}/{(100 * (neg >= x).mean()) if len(neg) else float('nan'):.1f}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--cam", default="thermal", choices=["thermal", "day"])
    ap.add_argument("--model", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--tol", type=float, default=0.021); ap.add_argument("--thr", default="0.4,0.5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = frame_sets(os.path.expanduser(a.work), a.cam); thrs = [float(x) for x in a.thr.split(",")]
    lines = [f"## {os.path.basename(a.work.rstrip('/'))} ({a.cam})", ""]
    for name, recs in S.items():
        rows = {m: summary(*table(recs, p, a.tol), thrs) for m, p in a.model}
        rows = {m: r for m, r in rows.items() if r}
        if not rows:
            lines.append(f"### {name}: no scorable drone frames"); lines.append(""); continue
        r0 = next(iter(rows.values()))
        lines += [f"### {name} labels: {r0['drone']} drone / {r0['empty']} empty frames", "",
                  "| model | " + " | ".join(k for k in r0 if k not in ("drone", "empty")) + " |", "|---|" + "---|" * (len(r0) - 2)]
        for m, r in rows.items():
            lines.append(f"| {m} | " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for k, v in r.items() if k not in ("drone", "empty")) + " |")
        lines.append("")
    txt = "\n".join(lines); print(txt)
    if a.out:
        open(a.out, "a").write(txt + "\n")


if __name__ == "__main__":
    main()

"""
Classical small-target detector (model-free) alone and fused with YOLO, scored like score_models.py:
location-aware recall at matched false-fire, on the labelled test frames.

Classical detector = the gt_boxes.py proposal generator: motion-compensated temporal median (+-2/4/6
frames) + white/black top-hat, each normalised by LOCAL clutter; top-6 peaks per frame with a score.
It is causal only up to +6 frames (0.12-0.24 s latency at 50/25 fps). Scores are on a per-video scale
(clutter normalisation differs per camera), so they are mapped to their percentile rank within the video.

Fusion: a frame "fires" / a drone is "hit" if YOLO fires at conf >= ty OR the classical detector fires at
rank >= tc. The best (ty, tc) under each false-fire target is reported -- an upper bound on what a tuned
fusion rule gives (the thresholds are chosen on the same frames; treat as a feasibility check).

    python3 src/eval/classical_fusion.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_recall_from_gt import on_target, wilson_ci, load_dets   # noqa: E402

DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NS = os.path.join(DET, "data", "new_sessions")
R = os.path.join(DET, "results", "detections")
SETS = {
    "07-02 s1 (key 0708)": dict(boxes=os.path.join(DET, "data", "recall_ground_truth", "boxes.csv"),
                                props=os.path.join(DET, "data", "recall_ground_truth", "proposals.json"),
                                yolo={"p2_noleak": os.path.join(R, "p2_noleak", "0708.json")}),
    "07-30 (test 2)": dict(boxes=os.path.join(NS, "0730_boxes.csv"), props=os.path.join(NS, "0730_test_proposals.json"),
                           yolo={"p2_noleak": os.path.join(R, "p2_noleak", "0730.json"),
                                 "p2_sessions_aug": os.path.join(R, "p2_sessions_aug", "0730.json")}),
}
FF_TARGETS = [0.05, 0.10, 0.25]


def classical_dets(props_path):
    P = json.load(open(props_path))["proposals"]
    scores = np.array([c["score"] for v in P.values() for c in v])
    srt = np.sort(scores)
    rank = lambda s: float(np.searchsorted(srt, s, side="right") / len(srt))
    return {int(f): [{"bbox": c["box"], "conf": rank(c["score"]), "size": max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1])}
                     for c in v] for f, v in P.items()}


def evaluate(pos, neg, sources, thresholds):
    """sources: list of det dicts; thresholds: same-length list. A hit/fire if ANY source fires above its threshold."""
    def fired(f, on=None):
        for dets, t in zip(sources, thresholds):
            for d in dets.get(int(f), []):
                if d["conf"] >= t and (on is None or on_target(d["bbox"], on)):
                    return True
        return False
    hit = sum(fired(r.frame_id, (r.x0, r.y0, r.x1, r.y1)) for r in pos.itertuples())
    ff = sum(fired(f) for f in neg.frame_id)
    return hit, ff


def main():
    lines = []
    for name, s in SETS.items():
        b = pd.read_csv(s["boxes"])
        pos, neg = b[b.verdict == "drone"], b[b.verdict.isin(["nothing", "bird_or_other"])]
        cl = classical_dets(s["props"])
        n, m = len(pos), len(neg)
        lines += [f"### {name}: {n} positive / {m} negative frames", "",
                  "| detector | " + " | ".join(f"recall @FF≤{int(t * 100)}%" for t in FF_TARGETS) + " |",
                  "|---|" + "---|" * len(FF_TARGETS)]
        # classical alone
        grid_c = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
        res_c = [(tc,) + evaluate(pos, neg, [cl], [tc]) for tc in grid_c]
        row = []
        for t in FF_TARGETS:
            ok = [r for r in res_c if r[2] / m <= t]
            best = max(ok, key=lambda r: r[1]) if ok else None
            row.append(f"{best[1] / n:.1%} (rank≥{best[0]})" if best else "—")
        lines.append("| classical only | " + " | ".join(row) + " |")
        for yname, ypath in s["yolo"].items():
            y = load_dets(ypath)
            floor = min(d["conf"] for v in y.values() for d in v)
            grid_y = [c for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] if c >= floor - 1e-9]
            res_y = [(ty,) + evaluate(pos, neg, [y], [ty]) for ty in grid_y]
            res_f = [(ty, tc) + evaluate(pos, neg, [y, cl], [ty, tc]) for ty in grid_y for tc in grid_c[3:] + [1.01]]
            for label, res, k in ((yname, res_y, 1), (f"{yname} + classical", res_f, 2)):
                row = []
                for t in FF_TARGETS:
                    ok = [r for r in res if r[k + 1] / m <= t]
                    best = max(ok, key=lambda r: r[k]) if ok else None
                    if not best:
                        row.append("—")
                        continue
                    lo, hi = wilson_ci(best[k], n)
                    th = f"conf≥{best[0]}" + (f", rank≥{best[1]}" if k == 2 and best[1] <= 1 else "")
                    row.append(f"{best[k] / n:.1%} [{lo:.0%}–{hi:.0%}] ({th})")
                lines.append(f"| {label} | " + " | ".join(row) + " |")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(DET, "results", "classical_fusion_2026-09-30.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()

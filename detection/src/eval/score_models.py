"""
One table for all models on all labelled test sets: location-aware recall at matched false-fire rates.

A model is a directory with one detection JSON per video (run_yolo_inference.py output), named by set key:
    0708.json   data/videos/thermal.mp4                       (test)
    0730.json   2026-07-30T09_27_04_thermal_clipped.mp4       (test 2)
    0715.json   2026-07-15_thermal_clipped.mp4                (scored on its held-out VAL chunk only)
    0623.json   2026-06-23_scenario2_thermal_clipped.mp4      (scored on its held-out VAL chunk only)
Missing files are skipped. 06-23/07-15 train chunks are never scored (newer models trained on them).

*_sky: only drones seen against sky (gt_background.py tags), all negatives -- the interceptor looks UP at
the target, so these are the operational numbers.

For each set: located recall and false-fire at conf 0.1, and located recall at the highest-recall
threshold whose false-fire rate stays <= each target (5/10/25%) -- the fair comparison between models that
fire at different rates.

    python3 src/eval/score_models.py --model p2_noleak results/detections/p2_noleak \\
        --model p2_sessions results/detections/p2_sessions --out results/scores_2026-09-30.md
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_recall_from_gt import on_target, wilson_ci, load_dets   # noqa: E402

DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NS = os.path.join(DET, "data", "new_sessions")
SETS = {   # name: (json key, boxes, frame range, manifest for its val split, GT background filter)
    "0708_all": ("0708", os.path.join(DET, "data", "recall_ground_truth", "boxes.csv"), None, None, None),
    "0708_airborne": ("0708", os.path.join(DET, "data", "recall_ground_truth", "boxes.csv"), (201, 23599), None, None),
    "0708_sky": ("0708", os.path.join(DET, "data", "recall_ground_truth", "boxes.csv"), None, None,
                 os.path.join(DET, "data", "recall_ground_truth", "gt_bg.csv")),
    "0730_test2": ("0730", os.path.join(NS, "0730_boxes.csv"), None, None, None),
    "0730_sky": ("0730", os.path.join(NS, "0730_boxes.csv"), None, None, os.path.join(NS, "0730_gt_bg.csv")),
    # 0715 val = everything labelled from frame 9775 on (train chunk ends at 9475; includes the 2026-10-01 densified
    # labels, which are not in 0715_manifest.csv -- selecting by manifest scored only the original 12/27 frames)
    "0715_val (dark over terrain)": ("0715", os.path.join(NS, "0715_boxes.csv"), (9775, 10 ** 9), None, None),
    "0623_val": ("0623", os.path.join(NS, "0623_boxes.csv"), None, os.path.join(NS, "0623_manifest.csv"), None),
}
CONFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
FF_TARGETS = [0.05, 0.10, 0.25]


def load_set(boxes, rng, manifest, bg=None):
    """bg: gt_background.py csv -> keep only positives seen against sky (negatives: all, a false fire anywhere
    counts). The interceptor looks up at the target, so *_sky is the operational case."""
    b = pd.read_csv(boxes)
    if rng:
        b = b[(b.frame_id >= rng[0]) & (b.frame_id <= rng[1])]
    if manifest:
        m = pd.read_csv(manifest)
        b = b[b.frame_id.isin(m.loc[m.split == "val", "frame_id"])]
    pos = b[b.verdict == "drone"]
    neg = b[b.verdict.isin(["nothing", "bird_or_other"])]
    if bg:
        t = pd.read_csv(bg)
        pos = pos[pos.frame_id.isin(t.loc[t.bg == "sky", "frame_id"])]
    return pos, neg


def curve(dets, pos, neg, confs):
    rows = []
    for c in confs:
        hit = sum(any(d["conf"] >= c and on_target(d["bbox"], (r.x0, r.y0, r.x1, r.y1))
                      for d in dets.get(int(r.frame_id), [])) for r in pos.itertuples())
        ff = sum(any(d["conf"] >= c for d in dets.get(int(f), [])) for f in neg.frame_id)
        rows.append((c, hit, ff))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    ap.add_argument("--out", default=None, help="also write the tables as markdown here")
    ap.add_argument("--floor", type=float, default=None,
                    help="lowest conf threshold used for EVERY model (default: the highest logging floor among the "
                         "models' JSONs -- a threshold below a model's logging floor is not recoverable)")
    a = ap.parse_args()
    import json as _json
    floors = []
    for _, d in a.model:
        for f in os.listdir(d):
            if f.endswith(".json"):
                c = [x["conf"] for e in _json.load(open(os.path.join(d, f))) for x in e["detections"]]
                if c:
                    floors.append(min(c))
    floor = a.floor if a.floor is not None else round(max(floors), 2)
    confs = [c for c in CONFS if c >= floor - 1e-9] or [floor]
    if floor not in confs:
        confs = [floor] + confs
    print(f"common confidence floor: {floor} (thresholds used: {confs})\n")
    lines = []
    for sname, (key, boxes, rng, manifest, bg) in SETS.items():
        pos, neg = load_set(boxes, rng, manifest, bg)
        head = f"### {sname}: {len(pos)} positive / {len(neg)} negative frames"
        tab = [f"| model | recall @{confs[0]} | FF @{confs[0]} | " + " | ".join(f"recall @FF≤{int(t*100)}%" for t in FF_TARGETS) + " |",
               "|---|---|---|" + "---|" * len(FF_TARGETS)]
        any_model = False
        for name, d in a.model:
            p = os.path.join(d, f"{key}.json")
            if not os.path.exists(p):
                continue
            any_model = True
            rows = curve(load_dets(p), pos, neg, confs)
            n, m = max(len(pos), 1), max(len(neg), 1)
            r01 = rows[0]
            lo, hi = wilson_ci(r01[1], n)
            cells = [f"{r01[1] / n:.1%} [{lo:.0%}–{hi:.0%}]", f"{r01[2] / m:.1%}"]
            for t in FF_TARGETS:
                ok = [r for r in rows if r[2] / m <= t]
                if not ok or len(neg) < 20:
                    cells.append("n/a" if len(neg) < 20 else "—")
                    continue
                best = max(ok, key=lambda r: r[1])
                cells.append(f"{best[1] / n:.1%} (conf {best[0]})")
            tab.append(f"| {name} | " + " | ".join(cells) + " |")
        if any_model:
            lines += [head, "", *tab, ""]
    txt = "\n".join(lines) + ("\n(n/a: fewer than 20 negative frames, false-fire not measurable)\n")
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()

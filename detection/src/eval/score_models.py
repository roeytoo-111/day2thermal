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

Thresholds (2026-10-02): every distinct score >= the floor is tried, so recall @FF<=x is exact. The old 13-value
grid added discretisation noise on top of run-to-run noise. "mean recall FF 1-25%" averages the exact recall
over FF budgets 1%..25% -- one smoother number per model, less sensitive to a single operating point.

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
    # Boson 2026-10-06 held-out sessions, human ground truth (gt_boxes.py review); skipped until those files exist
    "boson_0858 (held out, 4:46+)": ("0858", os.path.join(DET, "..", "data", "boson_work", "gt", "08_58_15_boxes.csv"), None, None, None),
    "boson_1015 (held out, 0-1:30)": ("1015", os.path.join(DET, "..", "data", "boson_work", "gt", "10_15_59_boxes.csv"), None, None, None),
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


def frame_scores(dets, pos, neg):
    """Per positive frame: the highest conf among detections ON the drone (0 if none). Per negative frame: the
    highest conf of any detection (0 if none). Every threshold's located recall / false-fire follows from these."""
    sp = np.array([max([d["conf"] for d in dets.get(int(r.frame_id), [])
                        if on_target(d["bbox"], (r.x0, r.y0, r.x1, r.y1))], default=0.0) for r in pos.itertuples()])
    sn = np.array([max([d["conf"] for d in dets.get(int(f), [])], default=0.0) for f in neg.frame_id])
    return sp, sn


def at_ff(sp, sn, floor, t):
    """Best located recall over ALL thresholds c >= floor with false-fire rate <= t (exact, not a coarse grid).
    Returns (recall, threshold) or (None, None) if even the highest threshold fires too often."""
    cands = np.unique(np.concatenate([sp[sp >= floor], sn[sn >= floor], [floor]]))
    m = max(len(sn), 1)
    best = (None, None)
    for c in cands:                      # ascending: the first c meeting the FF budget has the highest recall
        if (sn >= c).sum() / m <= t:
            best = ((sp >= c).mean() if len(sp) else 0.0, float(c))
            break
    return best


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
    print(f"common confidence floor: {floor} (every threshold >= floor is tried; exact recall at each FF budget)\n")
    lines = []
    for sname, (key, boxes, rng, manifest, bg) in SETS.items():
        if not os.path.exists(boxes):
            continue
        pos, neg = load_set(boxes, rng, manifest, bg)
        head = f"### {sname}: {len(pos)} positive / {len(neg)} negative frames"
        tab = [f"| model | recall @{floor} | FF @{floor} | " + " | ".join(f"recall @FF≤{int(t*100)}%" for t in FF_TARGETS)
               + " | mean recall FF 1–25% |", "|---|---|---|" + "---|" * (len(FF_TARGETS) + 1)]
        any_model = False
        for name, d in a.model:
            p = os.path.join(d, f"{key}.json")
            if not os.path.exists(p):
                continue
            any_model = True
            sp, sn = frame_scores(load_dets(p), pos, neg)
            n, m = max(len(pos), 1), max(len(neg), 1)
            hit0, ff0 = int((sp >= floor).sum()), int((sn >= floor).sum())
            lo, hi = wilson_ci(hit0, n)
            cells = [f"{hit0 / n:.1%} [{lo:.0%}–{hi:.0%}]", f"{ff0 / m:.1%}"]
            if len(neg) < 20:
                cells += ["n/a"] * (len(FF_TARGETS) + 1)
            else:
                for t in FF_TARGETS:
                    r, c = at_ff(sp, sn, floor, t)
                    cells.append("—" if r is None else f"{r:.1%} (conf {c:.2f})")
                curve_vals = [at_ff(sp, sn, floor, t / 100)[0] or 0.0 for t in range(1, 26)]
                cells.append(f"{np.mean(curve_vals):.1%}")
            tab.append(f"| {name} | " + " | ".join(cells) + " |")
        if any_model:
            lines += [head, "", *tab, ""]
    txt = "\n".join(lines) + ("\n(n/a: fewer than 20 negative frames, false-fire not measurable)\n")
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()

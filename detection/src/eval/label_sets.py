"""
One entry point for every labelled set, with absolute paths (works from any directory).

    python3 detection/src/eval/label_sets.py status
    python3 detection/src/eval/label_sets.py review 0715 --verify              # at-risk frames, prev label pre-filled
    python3 detection/src/eval/label_sets.py review 0708 --verify --only all   # everything
    python3 detection/src/eval/label_sets.py review 0730                        # continue unlabelled frames

Extra arguments after the set name are passed to gt_boxes.py review (--verify, --only, --redo-capped).
"""

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DET = os.path.join(ROOT, "detection")
NS = os.path.join(DET, "data", "new_sessions")
CLIP = os.path.join(ROOT, "data", "new_videos", "clipped")

SETS = {
    "0708": {"role": "test (eval video)",
             "video": os.path.join(DET, "data", "videos", "thermal.mp4"),
             "manifest": os.path.join(DET, "data", "recall_ground_truth", "manifest.csv"),
             "proposals": os.path.join(DET, "data", "recall_ground_truth", "proposals.json"),
             "boxes": os.path.join(DET, "data", "recall_ground_truth", "boxes.csv"),
             "registration": os.path.join(ROOT, "data", "vid_pairs", "registration.json"),
             "day": os.path.join(DET, "data", "videos", "day.mp4")},
    "0623": {"role": "train/val",
             "video": os.path.join(CLIP, "2026-06-23_scenario2_thermal_clipped.mp4"),
             "manifest": os.path.join(NS, "0623_manifest.csv"),
             "proposals": os.path.join(NS, "0623_proposals.json"),
             "boxes": os.path.join(NS, "0623_boxes.csv"),
             "registration": os.path.join(ROOT, "data", "vid_pairs_0623", "registration.json"),
             "day": os.path.join(CLIP, "2026-06-23_scenario2_day_clipped.mp4")},
    "0715": {"role": "train/val",
             "video": os.path.join(CLIP, "2026-07-15_thermal_clipped.mp4"),
             "manifest": os.path.join(NS, "0715_manifest.csv"),
             "proposals": os.path.join(NS, "0715_proposals.json"),
             "boxes": os.path.join(NS, "0715_boxes.csv"),
             "registration": os.path.join(ROOT, "data", "vid_pairs_0715", "registration.json"),
             "day": os.path.join(CLIP, "2026-07-15_day_clipped.mp4")},
    "0730": {"role": "test (second session, thermal only)",
             "video": os.path.join(CLIP, "2026-07-30T09_27_04_thermal_clipped.mp4"),
             "manifest": os.path.join(NS, "0730_test_manifest.csv"),
             "proposals": os.path.join(NS, "0730_test_proposals.json"),
             "boxes": os.path.join(NS, "0730_boxes.csv"),
             "registration": None, "day": None},
}


def status():
    for name, s in SETS.items():
        m = pd.read_csv(s["manifest"])
        if not os.path.exists(s["boxes"]):
            print(f"{name} [{s['role']}]: 0/{len(m)} labelled")
            continue
        b = pd.read_csv(s["boxes"]).sort_values("frame_id").reset_index(drop=True)
        ver = b["verified"].fillna("").eq("v2") if "verified" in b else pd.Series(False, index=b.index)
        drone = b.verdict.eq("drone")
        capped = drone & (((b.x1 - b.x0) >= 50) | ((b.y1 - b.y0) >= 50))
        v = b.verdict.tolist()
        susp = pd.Series([v[j] == "nothing" and ((j > 0 and v[j - 1] == "drone") or (j + 1 < len(v) and v[j + 1] == "drone"))
                          for j in range(len(v))])
        at_risk = (capped | b.verdict.isin(["nothing", "unsure"])) & ~ver
        print(f"{name} [{s['role']}]: {len(b)}/{len(m)} labelled {b.verdict.value_counts().to_dict()}")
        print(f"      verified v2: {int(ver.sum())} | capped: {int(capped.sum())} | suspect 'nothing': {int(susp.sum())}"
              f" | still at risk (capped/nothing/unsure, not verified): {int(at_risk.sum())}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "review"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "status":
        return status()
    name = sys.argv[2]
    s = SETS[name]
    argv = ["gt_boxes.py", "review", "--video", s["video"], "--manifest", s["manifest"],
            "--proposals", s["proposals"], "--out", s["boxes"]]
    if s["registration"]:
        argv += ["--registration", s["registration"], "--day", s["day"]]
    sys.argv = argv + sys.argv[3:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gt_boxes
    gt_boxes.main()


if __name__ == "__main__":
    main()

r"""
Undo part of a bulk "not airborne" edit in a tools/labeling work dir: copy the labels of frames [first, last] of one camera back
from a backup labels file (the backup the bulk edit wrote), into labels.json. labels.json is backed up first.

2026-11-02: 07_38 was bulk-marked "not airborne" from thermal frame 32025 / day frame 32000 on; Daniel then gave 7:50-18:20 as
airborne (frames 14100-33000), so 17:42-18:20 needs its auto labels back for review.

    python3 src/data/restore_auto_range.py --work ~/label_work/f0738_1020 --cam thermal \
        --backup labels.before_none_thermal_32025_36507.json --first 32025 --last 33001
"""
import os
import json
import shutil
import argparse
import datetime


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--cam", required=True, choices=["day", "thermal"])
    ap.add_argument("--backup", required=True, help="file name inside --work"); ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    a = ap.parse_args()
    work = os.path.expanduser(a.work)
    lp = os.path.join(work, "labels.json")
    L, B = json.load(open(lp)), json.load(open(os.path.join(work, a.backup)))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(lp, os.path.join(work, f"labels.before_restore_{a.cam}_{a.first}_{a.last}_{stamp}.json"))
    n = 0
    for f in range(a.first, a.last + 1):
        k = str(f)
        if k in B[a.cam]:
            L[a.cam][k] = B[a.cam][k]; n += 1
    L["meta"].setdefault("restored", []).append({"cam": a.cam, "frames": [a.first, a.last], "from": a.backup, "when": stamp,
                                                  "why": "Daniel 2026-11-02: airborne until 18:20"})
    json.dump(L, open(lp, "w"))
    print(f"restored {n} {a.cam} frames {a.first}-{a.last} from {a.backup}; backup of the previous labels.json written")


if __name__ == "__main__":
    main()

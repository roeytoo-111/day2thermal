"""Day-based train / val / test split of the registered RGB->thermal pairs for diffusion training.

Why by day, not random: pairs are sampled ~1-2 s apart from continuous video, so neighbours are
near-duplicates. A random val/test would share scenes with train and measure memorisation. The question
that decided pix2pix (NO-GO, notebook 2026-09-29) was "does it work on a day it has never seen?", so:

  test  = every pair of --test-day              (hidden until the end; same protocol as pix2pix)
  val   = the last --val-per-day pairs of each other day, after a --gap of pairs dropped
          (contiguous, so it is not a near-copy of train)
  train = the rest

Sources are "day:dir" with dir/{train,val}/{rgb,thermal}/ as written by `video_pairs extract`.

    python -m day2thermal.diffusion.prep --out data/diff_pairs --test-day 0715 \\
        --source 0708:data/vid_pairs --source 0623:data/vid_pairs_0623 --source 0715:data/vid_pairs_0715
"""
import argparse
import json
import os
import re
import shutil

from ..utils import ensure_dir, list_images


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def frame_of(name):
    m = re.search(r"(\d+)\.png$", name)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="day:dir, repeatable")
    ap.add_argument("--test-day", default=None, help="hold out this whole day/source as test (unseen-day test)")
    ap.add_argument("--test-tail-frac", type=float, default=0.0,
                    help="instead: the last fraction of EVERY source is test (contiguous, after --gap); covers all "
                         "content types, but is not an unseen-day test")
    ap.add_argument("--oversample", default="",
                    help="train oversampling by RGB content, e.g. 'terrain:3,mixed:2' (hard-linked copies)")
    ap.add_argument("--val-per-day", type=int, default=20, help="contiguous pairs at the end of each train day")
    ap.add_argument("--gap", type=int, default=5, help="pairs dropped between a day's train and val parts")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; remove it first.")
    split = {"train": [], "val": [], "test": [], "dropped_gap": []}
    for spec in a.source:
        day, d = spec.split(":", 1)
        items = []
        for sub in ("train", "val"):
            rd = os.path.join(d, sub, "rgb")
            if os.path.isdir(rd):
                items += [(frame_of(n), os.path.join(d, sub, "rgb", n), os.path.join(d, sub, "thermal", n), n)
                          for n in list_images(rd)]
        items.sort()
        if a.test_day is not None and day == a.test_day:
            split["test"] += [(day,) + it for it in items]
            continue
        if a.test_tail_frac > 0:
            n_test = max(int(round(a.test_tail_frac * len(items))), 1)
            split["test"] += [(day,) + it for it in items[len(items) - n_test:]]
            split["dropped_gap"] += [(day,) + it for it in items[max(len(items) - n_test - a.gap, 0):len(items) - n_test]]
            items = items[:max(len(items) - n_test - a.gap, 0)]
        n_val = min(a.val_per_day, max(len(items) // 5, 1))
        cut = len(items) - n_val
        split["train"] += [(day,) + it for it in items[:max(cut - a.gap, 0)]]
        split["dropped_gap"] += [(day,) + it for it in items[max(cut - a.gap, 0):cut]]
        split["val"] += [(day,) + it for it in items[cut:]]
    for sp in ("train", "val", "test"):
        for m in ("rgb", "thermal"):
            ensure_dir(os.path.join(a.out, sp, m))
        for day, _, rgb, th, name in split[sp]:
            link_or_copy(rgb, os.path.join(a.out, sp, "rgb", name))
            link_or_copy(th, os.path.join(a.out, sp, "thermal", name))
    if a.oversample:
        import cv2
        from .diagnose import sky_fraction
        mult = {k: int(v) for k, v in (x.split(":") for x in a.oversample.split(","))}
        added = {k: 0 for k in mult}
        for day, _, rgb, th, name in split["train"]:
            sf = sky_fraction(cv2.imread(rgb))
            kind = "terrain" if sf < 0.2 else ("mixed" if sf <= 0.5 else "sky")
            for k in range(1, mult.get(kind, 1)):
                stem, ext = os.path.splitext(name)
                link_or_copy(rgb, os.path.join(a.out, "train", "rgb", f"{stem}__os{k}{ext}"))
                link_or_copy(th, os.path.join(a.out, "train", "thermal", f"{stem}__os{k}{ext}"))
                added[kind] += 1
        print("oversampling copies added:", added)
    summary = {sp: {"n": len(v), "by_day": {d: sum(1 for x in v if x[0] == d) for d in sorted({x[0] for x in v})}}
               for sp, v in split.items()}
    summary["args"] = vars(a)
    with open(os.path.join(a.out, "split.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()

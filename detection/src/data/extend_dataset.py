r"""
Grow a dataset WITHOUT touching its existing split, so models along the ladder (fw4 -> v5a -> v5 -> ...) stay comparable
on the same test images. The base train/valid/test are hard-linked unchanged; each added folder (YOLO train/ layout,
class order from its data.yaml, remapped to --names) is split one of two ways, and its held-out part becomes a NEW
test set of its own (<out>/extra_tests/<name>/), never mixed into the base test:

  --add-time NAME:DIR   video tiles (stems <prefix><session>_<frame6>_<k>): per session, frames in the last
                        --tail-frac go to the new test set, a --gap-frac gap before them is dropped, the rest -> train.
  --add-frac NAME:DIR   independent images: --hold-frac of them (by stable hash of the source key) -> new test, rest -> train.

    python3 src/data/extend_dataset.py --base ../data/rgb_v5a --out ../data/rgb_v5 \
        --add-time flight1014:../data/flight_20261014/rgb_tracks_addon --add-frac oi_sky:../data/oi_negatives
"""
import os
import sys
import shutil
import hashlib
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgb_ds as R   # noqa: E402


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def put(img, lab_lines, d):
    stem = os.path.splitext(os.path.basename(img))[0]
    link(img, os.path.join(d, "images", os.path.basename(img)))
    open(os.path.join(d, "labels", stem + ".txt"), "w").write("".join(lab_lines))


def remapped(img, m):
    return [f"{m[c]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in R.read_labels(R.label_path(img)) if c in m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--add-time", action="append", default=[]); ap.add_argument("--add-frac", action="append", default=[])
    ap.add_argument("--names", default="uav,airplane,bird")
    ap.add_argument("--tail-frac", type=float, default=0.15); ap.add_argument("--gap-frac", type=float, default=0.05)
    ap.add_argument("--hold-frac", type=float, default=0.3)
    a = ap.parse_args()
    names = a.names.split(",")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    rows = []
    for s in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, sub))
        for p in R.list_split(a.base, s):
            link(p, os.path.join(a.out, s, "images", os.path.basename(p)))
            link(R.label_path(p), os.path.join(a.out, s, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"))
        rows.append({"part": "base", "split": s, "images": len(R.list_split(a.base, s))})
    for kind, specs in (("time", a.add_time), ("frac", a.add_frac)):
        for spec in specs:
            name, d = spec.split(":", 1)
            root = R.find_root(d); src_names = R.read_names(root)
            m = {i: names.index(n) for i, n in enumerate(src_names) if n in names}
            imgs = R.list_split(root, "train")
            tdir = os.path.join(a.out, "extra_tests", name)
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(tdir, sub), exist_ok=True)
            keys = pd.Series([R.source_key(p) for p in imgs])
            dest = []
            if kind == "time":
                fr = np.array([R.frame_number(p) for p in imgs])
                for k in keys.unique():
                    idx = np.nonzero(keys.values == k)[0]
                    lo, hi = np.quantile(fr[idx], 1 - a.tail_frac - a.gap_frac), np.quantile(fr[idx], 1 - a.tail_frac)
                    for i in idx:
                        dest.append((i, "test" if fr[i] > hi else "gap" if fr[i] > lo else "train"))
            else:
                for i, k in enumerate(keys):
                    h = int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    dest.append((i, "test" if h < a.hold_frac else "train"))
            n = {"train": 0, "test": 0, "gap": 0}
            for i, where in dest:
                n[where] += 1
                if where == "train":
                    put(imgs[i], remapped(imgs[i], m), os.path.join(a.out, "train"))
                elif where == "test":
                    put(imgs[i], remapped(imgs[i], m), tdir)
            open(os.path.join(tdir, "data.yaml"), "w").write(
                f"path: {os.path.abspath(tdir)}\ntrain: images\nval: images\ntest: images\n\nnc: {len(names)}\nnames: {names}\n")
            rows.append({"part": name, "split": f"train {n['train']} / new test {n['test']} / gap dropped {n['gap']}", "images": len(imgs)})
    open(os.path.join(a.out, "data.yaml"), "w").write(
        f"path: {os.path.abspath(a.out)}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: {len(names)}\nnames: {names}\n")
    T = pd.DataFrame(rows)
    txt = "# Extended dataset\n\nBase split unchanged (hard links).\n\n" + "\n".join(f"- {r.part}: {r.split} ({r.images} images)" for r in T.itertuples())
    open(os.path.join(a.out, "EXTEND.md"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()

r"""
Simulator ablation for the RGB detector. The 7,120 simulator renders (stems img_NNNN / img_NNNNNN, Roboflow copies of
1,495 pictures from two frame sequences 0-499 and 1-1000) are train-only in rgb_v5a. This builds, from rgb_v5a with
hard links, two training sets that differ ONLY in the simulator, plus a simulator test set neither trains on:

  <out>/with_sim/     rgb_v5a minus the sim TAIL (last --tail-frac of each sequence by picture number, and a --gap-frac gap
                      before it, all dropped from train: consecutive frames are near-identical)
  <out>/no_sim/       rgb_v5a minus every simulator image
  <out>/sim_tail/     the tail itself (images + labels, data.yaml): the test of "does the model work on unseen sim frames"
Both training sets keep rgb_v5a's valid/test unchanged, so real-data scores stay comparable with v5a / fw4.
Roboflow copies of one picture share a number, so they stay together.

    python3 src/data/make_sim_ablation.py --base ../data/rgb_v5a --out ../data/rgb_sim
"""
import os
import re
import sys
import shutil
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgb_ds as R   # noqa: E402

SIM = re.compile(r"^img_(\d+)_png")


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tail-frac", type=float, default=0.15); ap.add_argument("--gap-frac", type=float, default=0.05)
    a = ap.parse_args()
    names = R.read_names(a.base)
    sim = {}
    for p in R.list_split(a.base, "train"):
        m = SIM.match(os.path.basename(p))
        if m:
            digits = m.group(1); sim[p] = (len(digits) >= 6, int(digits))
    seq = {k: np.array(sorted({n for kk, n in sim.values() if kk == k})) for k in (False, True)}
    cut = {k: (np.quantile(v, 1 - a.tail_frac - a.gap_frac), np.quantile(v, 1 - a.tail_frac)) for k, v in seq.items()}
    role = {}
    for p, (k, n) in sim.items():
        lo, hi = cut[k]
        role[p] = "tail" if n > hi else "gap" if n > lo else "train"
    print({k: (int(v.min()), int(v.max()), len(v)) for k, v in seq.items()}, "->", {r: sum(1 for x in role.values() if x == r) for r in ("train", "tail", "gap")})
    for name in ("with_sim", "no_sim", "sim_tail"):
        d = os.path.join(a.out, name)
        if os.path.exists(d):
            raise SystemExit(f"{d} exists")
    for name in ("with_sim", "no_sim"):
        for s in ("train", "valid", "test"):
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(a.out, name, s, sub))
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out, "sim_tail", sub))
    n = {"with_sim": 0, "no_sim": 0}
    for s in ("train", "valid", "test"):
        for p in R.list_split(a.base, s):
            lab = R.label_path(p); stem = os.path.splitext(os.path.basename(p))[0]
            r = role.get(p) if s == "train" else None
            targets = []
            if r is None:
                targets = ["with_sim", "no_sim"]
            elif r == "train":
                targets = ["with_sim"]
            elif r == "tail":
                dst = os.path.join(a.out, "sim_tail")
                link(p, os.path.join(dst, "images", os.path.basename(p))); link(lab, os.path.join(dst, "labels", stem + ".txt"))
            for t in targets:
                link(p, os.path.join(a.out, t, s, "images", os.path.basename(p)))
                link(lab, os.path.join(a.out, t, s, "labels", stem + ".txt"))
                n[t] += s == "train"
    for name in ("with_sim", "no_sim"):
        open(os.path.join(a.out, name, "data.yaml"), "w").write(
            f"path: {os.path.abspath(os.path.join(a.out, name))}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: {len(names)}\nnames: {names}\n")
    d = os.path.join(a.out, "sim_tail")
    open(os.path.join(d, "data.yaml"), "w").write(
        f"path: {os.path.abspath(d)}\ntrain: images\nval: images\ntest: images\n\nnc: {len(names)}\nnames: {names}\n")
    print("train images:", n, "| sim_tail test:", len(os.listdir(os.path.join(d, "images"))))


if __name__ == "__main__":
    main()

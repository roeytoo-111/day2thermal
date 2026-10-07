r"""
Copy a thermal YOLO dataset to the uav = 0 convention: the old thermal sets are [bird, thermal-uav] (uav = 1); the
copy is [uav, bird] (labels 0 <-> 1 swapped, polygons kept). Images are hard links to the source, so only labels take
space (not a folder symlink: Ultralytics resolves it and would read the source's old-order labels). Counts per class are printed for both, so the swap can be checked.

    python3 src/data/swap_uav_class.py --src data/thermal-4-boson3PR --out data/thermal-5-boson3PR-u0
"""
import os
import glob
import argparse
from collections import Counter

SWAP = {0: 1, 1: 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    before, after = Counter(), Counter()
    for split in ("train", "valid", "test"):
        if not os.path.isdir(os.path.join(src, split, "images")):
            continue
        os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        for im in glob.glob(os.path.join(src, split, "images", "*")):    # hard links: a symlinked folder is
            dst = os.path.join(out, split, "images", os.path.basename(im))  # resolved by Ultralytics back to the
            if not os.path.exists(dst):                                  # source labels (old class order)
                os.link(os.path.realpath(im), dst)
        for f in glob.glob(os.path.join(src, split, "labels", "*.txt")):
            lines = []
            for l in open(f):
                p = l.split()
                if not p:
                    continue
                c = int(p[0]); before[c] += 1
                c = SWAP.get(c, c); after[c] += 1
                lines.append(" ".join([str(c)] + p[1:]))
            open(os.path.join(out, split, "labels", os.path.basename(f)), "w").write(
                "\n".join(lines) + ("\n" if lines else ""))
    open(os.path.join(out, "data.yaml"), "w").write(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\nnames:\n- uav\n- bird\n")
    print(f"{a.src} -> {a.out}: classes before {dict(before)} (0=bird, 1=uav), after {dict(after)} (0=uav, 1=bird)")


if __name__ == "__main__":
    main()

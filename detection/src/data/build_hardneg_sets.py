r"""
Build the N1 / N2 / N3 thermal datasets (notebook 2026-10-12 section 2) from M0's uav = 0 set, with hard links
(a folder symlink would be resolved by Ultralytics back to the source labels):

  N1 = base + mined hard Boson frames (mine_thermal_hard.py, part A) + a second copy of every hard existing real
       training image (part B, prefix "hd_") + real drone-free backgrounds as empty images (make_clean_negatives.py)
  N2 = base with half of the pasted images (bgpaste_ / bgboson_) removed (seeded)
  N3 = N2 + the N1 additions (hard copies only for images N2 still has)
valid / test are the base's, unchanged. Counts go to <out>/build_config.json.

    python3 src/data/build_hardneg_sets.py --base data/thermal-5-bosonP-u0 --hard data/boson_work/hardneg \
        --out-prefix data/thermal-7
"""
import os
import glob
import json
import random
import argparse
import shutil

PASTE = ("bgpaste_", "bgboson_")


def link(src, dst):
    if not os.path.exists(dst):
        os.link(os.path.realpath(src), dst)


def lab_of(img):
    return img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"


def build(out, base, keep_train, extra_pairs, dup_names, note):
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(f"{out}/{split}/{sub}", exist_ok=True)
    for split in ("valid", "test"):
        for f in glob.glob(f"{base}/{split}/images/*"):
            link(f, f"{out}/{split}/images/{os.path.basename(f)}")
            if os.path.exists(lab_of(f)):
                link(lab_of(f), f"{out}/{split}/labels/{os.path.basename(lab_of(f))}")
    n = {"base_train_kept": 0, "extra": 0, "hard_duplicates": 0}
    for f in keep_train:
        link(f, f"{out}/train/images/{os.path.basename(f)}")
        if os.path.exists(lab_of(f)):
            link(lab_of(f), f"{out}/train/labels/{os.path.basename(lab_of(f))}")
        n["base_train_kept"] += 1
    for img, lab in extra_pairs:
        link(img, f"{out}/train/images/{os.path.basename(img)}"); link(lab, f"{out}/train/labels/{os.path.basename(lab)}")
        n["extra"] += 1
    kept = {os.path.basename(f) for f in keep_train}
    for name in dup_names:
        if name not in kept:
            continue
        src = f"{base}/train/images/{name}"
        link(src, f"{out}/train/images/hd_{name}")
        if os.path.exists(lab_of(src)):
            shutil.copy(lab_of(src), f"{out}/train/labels/hd_{os.path.basename(lab_of(src))}")
        else:
            open(f"{out}/train/labels/hd_{name.rsplit('.', 1)[0]}.txt", "w").close()
        n["hard_duplicates"] += 1
    shutil.copy(f"{base}/data.yaml", f"{out}/data.yaml")
    n["train_total"] = len(os.listdir(f"{out}/train/images"))
    json.dump({"note": note, **n}, open(f"{out}/build_config.json", "w"), indent=1)
    print(out, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--hard", required=True)
    ap.add_argument("--out-prefix", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg", default=None, help="make_clean_negatives.py output (images/, labels/)")
    a = ap.parse_args()
    train = sorted(glob.glob(f"{a.base}/train/images/*"))
    pastes = [f for f in train if os.path.basename(f).startswith(PASTE)]
    real = [f for f in train if not os.path.basename(f).startswith(PASTE)]
    rng = random.Random(a.seed); half = set(rng.sample(pastes, len(pastes) // 2))
    extra = [(f, lab_of(f)) for f in sorted(glob.glob(f"{a.hard}/boson/images/*"))]
    dup = [l.strip() for l in open(f"{a.hard}/hard_existing.txt") if l.strip()]
    if a.neg:
        extra += [(f, lab_of(f)) for f in sorted(glob.glob(f"{a.neg}/images/*"))]
    build(f"{a.out_prefix}-N1-u0", a.base, train, extra, dup, "base + mined hard Boson frames + hard real images x2 + clean real negatives")
    build(f"{a.out_prefix}-N2-u0", a.base, real + sorted(half), [], [], f"base with {len(pastes) - len(half)} of {len(pastes)} pastes removed")
    build(f"{a.out_prefix}-N3-u0", a.base, real + sorted(half), extra, dup, "N2 + N1 additions")


if __name__ == "__main__":
    main()

r"""
Re-split RGB detection data so that no video / photo series and no near-duplicate picture is in two splits
(Fixed_wing_v3: ~58% of valid images share a source with train, ~23% have a near-duplicate there; see rgb_audit.py).

  1. Pool every split of every --data folder (plus their labels, or the corrected ones from --labels-override).
  2. Groups = union of (same source key, rgb_ds.source_key) and (near-duplicate: coarse dHash Hamming <= --dup-thr
     confirmed by high-pass thumbnail correlation, see rgb_ds.fingerprint).
  3. --force split:REGEX pins groups whose source key matches (e.g. our held-out Boson sessions -> test).
     Groups larger than --max-eval-group images go to train (one video must not dominate valid/test), except that a
     long single video is split in time: last --tail-frac may go to valid/test, a --gap-frac gap before it is dropped.
     Balancing targets are fractions of all data except what --force pins to train (simulator renders would set
     small-drone targets no real split can meet); a group goes to the split with the largest relative deficit in
     the strata it contains.
  4. Remaining groups, largest first, go to the split that most needs what they contain, by stratum:
     uav <16 px, uav 16-64, uav >= 64, airplane, bird, empty image (rare strata weigh more), so valid/test get
     their share of small drones and negatives (v3 valid: 0.7% of uav < 16 px vs 5.2% in train).
  5. valid/test keep one image per near-duplicate cluster (train keeps all).
Images are hard-linked (copied across file systems); labels are rewritten in the --names order (uav = 0).
Writes <out>/{train,valid,test}/{images,labels}, data.yaml, split_manifest.csv, SPLIT.md.

    python3 src/data/resplit_by_source.py --data ../ANN-Detection/data/datasets/Fixed_wing_v4_20261011_191146 \
        --labels-override data/rgb_fixes/v4/labels --exclude data/rgb_fixes/v4/excluded.txt \
        --force "test:^boson(08_58_15|10_15_59)$" --out ../ANN-Detection/data/datasets/Fixed_wing_v5
"""
import os
import re
import sys
import shutil
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgb_ds as R   # noqa: E402

STRATA = ["uav_s", "uav_m", "uav_l", "airplane", "bird", "empty"]


def scan(args):
    path, lab = args
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    H, W = g.shape
    c, f, flat = R.fingerprint(g)
    return path, lab, c, W, H, f, flat


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        if i != j:
            self.p[max(i, j)] = min(i, j)


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="uav,airplane,bird", help="output class order")
    ap.add_argument("--in-names", default=None, help="input class order if a --data folder has no data.yaml "
                                                     "(default: uav,airplane,bird)")
    ap.add_argument("--labels-override", default=None, help="folder of corrected <stem>.txt (review_label_fixes --finalize)")
    ap.add_argument("--exclude", default=None, help="file with image stems to drop (out of scope / broken)")
    ap.add_argument("--frac", default="0.80,0.12,0.08", help="train,valid,test")
    ap.add_argument("--force", action="append", default=[], help="split:REGEX on the source key (repeatable)")
    ap.add_argument("--max-eval-group", type=int, default=150)
    ap.add_argument("--tail-frac", type=float, default=0.15); ap.add_argument("--gap-frac", type=float, default=0.05)
    ap.add_argument("--dup-thr", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="write SPLIT.md / manifest only")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    out_names = a.names.split(",")
    excl = set(x.strip() for x in open(a.exclude)) if a.exclude else set()
    jobs, remap = [], {}
    if a.labels_override:            # corrected labels carry their own class order (review_label_fixes: uav,airplane,bird)
        a.labels_override = os.path.abspath(a.labels_override)
        np_ = os.path.join(a.labels_override, "_names.txt")
        ovr = open(np_).read().strip().split(",") if os.path.exists(np_) else R.DEFAULT_NAMES
        m_ovr = {i: out_names.index(x) for i, x in enumerate(ovr) if x in out_names}
    for d in a.data:
        root = R.find_root(d)
        has_yaml = os.path.exists(os.path.join(root, "data.yaml"))
        names = R.read_names(root) if has_yaml or not a.in_names else a.in_names.split(",")
        m = {i: out_names.index(n) for i, n in enumerate(names) if n in out_names}
        print(f"{root}: names {names} -> {out_names}", flush=True)
        for s in ("train", "valid", "test", "test_boson_heldout"):
            for p in R.list_split(root, s):
                if os.path.splitext(os.path.basename(p))[0] in excl:
                    continue
                lp = R.label_path(p, a.labels_override)
                jobs.append((p, lp)); remap[p] = m_ovr if a.labels_override and lp.startswith(os.path.abspath(a.labels_override)) else m
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap(scan, jobs, chunksize=64) if r]
    n = len(rows); print(f"{n} images pooled ({len(excl)} excluded stems)", flush=True)
    stems = [os.path.splitext(os.path.basename(r[0]))[0] for r in rows]
    dup = pd.Series(stems).duplicated(keep=False)
    if dup.any():
        raise SystemExit(f"stem collision across inputs, e.g. {pd.Series(stems)[dup].iloc[0]}")
    uf = UF(n); first = {}
    srcs = [R.source_key(r[0], (r[3], r[4])) for r in rows]
    for i, s in enumerate(srcs):
        if s in first:
            uf.union(i, first[s])
        else:
            first[s] = i
    hashes = np.array([r[2] for r in rows], np.uint64)
    hp = np.stack([r[5] for r in rows]); en = np.array([r[6] for r in rows])
    pairs = R.confirm_pairs(R.hamming_pairs(hashes, hashes, a.dup_thr, same=True, chunk=1024), hp, hp, en, en)
    for i, j in pairs:
        uf.union(int(i), int(j))
    print(f"{len(first)} sources, {len(pairs)} near-duplicate pairs", flush=True)

    feats, labs = np.zeros((n, len(STRATA))), []
    for i, (p, lp, h, W, H, _, _) in enumerate(rows):
        L = [(remap[p][c], *b) for c, *b in R.read_labels(lp) if c in remap[p]]
        labs.append(L)
        if not L:
            feats[i, 5] = 1
        for c, *b in L:
            nm = out_names[c]
            if nm == "uav":
                px = R.box_px((c, *b), W, H); feats[i, 0 if px < 16 else 1 if px < 64 else 2] += 1
            elif nm in ("airplane", "bird"):
                feats[i, STRATA.index(nm)] += 1
    gid = np.array([uf.find(i) for i in range(n)])
    split_names = ["train", "valid", "test"]; frac = np.array([float(x) for x in a.frac.split(",")])
    forced = [(s.split(":", 1)[0], re.compile(s.split(":", 1)[1])) for s in a.force]
    pin_of = lambda gsrc: next((sp for sp, rx in forced if any(rx.search(x) for x in gsrc)), None)

    # long single videos (> --max-eval-group frames): body -> train, a --gap-frac gap dropped, the last --tail-frac
    # becomes its own group that may go to valid/test (real footage of a kind that only exists in long videos, e.g. the
    # 07-08 4K flight with most of the real small drones, would otherwise never be evaluated)
    gap = np.zeros(n, bool); tail_g, body_g = set(), set()
    for g in np.unique(gid):
        idx = np.nonzero(gid == g)[0]
        gs = {srcs[i] for i in idx}
        if len(idx) <= a.max_eval_group or len(gs) != 1 or pin_of(gs) or a.tail_frac <= 0:
            continue
        fr = np.array([R.frame_number(rows[i][0]) for i in idx], float)
        if np.isnan(fr).mean() > 0.05:
            continue
        lo, hi = np.nanquantile(fr, 1 - a.tail_frac - a.gap_frac), np.nanquantile(fr, 1 - a.tail_frac)
        t = idx[fr > hi]; gp = idx[(fr > lo) & (fr <= hi)]
        if len(t):
            gid[t] = n + g; tail_g.add(n + g); gap[gp] = True; body_g.add(g)
    why = {}
    groups = pd.DataFrame({"gid": gid, "src": srcs}).groupby("gid")
    order = []
    rng = np.random.default_rng(a.seed)
    for g, d in groups:
        idx = d.index.values[~gap[d.index.values]]
        if len(idx):
            order.append((len(idx), rng.random(), g, idx, set(d.src)))
    order.sort(key=lambda x: (-x[0], x[1]))
    # targets = fractions of all data except what --force pins to train (the simulator renders: 78% of v4's small
    # drones, which no valid/test split could match). Long videos count towards train.
    plan = {}
    for size, _, g, idx, gsrc in order:
        pin = pin_of(gsrc)
        plan[g] = ("forced", split_names.index(pin)) if pin else ("large group", 0) if (size > a.max_eval_group and g not in tail_g) or g in body_g else None
    total = sum(feats[idx].sum(0) for size, _, g, idx, _ in order if not (plan[g] and plan[g][0] == "forced" and plan[g][1] == 0)) + 1e-9
    w = 1 / np.sqrt(total)                                        # rarer strata count more
    target = frac[:, None] * total[None, :]
    cur = np.zeros((3, len(STRATA))); cur_n = np.zeros(3); assign = {}
    for size, _, g, idx, gsrc in order:
        if plan[g]:
            why[g], k = plan[g]; assign[g] = k; cur_n[k] += size
            if not (why[g] == "forced" and k == 0):
                cur[k] += feats[idx].sum(0)
    for size, _, g, idx, gsrc in order:
        if plan[g]:
            continue
        f = feats[idx].sum(0)
        deficit = (target - cur) / target                     # relative room left per split and stratum
        score = (deficit * (f * w)[None, :]).sum(1) / max((f * w).sum(), 1e-9)
        k = int(np.argmax(score + 1e-9 * rng.random(3))); why[g] = "video tail" if g in tail_g else "balanced"
        assign[g] = k; cur[k] += f; cur_n[k] += size

    keep = ~gap
    for i, j in pairs:                                           # eval splits: one picture per near-dup cluster
        if gid[i] in assign and gid[j] in assign and assign[gid[i]] > 0 and keep[i] and keep[j]:
            keep[j] = False
    man = pd.DataFrame({"image": [r[0] for r in rows], "stem": stems, "source": srcs, "group": gid,
                        "split": [split_names[assign[g]] if g in assign else "gap" for g in gid], "kept": keep,
                        "reason": ["gap" if gap[i] else why[g] for i, g in enumerate(gid)]})
    for k, s in enumerate(STRATA):
        man[s] = feats[:, k].astype(int)
    os.makedirs(a.out, exist_ok=True)
    man.to_csv(os.path.join(a.out, "split_manifest.csv"), index=False)
    M = man[man.kept]
    T = ["# Split by source", "", f"Inputs: {', '.join(a.data)}", f"Labels override: {a.labels_override}; excluded stems: {len(excl)}",
         f"Forced: {a.force or 'none'}; groups > {a.max_eval_group} images -> train; near-dup threshold dHash <= {a.dup_thr}", "",
         "| | images | groups | " + " | ".join(STRATA) + " |", "|---|---|---|" + "---|" * len(STRATA)]
    for s in split_names:
        d = M[M.split == s]
        T.append(f"| {s} | {len(d):,} | {d.group.nunique():,} | " + " | ".join(f"{int(d[c].sum()):,}" for c in STRATA) + " |")
    T += ["", f"Dropped from valid/test as near-duplicates: {int((~man.kept & ~gap).sum()):,}; video-tail gap frames dropped: {int(gap.sum()):,}",
          f"Long videos split in time (last {a.tail_frac:.0%} eligible for valid/test): {len(tail_g)}",
          f"Groups sent to train for size: {sum(1 for v in why.values() if v == 'large group')}"]
    open(os.path.join(a.out, "SPLIT.md"), "w").write("\n".join(T) + "\n"); print("\n".join(T))
    if a.dry_run:
        return
    for s in split_names:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, sub), exist_ok=True)
    for i in np.nonzero(keep)[0]:
        s = split_names[assign[gid[i]]]; p = rows[i][0]
        dst = os.path.join(a.out, s, "images", os.path.basename(p))
        if not os.path.exists(dst):
            link(p, dst)
        open(os.path.join(a.out, s, "labels", stems[i] + ".txt"), "w").write(
            "".join(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for c, *b in labs[i]))
    open(os.path.join(a.out, "data.yaml"), "w").write(
        f"path: {os.path.abspath(a.out)}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(out_names)}\nnames: [{', '.join(repr(x) for x in out_names)}]\n")
    print("written", a.out)


if __name__ == "__main__":
    main()

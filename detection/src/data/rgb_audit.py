r"""
Audit an RGB detection dataset (Fixed_wing layout: train/valid/test with images/ + labels/) for coverage holes and
split leakage. Writes <out>/AUDIT.md plus images.csv / boxes.csv (reused by resplit_by_source.py).

What is measured, per split:
  * images, empty images, boxes per class
  * uav / airplane / bird target size in model-input px (sqrt(w*h) after letterbox to 640; the fw4 README bins)
  * uav background: the ring around each box is "smooth" (sky, haze, flat cloud) or "textured" (terrain, cumulus,
    trees, buildings), and the drone's contrast against that ring (|mean box - mean ring| in gray levels;
    < 12 = low contrast). Small + low-contrast + textured is the fw4 Boson failure (drones against cumulus).
  * scene lighting (mean luminance: dark < 60, bright > 190) and composition (share of smooth 8x8 thumbnail blocks)
  * sources (video / photo series, rgb_ds.source_key): count, concentration
  * leakage: sources present in two splits, and near-duplicate pictures across splits (dHash Hamming <= --dup-thr,
    confirmed by high-pass thumbnail correlation, see rgb_ds.fingerprint)
Cells with fewer than --min-cell uav boxes in train (or a valid/test share under a third of train's) are flagged
as holes.

    python3 src/data/rgb_audit.py --data ../ANN-Detection/data/datasets/Fixed_wing_v3 --out results/rgb_audit/v3
"""
import os
import sys
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgb_ds as R   # noqa: E402

SPLITS = ("train", "valid", "test")


def analyse(args):
    path, split, lab_path = args
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, []
    H, W = g.shape
    th = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    blocks = th.reshape(8, 8, 8, 8).std(axis=(1, 3))
    img = {"path": path, "split": split, "source": R.source_key(path), "W": W, "H": H, "lum": float(g.mean()),
           "smooth_frac": float((blocks < 6).mean())}
    img["dhash"], hp, img["detail"] = R.fingerprint(g); img["_hp"] = hp
    boxes = []
    for c, cx, cy, w, h in R.read_labels(lab_path):
        x0, x1 = int(max(0, (cx - w / 2) * W)), int(min(W, (cx + w / 2) * W))
        y0, y1 = int(max(0, (cy - h / 2) * H)), int(min(H, (cy + h / 2) * H))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        pad = max(4, int(0.75 * max(x1 - x0, y1 - y0)))
        X0, X1, Y0, Y1 = max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad)
        ring = g[Y0:Y1, X0:X1].astype(np.float32).copy()
        inner = g[y0:y1, x0:x1].astype(np.float32)
        m = np.ones(ring.shape, bool); m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        rv = ring[m] if m.any() else ring.ravel()
        boxes.append({"path": path, "split": split, "cls": c, "px": R.box_px((c, cx, cy, w, h), W, H),
                      "contrast": float(abs(inner.mean() - rv.mean())), "ring_std": float(rv.std()),
                      "edge": bool(x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1)})
    img["n_boxes"] = len(boxes)
    return img, boxes


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "–"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="comma list, overrides data.yaml (e.g. uav,airplane,bird)")
    ap.add_argument("--dup-thr", type=int, default=4); ap.add_argument("--min-cell", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    root = R.find_root(a.data)
    names = a.names.split(",") if a.names else R.read_names(root)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(p, s, R.label_path(p)) for s in SPLITS for p in R.list_split(root, s)]
    print(f"{root}: {len(jobs)} images, names {names}", flush=True)
    imgs, boxes = [], []
    with Pool(a.workers) as pool:
        for k, (im, bx) in enumerate(pool.imap(analyse, jobs, chunksize=64)):
            if im:
                imgs.append(im); boxes += bx
            if k % 5000 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    I = pd.DataFrame(imgs); B = pd.DataFrame(boxes, columns=["path", "split", "cls", "px", "contrast", "ring_std", "edge"])
    B["name"] = B["cls"].map(lambda c: names[c] if 0 <= c < len(names) else f"cls{c}")
    B["size"] = B["px"].map(R.size_bin)
    B["bg"] = np.where(B["ring_std"] < 8, "smooth", "textured")
    B["lowc"] = B["contrast"] < 12
    I["light"] = pd.cut(I["lum"], [-1, 60, 190, 256], labels=["dark", "normal", "bright"]).astype(str)
    I["comp"] = pd.cut(I["smooth_frac"], [-1, 0.2, 0.6, 1.01], labels=["cluttered", "mixed", "mostly smooth"]).astype(str)
    I.drop(columns="_hp").to_csv(os.path.join(a.out, "images.csv"), index=False); B.to_csv(os.path.join(a.out, "boxes.csv"), index=False)
    present = [s for s in SPLITS if (I.split == s).any()]
    L = [f"# RGB dataset audit: `{os.path.basename(os.path.normpath(root))}`", "",
         f"Classes {names}. Sizes are sqrt(w·h) in model-input px at 640. Script: `detection/src/data/rgb_audit.py`.", ""]

    L += ["## Counts", "", "| | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    row = lambda lab, f: L.append(f"| {lab} | " + " | ".join(f(s) for s in present) + " |")
    row("images", lambda s: f"{(I.split == s).sum():,}")
    row("empty images", lambda s: f"{((I.split == s) & (I.n_boxes == 0)).sum():,} ({pct(((I.split == s) & (I.n_boxes == 0)).sum(), (I.split == s).sum())})")
    for n in names:
        row(f"{n} boxes", lambda s, n=n: f"{((B.split == s) & (B.name == n)).sum():,}")
    row("sources", lambda s: f"{I[I.split == s].source.nunique():,}")
    row("top-10 sources' share", lambda s: pct(I[I.split == s].source.value_counts().head(10).sum(), (I.split == s).sum()))
    L.append("")

    bins = [R.size_bin(lo) for lo, _ in R.SIZE_BINS]
    for n in names:
        L += [f"## {n} target size", "", "| split | n | " + " | ".join(bins) + " |", "|---|---|" + "---|" * len(bins)]
        for s in present:
            b = B[(B.split == s) & (B.name == n)]
            L.append(f"| {s} | {len(b):,} | " + " | ".join(pct((b["size"] == x).sum(), len(b)) for x in bins) + " |")
        L.append("")

    U = B[B.name == "uav"].copy(); U["small"] = U.px < 16
    cells = [("small <16 px, smooth bg (sky)", lambda d: d.small & (d.bg == "smooth")),
             ("small <16 px, textured bg (cloud/terrain)", lambda d: d.small & (d.bg == "textured")),
             ("small <16 px, low contrast", lambda d: d.small & d.lowc),
             ("small, low contrast, textured", lambda d: d.small & d.lowc & (d.bg == "textured")),
             ("16-64 px, textured bg", lambda d: (d.px >= 16) & (d.px < 64) & (d.bg == "textured")),
             (">=64 px (close range)", lambda d: d.px >= 64),
             ("cut by image edge", lambda d: d.edge)]
    L += ["## uav: background and contrast", "",
          "Ring = 0.75 × box size around the box. smooth = ring gray std < 8; low contrast = |box − ring| < 12 gray levels.", "",
          "| uav boxes | " + " | ".join(present) + " | hole? |", "|---|" + "---|" * len(present) + "---|"]
    holes = []
    for lab, f in cells:
        vals = {s: (int(f(U[U.split == s]).sum()), int((U.split == s).sum())) for s in present}
        tr_n, tr_d = vals.get("train", (0, 0)); flag = []
        if tr_n < a.min_cell:
            flag.append(f"train < {a.min_cell}")
        for s in present[1:]:
            n, d = vals[s]
            if tr_d and d and n / d < (tr_n / tr_d) / 3:
                flag.append(f"{s} under-represented")
        if flag and lab != "cut by image edge":
            holes.append((lab, ", ".join(flag)))
        L.append(f"| {lab} | " + " | ".join(f"{n:,} ({pct(n, d)})" for n, d in vals.values()) + f" | {'; '.join(flag)} |")
    L.append("")

    L += ["## Scenes (images with a uav box)", "", "| | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    has_uav = set(U.path)
    for col, vals in (("light", ["dark", "normal", "bright"]), ("comp", ["mostly smooth", "mixed", "cluttered"])):
        for v in vals:
            row(f"{col}: {v}", lambda s, v=v, col=col: pct(((I.split == s) & I.path.isin(has_uav) & (I[col] == v)).sum(),
                                                           ((I.split == s) & I.path.isin(has_uav)).sum()))
    L.append("")

    L += ["## Leakage", ""]
    src = {s: set(I[I.split == s].source) for s in present}
    hs = {s: I[I.split == s].dhash.astype("uint64").values for s in present}
    fn = {s: np.stack(I[I.split == s]._hp.values) for s in present}
    fl = {s: I[I.split == s].detail.values for s in present}
    for s in present[1:]:
        shared = src[s] & src["train"] if "train" in src else set()
        n_img = I[(I.split == s) & I.source.isin(shared)].shape[0]
        pairs = (R.confirm_pairs(R.hamming_pairs(hs[s], hs["train"], a.dup_thr), fn[s], fn["train"], fl[s], fl["train"])
                 if "train" in hs else np.zeros((0, 2)))
        n_dup = len(set(pairs[:, 0].tolist())) if len(pairs) else 0
        L.append(f"- **{s}**: {len(shared)} sources also in train ({n_img:,} of {(I.split == s).sum():,} {s} images, "
                 f"{pct(n_img, (I.split == s).sum())}); {n_dup:,} {s} images ({pct(n_dup, (I.split == s).sum())}) have a "
                 f"near-duplicate (dHash ≤ {a.dup_thr}) in train.")
    L.append("")
    L += ["## Holes flagged automatically", ""] + ([f"- uav, {h}: {why}" for h, why in holes] or ["- none"])
    for n in names:
        nn = B[(B.split == "train") & (B.name == n)].shape[0]
        if nn < 3000:
            L.append(f"- class **{n}**: only {nn:,} train boxes")
    for s in present:
        d = I[I.split == s]
        if (d.light == "dark").mean() < 0.02:
            L.append(f"- {s}: dusk / low light {pct((d.light == 'dark').sum(), len(d))} of images")
        if (d.n_boxes == 0).mean() < 0.05:
            L.append(f"- {s}: only {pct((d.n_boxes == 0).sum(), len(d))} empty (negative) images")
    open(os.path.join(a.out, "AUDIT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

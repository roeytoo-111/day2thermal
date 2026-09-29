"""
Build a YOLO training dataset: base dataset + labelled new sessions + data-driven augmentations,
with a stats report and contact sheets to check BEFORE training.

  train = base train + each --add session's train split (x --oversample)
          + zoom-out tiles       (--zoomout_frac)        size: drones shrunk to the eval-set scale
          + local polarity flips (--polarity_flip_frac)  bright drone -> dark drone, scene unchanged
          + global inversions    (--invert_frac)         whole image 255-I (unphysical sky; ablation only)
  val   = each --add session's val split (contiguous held-out chunk of that recording)
  test  = base test (the real test sets -- 07-08 video, 07-30 session -- are scored separately)

Why these two augmentations (measured 2026-09-30, corrected labels, drone size in NETWORK pixels at
imgsz 512):
                          median size   < 8 px   dark
  07-08 eval (airborne)       6.2 px      66%     28%
  base train set             40.6 px       3%      6%
Size is the largest train/test mismatch; polarity is the second. Blur is deliberately NOT added: the
base set is already less sharp than the deployment camera (sharpness 421 vs 1164).

Zoom-out tile: k x k base images, each resized to 512/k, labels scaled by 1/k -> every drone k times
smaller with real appearance and real context. k per canvas is drawn so drone sizes fall in
--target_px (log-uniform), and never below 3 px (smaller boxes would be dropped -> unlabelled drones).

Local polarity flip: for each (bright) drone box, pixels become 2*bg - I inside a feathered mask, where
bg = median of a ring around the box. The drone turns dark on the SAME background (unlike 255-I).

    python3 src/data/build_dataset.py --base data/thermal-1-noleak \\
        --add data/new_sessions/yolo_0623 --add data/new_sessions/yolo_0715 --oversample 3 \\
        --zoomout_frac 0.3 --polarity_flip_frac 0.3 --out data/thermal-2-aug [--dry_run]
    # re-report an existing build:
    python3 src/data/build_dataset.py --report_only --out data/thermal-2-aug
"""

import os
import glob
import json
import shutil
import argparse

import cv2
import numpy as np
import yaml

UAV = 1
REF = {"name": "07-08 eval, airborne (2026-09-30)", "median_px": 6.2, "lt8": 0.66, "dark": 0.28}


def parse_args():
    p = argparse.ArgumentParser(description="Base + sessions + size/polarity augmentation, with a pre-train report.")
    p.add_argument("--base", help="Base YOLO dataset (train/valid/test + data.yaml).")
    p.add_argument("--add", action="append", default=[], help="Exported session dir (train/, val/), repeatable.")
    p.add_argument("--out", required=True)
    p.add_argument("--oversample", type=int, default=1, help="Copies of each added-session TRAIN image.")
    p.add_argument("--zoomout_frac", type=float, default=0.0,
                   help="Zoom-out tile canvases to add, as a fraction of the base train image count.")
    p.add_argument("--target_px", default="3,12", help="Drone size range (network px) for zoom-out tiles.")
    p.add_argument("--polarity_flip_frac", type=float, default=0.0,
                   help="Fraction of eligible train images (all drones bright) to add as local-flip copies.")
    p.add_argument("--invert_frac", type=float, default=0.0, help="Whole-image inversion copies (ablation).")
    p.add_argument("--keep_base_valid", action="store_true", help="Also use base valid/ as val (optimistic).")
    p.add_argument("--imgsz", type=int, default=512, help="Training imgsz, for the network-pixel stats.")
    p.add_argument("--dry_run", action="store_true", help="Build in <out>.dryrun, write the report, delete the images.")
    p.add_argument("--report_only", action="store_true", help="Only (re)write <out>/report for an existing build.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------ io helpers
def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def pairs(split_dir):
    for img in sorted(glob.glob(os.path.join(split_dir, "images", "*"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        yield img, os.path.join(split_dir, "labels", stem + ".txt")


def read_labels(lbl):
    if not os.path.exists(lbl):
        return []
    return [[int(r[0])] + [float(v) for v in r[1:5]] for r in (l.split() for l in open(lbl)) if len(r) >= 5]


def write_labels(path, rows):
    with open(path, "w") as f:
        for c, x, y, w, h in rows:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def put(img, lbl, out_split, name):
    link_or_copy(img, os.path.join(out_split, "images", name + os.path.splitext(img)[1]))
    if os.path.exists(lbl):
        link_or_copy(lbl, os.path.join(out_split, "labels", name + ".txt"))
    else:
        open(os.path.join(out_split, "labels", name + ".txt"), "w").close()


def box_px(row, W, H):
    _, x, y, w, h = row
    return x * W - w * W / 2, y * H - h * H / 2, x * W + w * W / 2, y * H + h * H / 2


def ring_bg(g, x0, y0, x1, y1):
    H, W = g.shape
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)
    if x1 <= x0 or y1 <= y0:
        return None
    m = max(4, int(max(x1 - x0, y1 - y0) / 2))
    X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)
    r = g[Y0:Y1, X0:X1]
    mk = np.ones_like(r, bool)
    mk[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    if mk.sum() == 0:
        return None
    return float(np.median(r[mk])), (x0, y0, x1, y1)


def polarity(g, row):
    H, W = g.shape
    rb = ring_bg(g, *box_px(row, W, H))
    if rb is None:
        return None
    bg, (x0, y0, x1, y1) = rb
    b = g[y0:y1, x0:x1]
    return "bright" if b.max() - bg >= bg - b.min() else "dark"


# ------------------------------------------------------------------ augmentations
def local_polarity_flip(img, rows):
    """Each drone -> 2*bg - I inside a feathered box mask (bg from a ring). Returns None if any drone is dark."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    out = g.copy()
    for r in rows:
        if r[0] != UAV:
            continue
        rb = ring_bg(g, *box_px(r, W, H))
        if rb is None:
            return None
        bg, (x0, y0, x1, y1) = rb
        b = g[y0:y1, x0:x1]
        if b.max() - bg < bg - b.min():
            return None                                  # already dark: nothing to teach
        m = np.zeros_like(g)
        m[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), 1.0)
        out = out * (1 - m) + np.clip(2 * bg - g, 0, 255) * m
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def zoomout_canvas(first, pool, rng, target, size=512):
    """k x k tile of base images (first + random others) -> (image, label rows, k)."""
    img0, rows0 = first
    sizes = [np.sqrt(r[3] * size * r[4] * size) for r in rows0 if r[0] == UAV]
    if not sizes:
        return None
    t = float(np.exp(rng.uniform(np.log(target[0]), np.log(target[1]))))
    k = int(np.clip(round(np.median(sizes) / t), 2, 8))
    k = max(2, min(k, int(min(sizes) // 3)))           # keep every drone >= 3 px
    if k < 2:
        return None
    c = size // k
    canvas = np.zeros((c * k, c * k, 3), np.uint8)
    rows = []
    items = [first] + [pool[int(i)] for i in rng.choice(len(pool), k * k - 1)]
    for n, (img, rs) in enumerate(items):
        i, j = divmod(n, k)
        im = cv2.imread(img) if isinstance(img, str) else img
        canvas[i * c:(i + 1) * c, j * c:(j + 1) * c] = cv2.resize(im, (c, c), interpolation=cv2.INTER_AREA)
        for cl, x, y, w, h in rs:
            if cl == UAV and min(w * c, h * c) < 2.0:
                return None                              # a drone would vanish: reject the canvas
            rows.append([cl, (j + x) / k, (i + y) / k, w / k, h / k])
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA), rows, k


# ------------------------------------------------------------------ report
def source_of(name):
    for tag in ("__zoom", "__pflip", "__inv"):
        if tag in name:
            return tag.strip("_")
    return "session" if name.startswith("s0") else "base"


def report(out, imgsz, seed=0):
    rng = np.random.default_rng(seed)
    rdir = os.path.join(out, "report")
    os.makedirs(rdir, exist_ok=True)
    stats, samples = {}, {}
    for split in ("train", "valid", "test"):
        for img, lbl in pairs(os.path.join(out, split)):
            src = f"{split}/{source_of(os.path.basename(img))}"
            s = stats.setdefault(src, {"images": 0, "bg_images": 0, "sizes": [], "pol": []})
            s["images"] += 1
            rows = [r for r in read_labels(lbl) if r[0] == UAV]
            if not rows:
                s["bg_images"] += 1
                continue
            samples.setdefault(src, []).append((img, rows))
    # polarity + size (network px) on a sample per source (reading every image is slow)
    for src, lst in samples.items():
        for idx in rng.permutation(len(lst))[:400]:
            img, rows = lst[idx]
            g = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            H, W = g.shape
            sc = imgsz / max(H, W)
            for r in rows:
                stats[src]["sizes"].append(np.sqrt(r[3] * W * r[4] * H) * sc)
                p = polarity(g, r)
                if p:
                    stats[src]["pol"].append(p)
    lines = [f"Dataset report: {out}   (sizes in network px at imgsz {imgsz}; size/polarity on <=400 images per source)",
             f"REFERENCE {REF['name']}: median {REF['median_px']} px, <8 px {REF['lt8']:.0%}, dark {REF['dark']:.0%}", ""]
    lines.append(f"{'source':<18}{'images':>7}{'bg imgs':>8}{'drones*':>8}{'p10':>6}{'p50':>6}{'p90':>6}{'<8px':>6}{'dark':>6}")
    js = {}
    tot_s, tot_p, tot_w = [], [], []
    for src in sorted(stats):
        s = stats[src]
        sz, pol = np.array(s["sizes"]), np.array(s["pol"])
        if src.startswith("train/") and len(sz):
            n_img = s["images"] - s["bg_images"]                       # images with drones in this source
            n_samp = min(400, len(samples.get(src, [])))
            w = n_img / max(n_samp, 1)                                # each sampled drone stands for w drones
            tot_s += list(sz)
            tot_p += list(pol)
            tot_w += [w] * len(sz)
        pc = np.percentile(sz, [10, 50, 90]) if len(sz) else [np.nan] * 3
        lt8 = float((sz < 8).mean()) if len(sz) else float("nan")
        dk = float((pol == "dark").mean()) if len(pol) else float("nan")
        lines.append(f"{src:<18}{s['images']:>7}{s['bg_images']:>8}{len(sz):>8}{pc[0]:>6.1f}{pc[1]:>6.1f}{pc[2]:>6.1f}{lt8:>6.0%}{dk:>6.0%}")
        js[src] = {"images": s["images"], "bg_images": s["bg_images"], "p10_50_90": [float(v) for v in pc],
                   "lt8": lt8, "dark": dk}
    ts, tp, tw = np.array(tot_s), np.array(tot_p), np.array(tot_w)
    order_ = np.argsort(ts)
    wmed = ts[order_][np.searchsorted(np.cumsum(tw[order_]), tw.sum() / 2)] if len(ts) else float("nan")
    wl8 = float((tw * (ts < 8)).sum() / tw.sum()) if len(ts) else float("nan")
    wdk = float((tw[:len(tp)] * (tp == "dark")).sum() / tw[:len(tp)].sum()) if len(tp) == len(ts) and len(tp) else float("nan")
    lines += ["", "TRAIN, weighted by each source's size: "
              f"median {wmed:.1f} px, <8 px {wl8:.0%}, dark {wdk:.0%}"
              f"   vs reference median {REF['median_px']} / <8 {REF['lt8']:.0%} / dark {REF['dark']:.0%}",
              "(* drones in the sampled images)"]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(rdir, "stats.txt"), "w").write(txt + "\n")
    json.dump(js, open(os.path.join(rdir, "stats.json"), "w"), indent=2)

    # contact sheets: crops around drones per source, and full images for the augmentations
    for src, lst in samples.items():
        tiles = []
        for idx in rng.permutation(len(lst))[:32]:
            img, rows = lst[idx]
            im = cv2.imread(img)
            H, W = im.shape[:2]
            r = rows[int(rng.integers(len(rows)))]
            x0, y0, x1, y1 = box_px(r, W, H)
            cx, cy, hw = (x0 + x1) / 2, (y0 + y1) / 2, max(24, int(max(x1 - x0, y1 - y0)))
            X0, Y0 = int(np.clip(cx - hw, 0, max(W - 2 * hw, 0))), int(np.clip(cy - hw, 0, max(H - 2 * hw, 0)))
            c = im[Y0:Y0 + 2 * hw, X0:X0 + 2 * hw].copy()
            sc = 160 / c.shape[1]
            c = cv2.resize(c, (160, int(c.shape[0] * sc)), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(c, (int((x0 - X0) * sc) - 1, int((y0 - Y0) * sc) - 1), (int((x1 - X0) * sc) + 1, int((y1 - Y0) * sc) + 1), (0, 0, 255), 1)
            c = cv2.resize(c, (160, 160))
            cv2.putText(c, f"{np.sqrt((x1 - x0) * (y1 - y0)) * imgsz / max(H, W):.0f}px", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(c)
        if tiles:
            while len(tiles) % 8:
                tiles.append(np.zeros((160, 160, 3), np.uint8))
            cv2.imwrite(os.path.join(rdir, f"crops_{src.replace('/', '_')}.jpg"),
                        np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
        if any(t in src for t in ("zoom", "pflip", "inv")):
            full = []
            for idx in rng.permutation(len(lst))[:6]:
                img, rows = lst[idx]
                im = cv2.imread(img)
                H, W = im.shape[:2]
                for r in rows:
                    x0, y0, x1, y1 = box_px(r, W, H)
                    cv2.rectangle(im, (int(x0) - 2, int(y0) - 2), (int(x1) + 2, int(y1) + 2), (0, 0, 255), 1)
                full.append(cv2.resize(im, (384, 384)))
            while len(full) % 3:
                full.append(np.zeros((384, 384, 3), np.uint8))
            cv2.imwrite(os.path.join(rdir, f"full_{src.replace('/', '_')}.jpg"),
                        np.vstack([np.hstack(full[i:i + 3]) for i in range(0, len(full), 3)]))
    print(f"report: {rdir}/ (stats.txt, stats.json, crops_*.jpg, full_*.jpg)")


# ------------------------------------------------------------------ build
def main():
    a = parse_args()
    if a.report_only:
        return report(a.out, a.imgsz, a.seed)
    rng = np.random.default_rng(a.seed)
    out = a.out + ".dryrun" if a.dry_run else a.out
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; remove it first.")
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, split, sub))
    counts = {"train_base": 0, "train_added": 0, "zoomout": 0, "pflip": 0, "inverted": 0, "valid": 0, "test": 0}
    tr = os.path.join(out, "train")

    base_train = list(pairs(os.path.join(a.base, "train")))
    for img, lbl in base_train:
        put(img, lbl, tr, os.path.splitext(os.path.basename(img))[0])
        counts["train_base"] += 1
    for img, lbl in pairs(os.path.join(a.base, "test")):
        put(img, lbl, os.path.join(out, "test"), os.path.splitext(os.path.basename(img))[0])
        counts["test"] += 1
    if a.keep_base_valid:
        for img, lbl in pairs(os.path.join(a.base, "valid")):
            put(img, lbl, os.path.join(out, "valid"), os.path.splitext(os.path.basename(img))[0])
            counts["valid"] += 1
    originals = [(img, lbl) for img, lbl in base_train]
    for d in a.add:
        for img, lbl in pairs(os.path.join(d, "train")):
            stem = os.path.splitext(os.path.basename(img))[0]
            originals.append((img, lbl))
            for k in range(a.oversample):
                put(img, lbl, tr, f"{stem}__os{k}" if a.oversample > 1 else stem)
                counts["train_added"] += 1
        for img, lbl in pairs(os.path.join(d, "val")):
            put(img, lbl, os.path.join(out, "valid"), os.path.splitext(os.path.basename(img))[0])
            counts["valid"] += 1

    flipped_base = []
    # polarity: local flip of bright drones (base + session originals)
    if a.polarity_flip_frac > 0:
        cand = [(img, lbl) for img, lbl in originals if any(r[0] == UAV for r in read_labels(lbl))]
        for i in rng.permutation(len(cand))[:int(round(a.polarity_flip_frac * len(cand) * 1.3))]:
            if counts["pflip"] >= int(round(a.polarity_flip_frac * len(cand))):
                break
            img, lbl = cand[i]
            rows = read_labels(lbl)
            res = local_polarity_flip(cv2.imread(img), rows)
            if res is None:
                continue
            stem = os.path.splitext(os.path.basename(img))[0]
            cv2.imwrite(os.path.join(tr, "images", stem + "__pflip.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(lbl, os.path.join(tr, "labels", stem + "__pflip.txt"))
            if res.shape[0] == res.shape[1]:                 # base (square) images feed the zoom-out pool
                flipped_base.append((os.path.join(tr, "images", stem + "__pflip.jpg"), rows))
            counts["pflip"] += 1

    # size: zoom-out tiles from BASE images (512x512, large drones) and their polarity-flipped copies
    # (so small DARK drones exist too -- the eval's dark drones are 10-20 px, flipped base drones ~40 px)
    if a.zoomout_frac > 0:
        target = [float(v) for v in a.target_px.split(",")]
        pool = [(img, read_labels(lbl)) for img, lbl in base_train] + flipped_base
        with_drone = [p for p in pool if any(r[0] == UAV for r in p[1])]
        n_want, tries = int(round(a.zoomout_frac * len(base_train))), 0
        while counts["zoomout"] < n_want and tries < 5 * n_want:
            tries += 1
            res = zoomout_canvas(with_drone[int(rng.integers(len(with_drone)))], pool, rng, target)
            if res is None:
                continue
            im, rows, k = res
            name = f"zoom{counts['zoomout']:05d}_k{k}__zoom"
            cv2.imwrite(os.path.join(tr, "images", name + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_labels(os.path.join(tr, "labels", name + ".txt"), rows)
            counts["zoomout"] += 1

    if a.invert_frac > 0:
        cand = [(img, lbl) for img, lbl in originals if any(r[0] == UAV for r in read_labels(lbl))]
        for i in rng.permutation(len(cand))[:int(round(a.invert_frac * len(cand)))]:
            img, lbl = cand[i]
            stem = os.path.splitext(os.path.basename(img))[0]
            cv2.imwrite(os.path.join(tr, "images", stem + "__inv.jpg"), 255 - cv2.imread(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(lbl, os.path.join(tr, "labels", stem + "__inv.txt"))
            counts["inverted"] += 1

    with open(os.path.join(a.base, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update({"train": "../train/images", "val": "../valid/images", "test": "../test/images"})
    with open(os.path.join(out, "data.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out, "build_config.json"), "w") as f:
        json.dump({**vars(a), "counts": counts}, f, indent=2)
    print(f"built {out}: {counts}")
    report(out, a.imgsz, a.seed)
    if a.dry_run:
        rep = os.path.join(a.out + ".report")
        if os.path.exists(rep):
            shutil.rmtree(rep)
        shutil.move(os.path.join(out, "report"), rep)
        shutil.rmtree(out)
        print(f"dry run: images removed; report kept at {rep}/")


if __name__ == "__main__":
    main()

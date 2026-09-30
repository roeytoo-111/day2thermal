"""
Detector-in-the-loop test of the RGB->thermal translator: paste real thermal drone sprites onto
  arm A  REAL thermal backgrounds       (registered pair crops from the training sessions)
  arm B  TRANSLATED backgrounds         (the diffusion translation of the SAME pairs' RGB crops)
with IDENTICAL pastes (same sprite, size, contrast, position) in both arms. Only the background pixels
differ, so "B trains YOLO as well as A" <=> the translator is a usable background generator (and RGB-only
footage becomes a source of new thermal backgrounds). Pixel metrics can't answer that; the detector can.

Two steps:

  select   collect background crops from pair dirs, drop held-out frames, and find the REAL drones already
           in them (every flight frame has one) with YOLO on the crops. Those boxes are inpainted away
           in BOTH arms at build time (an unlabelled real drone would be a false negative).
  build    base YOLO dataset (hard links) + N synthetic images: sky-only bright pastes (synth_copy_paste
           v2 rules). Placement, visibility and SNR are decided on the (inpainted) REAL crop, then the
           same composite is applied to --bg_dir, so the rng stream and every paste are identical per seed.

Leak rules: never 07-02 (s1 is the eval video, s2 same day/site); 06-23 s2 and 07-15 only before their
scored val chunks (score_models.py 0623_val >= frame 19549, 0715_val >= 9775). 07-30 has no RGB.
Caveat for B: the translator was trained on most of these RGB crops (diff_pairs_all train), so B's
backgrounds are as good as the translator gets -- an optimistic arm. B < A is conclusive; B ~= A needs a
confirmation on RGB-only footage.

    python3 src/data/paste_on_backgrounds.py select --work data/bgpaste \\
        --pairs s0623:../data/vid_pairs_0623:19000 --pairs s0623a:../data/vid_pairs_0623s1 \\
        --pairs s0623c:../data/vid_pairs_0623s3 --pairs s0715:../data/vid_pairs_0715:9500 \\
        --weights ../runs/thermal_uav/rgb_transfer_noleak_p2_sessions/weights/best.pt
    # (GPU) translate the RGB crops: python -m day2thermal.diffusion.infer --controlnet runs/diff_cn_all/best \\
    #        --input detection/data/bgpaste/rgb --out detection/data/bgpaste/translated
    python3 src/data/paste_on_backgrounds.py build --work data/bgpaste --out data/thermal-3-pasteA
    python3 src/data/paste_on_backgrounds.py build --work data/bgpaste --out data/thermal-3-pasteB \\
        --bg_dir data/bgpaste/translated
"""

import os
import re
import csv
import sys
import glob
import json
import argparse

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_copy_paste import (SPLITS, link_or_copy, extract_crops, sample_size, sample_contrast,   # noqa: E402
                              resize_sprite, sky_ok, boxes_overlap, contact_sheet)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--work", required=True)
    s.add_argument("--pairs", action="append", required=True,
                   help="prefix:pair_dir[:max_frame] -- all */thermal/<prefix>_<frame>.png, frames < max_frame")
    s.add_argument("--weights", action="append", required=True, help="YOLO weights for real-drone removal (union)")
    s.add_argument("--conf", type=float, default=0.1, help="low on purpose: inpainting a false fire costs little")
    s.add_argument("--imgsz", type=int, default=512)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--bg_dir", default=None, help="background pixels (default: <work>/real)")
    b.add_argument("--base", default="data/thermal-2-sessions", help="YOLO dataset copied as is")
    b.add_argument("--sprites_from", default="data/thermal-1-noleak", help="dataset whose train boxes give sprites")
    b.add_argument("--n_images", type=int, default=None, help="default: 25%% of the final train set")
    b.add_argument("--synth_frac", type=float, default=0.25)
    b.add_argument("--min_size", type=float, default=3.0)
    b.add_argument("--max_size", type=float, default=24.0)
    b.add_argument("--max_targets", type=int, default=2)
    b.add_argument("--max_roughness", type=float, default=3.0)
    b.add_argument("--speck_thresh", type=float, default=25.0)
    b.add_argument("--max_bg_level", type=float, default=200.0)
    b.add_argument("--min_snr", type=float, default=4.0)
    b.add_argument("--remove_conf", type=float, default=0.25, help="inpaint real detections at >= this conf")
    b.add_argument("--remove_max_size", type=float, default=40.0, help="... and at most this long side (px)")
    b.add_argument("--drop_textured", type=float, default=12.0,
                   help="drop a background if a box to inpaint has a ring high-pass std above this")
    b.add_argument("--uav_class", type=int, default=1)
    b.add_argument("--jpeg_quality", type=int, default=90)
    b.add_argument("--tag", default="bgpaste")
    b.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------ select
def cmd_select(a):
    from ultralytics import YOLO
    for sub in ("real", "rgb"):
        os.makedirs(os.path.join(a.work, sub), exist_ok=True)
    items = []
    for spec in a.pairs:
        parts = spec.split(":")
        prefix, d = parts[0], parts[1]
        max_frame = int(parts[2]) if len(parts) > 2 else None
        for th in sorted(glob.glob(os.path.join(d, "*", "thermal", f"{prefix}_*.png"))):
            name = os.path.basename(th)
            frame = int(re.search(r"_(\d+)\.png$", name).group(1))
            if max_frame is not None and frame >= max_frame:
                continue
            rgb = th.replace(os.sep + "thermal" + os.sep, os.sep + "rgb" + os.sep)
            if not os.path.exists(rgb):
                continue
            for src, sub in ((th, "real"), (rgb, "rgb")):
                if not os.path.exists(os.path.join(a.work, sub, name)):
                    link_or_copy(src, os.path.join(a.work, sub, name))
            items.append((prefix, frame, name))
    paths = [os.path.join(a.work, "real", n) for _, _, n in items]
    found = {n: [] for _, _, n in items}
    for w in a.weights:
        model = YOLO(w)
        for k in range(0, len(paths), 32):                      # small batches: a list source is preloaded
            for p, r in zip(paths[k:k + 32], model.predict(paths[k:k + 32], conf=a.conf, imgsz=a.imgsz,
                                                           device="cpu", verbose=False)):
                for xyxy, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    found[os.path.basename(p)].append([round(v, 1) for v in xyxy] + [round(c, 3)])
    with open(os.path.join(a.work, "backgrounds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "session", "frame", "n_removed", "remove_boxes"])
        for prefix, frame, n in items:
            w.writerow([n, prefix, frame, len(found[n]), json.dumps(found[n])])
    json.dump(vars(a), open(os.path.join(a.work, "select_config.json"), "w"), indent=2)
    per = {}
    for prefix, _, n in items:
        per.setdefault(prefix, [0, 0])
        per[prefix][0] += 1
        per[prefix][1] += bool(found[n])
    print("backgrounds per session (total, with a real drone to inpaint):", per)
    print(f"-> {len(items)} backgrounds in {a.work}/ (real/, rgb/, backgrounds.csv)")


# ------------------------------------------------------------------ build
def remove_real_drones(img_u8, boxes, min_conf, max_size):
    """Inpaint detector boxes (the real drone would otherwise be an unlabelled positive). Only drone-like
    boxes: low-conf / large ones are mostly terrain false fires, and inpainting them leaves smooth smears
    on textured ground -- a cue that exists in no real frame."""
    boxes = [b for b in boxes if b[4] >= min_conf and max(b[2] - b[0], b[3] - b[1]) <= max_size]
    if not boxes:
        return img_u8
    m = np.zeros(img_u8.shape, np.uint8)
    H, W = img_u8.shape
    for x0, y0, x1, y1, _ in boxes:
        pad = int(max(2, 0.25 * max(x1 - x0, y1 - y0)))
        m[max(int(y0) - pad, 0):min(int(y1) + pad + 1, H), max(int(x0) - pad, 0):min(int(x1) + pad + 1, W)] = 255
    return cv2.inpaint(img_u8, m, 5, cv2.INPAINT_TELEA)


def removal_on_texture(img_u8, boxes, min_conf, max_size, margin=12):
    """Max high-pass std in the ring around each box to be inpainted. On textured ground an inpaint leaves a
    smooth smear (and the box may be a real dark drone): such backgrounds are dropped, not repaired."""
    g = img_u8.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    H, W = g.shape
    worst = 0.0
    for x0, y0, x1, y1, c in boxes:
        if c < min_conf or max(x1 - x0, y1 - y0) > max_size:
            continue
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        X0, Y0 = max(x0 - margin, 0), max(y0 - margin, 0)
        w = hp[Y0:min(y1 + margin, H), X0:min(x1 + margin, W)].copy()
        w[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = np.nan
        worst = max(worst, float(np.nanstd(w)))
    return worst


def paste_pair(ref, bg, m, contrast, x, y, rng, min_snr):
    """Same composite into ref (decides) and bg (if a different image). Draws the same rng numbers whatever
    the pixels, so both arms stay in lock-step. Returns the tight box or None (nothing written)."""
    H, W = ref.shape
    h, w = m.shape
    pad = 3
    X0, Y0, X1, Y1 = x - pad, y - pad, x + w + pad, y + h + pad
    if X0 < 0 or Y0 < 0 or X1 > W or Y1 > H:
        return None
    alpha = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    alpha[pad:pad + h, pad:pad + w] = m
    alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.4, 0.9))          # optics PSF
    z = rng.standard_normal(alpha.shape).astype(np.float32)

    def comp(img):
        region = img[Y0:Y1, X0:X1]
        hf = region - cv2.GaussianBlur(region, (0, 0), 2)
        sd = max(float(hf.std()), 0.5)                                        # the image's own noise level
        return np.clip(region + alpha * contrast + (alpha > 0.05) * z * sd * 0.5, 0, 255), region, sd

    out, region, sd = comp(ref)
    rise = float((out - region).max())
    ys, xs = np.nonzero(alpha > 0.2)
    if len(xs) == 0 or rise < max(20.0, min_snr * sd):
        return None
    if bg is not ref:
        bg[Y0:Y1, X0:X1] = comp(bg)[0]
    ref[Y0:Y1, X0:X1] = out
    return (max(X0 + xs.min() - 1, 0), max(Y0 + ys.min() - 1, 0), min(X0 + xs.max() + 2, W), min(Y0 + ys.max() + 2, H))


def cmd_build(a):
    rng = np.random.default_rng(a.seed)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists; remove it first.")
    for need in (os.path.join(a.base, "data.yaml"), os.path.join(a.sprites_from, "train", "labels")):
        if not os.path.exists(need):
            raise SystemExit(f"missing {need} (--base / --sprites_from)")
    bg_dir = a.bg_dir or os.path.join(a.work, "real")
    rows = list(csv.DictReader(open(os.path.join(a.work, "backgrounds.csv"))))
    missing = [r["name"] for r in rows if not os.path.exists(os.path.join(bg_dir, r["name"]))]
    if missing:
        raise SystemExit(f"{len(missing)} backgrounds missing in {bg_dir} (e.g. {missing[0]}); every arm must "
                         f"use the same list.")
    n_all = len(rows)
    rows = [r for r in rows if removal_on_texture(cv2.imread(os.path.join(a.work, "real", r["name"]), cv2.IMREAD_GRAYSCALE),
                                                  json.loads(r["remove_boxes"]), a.remove_conf,
                                                  a.remove_max_size) <= a.drop_textured]
    print(f"backgrounds: {len(rows)} of {n_all} (dropped: a real detection on textured ground)")
    sprites = extract_crops(a.sprites_from, a.uav_class, 24.0)
    print(f"real drone sprites: {len(sprites)}")
    if not sprites:
        raise SystemExit(f"no usable drone sprites in {a.sprites_from}/train")

    for split in SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, split, sub))
            for f in glob.glob(os.path.join(a.base, split, sub, "*")):
                link_or_copy(f, os.path.join(a.out, split, sub, os.path.basename(f)))
    cfg = yaml.safe_load(open(os.path.join(a.base, "data.yaml")))
    yaml.safe_dump(cfg, open(os.path.join(a.out, "data.yaml"), "w"), sort_keys=False)
    n_base = len(os.listdir(os.path.join(a.base, "train", "images")))
    n = a.n_images or int(round(a.synth_frac / (1 - a.synth_frac) * n_base))

    def load(d, r):
        g = cv2.imread(os.path.join(d, r["name"]), cv2.IMREAD_GRAYSCALE)
        return remove_real_drones(g, json.loads(r["remove_boxes"]), a.remove_conf, a.remove_max_size).astype(np.float32)

    order = rng.permutation(np.resize(np.arange(len(rows)), n))
    mf = open(os.path.join(a.out, "synth_manifest.csv"), "w", newline="")
    mw = csv.writer(mf)
    mw.writerow(["out_image", "background", "sprite_id", "size", "contrast", "bg_level", "x0", "y0", "x1", "y1"])
    qa, n_written, n_pastes, n_skipped = [], 0, 0, 0
    for i, bi in enumerate(order):
        r = rows[bi]
        ref = load(os.path.join(a.work, "real"), r)
        bg = ref if a.bg_dir is None else load(bg_dir, r)
        if bg.shape != ref.shape:
            bg = cv2.resize(bg, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        H, W = ref.shape
        hp = ref - cv2.GaussianBlur(ref, (0, 0), 3)
        occupied, boxes, meta = [], [], []
        for _ in range(int(rng.integers(1, a.max_targets + 1))):
            size = sample_size(rng, a.min_size, a.max_size)
            sid = int(rng.integers(len(sprites)))
            m = resize_sprite(sprites[sid], size, rng)
            contrast = sample_contrast(rng, size)
            for _attempt in range(60):
                x = int(rng.integers(4, max(W - m.shape[1] - 4, 5)))
                y = int(rng.integers(4, max(H - m.shape[0] - 4, 5)))
                cand = (x, y, x + m.shape[1], y + m.shape[0])
                if any(boxes_overlap(cand, o, 6) for o in occupied):
                    continue
                if not sky_ok(hp, *cand, a.max_roughness, a.speck_thresh):
                    continue
                lvl = float(np.median(ref[max(y - 8, 0):y + m.shape[0] + 8, max(x - 8, 0):x + m.shape[1] + 8]))
                if lvl > a.max_bg_level:
                    continue
                box = paste_pair(ref, bg, m, contrast, x, y, rng, a.min_snr)
                if box is None:
                    continue
                occupied.append(box)
                boxes.append(box)
                meta.append([sid, round(size, 2), round(contrast, 1), round(lvl, 1), *box])
                break
            else:
                n_skipped += 1
        if not boxes:
            continue
        stem = f"{a.tag}_{i:06d}__{os.path.splitext(r['name'])[0]}"
        u8 = bg.round().astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, "train", "images", stem + ".jpg"), cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        with open(os.path.join(a.out, "train", "labels", stem + ".txt"), "w") as f:
            for b_ in boxes:
                f.write(f"{a.uav_class} {(b_[0] + b_[2]) / 2 / W:.6f} {(b_[1] + b_[3]) / 2 / H:.6f} "
                        f"{(b_[2] - b_[0]) / W:.6f} {(b_[3] - b_[1]) / H:.6f}\n")
        for mm in meta:
            mw.writerow([stem + ".jpg", r["name"], *mm])
        n_written += 1
        n_pastes += len(boxes)
        if len(qa) < 24:
            qa.append((u8, boxes))
    mf.close()
    contact_sheet(qa, os.path.join(a.out, "synth_qa_contact_sheet.png"))
    # whole-frame QA: 8 synthetic images with their boxes
    imgs = sorted(glob.glob(os.path.join(a.out, "train", "images", a.tag + "_*.jpg")))[:8]
    tiles = []
    for p in imgs:
        v = cv2.imread(p)
        h_, w_ = v.shape[:2]
        for l in open(p.replace("images", "labels").replace(".jpg", ".txt")):
            _, cx, cy, bw, bh = (float(t) for t in l.split())
            cv2.rectangle(v, (int((cx - bw / 2) * w_) - 3, int((cy - bh / 2) * h_) - 3),
                          (int((cx + bw / 2) * w_) + 3, int((cy + bh / 2) * h_) + 3), (0, 200, 255), 1)
        tiles.append(cv2.resize(v, (466, 262)))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.out, "synth_qa_frames.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
    json.dump({**vars(a), "n_base_train": n_base, "n_target": n, "n_written": n_written, "n_pasted": n_pastes,
               "n_skipped": n_skipped, "n_backgrounds": len(rows), "n_backgrounds_listed": n_all},
              open(os.path.join(a.out, "synth_config.json"), "w"), indent=2)
    print(f"{a.out}: base {n_base} + {n_written} synthetic images ({n_pastes} pasted drones, {n_skipped} "
          f"pastes found no smooth spot) from {len(rows)} backgrounds [{bg_dir}]")


if __name__ == "__main__":
    a = parse_args()
    cmd_select(a) if a.cmd == "select" else cmd_build(a)

r"""
Thermal training pastes for the case the data never shows: drones (and aircraft) seen ABOVE textured terrain, in every
shade: black (colder than the ground), grey (barely different, either sign) and white (hotter than the ground).

Why: every uav box in thermal-7-N2-u0 train is BRIGHT polarity (0 of 2,600+ real boxes darker than their surroundings,
measured 2026-10-18), and the earlier pastes (synth_copy_paste / paste_on_backgrounds) are bright sprites on smooth sky
only, by design. The 0715_val test drones are dark airframes on bright warm terrain seen from above (15 of 24 sampled boxes
dark-polarity), every thermal model scores 13-21 % mean recall there, and that look is the end goal "aircraft below the
camera".

How:
  shapes      real drone sprites cut from bright boxes of the base train split (synth_copy_paste.extract_crops rules,
              pasted images excluded): a soft mask in [0, 1] that carries the real silhouette and blur. The SHADE is set at
              paste time, so a bright sprite's shape becomes a dark or grey drone.
  shade       dark strong 45 %: contrast -40..-130 grey levels; grey 35 %: +-15..40 (either sign); white 20 %: +40..+150
  where       random spots of real-session train frames (no pasted / public images): the window around the box must be
              TEXTURED (grey-level std >= --min-texture: terrain, not sky), warm enough for a dark drone to show
              (median >= --min-level-dark) or cool enough for a white one (<= --max-level-light), and at least 2 sprite sizes
              from every labelled box. The paste must stay visible: |median change under the sprite core| >=
              max(--min-delta, 1.5 x local noise), else the spot is rejected.
  size        log-uniform --size-min..--size-max px (0715_val boxes are ~15-40 px).
  output      base dataset hard-linked + N new train images pterr_<source>_<k>.jpg with the source's labels plus the new
              uav box (class 0); valid/test untouched. manifest.csv logs shade, contrast, size, background level/texture.
Leak rules: only the base TRAIN split is read (it is already leak-free: no 07-02, 07-30, held-out Boson sessions, or 07-15
frames from the val chunk).

    python3 src/data/paste_terrain.py --base data/thermal-7-N2-u0 --out data/thermal-9-N2D-u0 --n 800
"""
import os
import csv
import sys
import glob
import shutil
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_copy_paste import read_labels, yolo_to_xyxy, resize_sprite, sample_size, boxes_overlap   # noqa: E402

PASTED = ("bgpaste", "bgboson", "pterr", "synth", "paste")
PUBLIC = ("frame", "IR_BIRD", "ir_", "ir.", "img", "0_0")


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def real_sprites(train_dir, min_crop=8):
    """Bright real drone sprites (soft masks) from non-pasted images; same rules as synth_copy_paste.extract_crops."""
    out = []
    for lbl in sorted(glob.glob(os.path.join(train_dir, "labels", "*.txt"))):
        stem = os.path.splitext(os.path.basename(lbl))[0]
        if stem.startswith(PASTED):
            continue
        rows = [l for l in read_labels(lbl) if int(l[0]) == 0]
        if not rows:
            continue
        imgs = glob.glob(os.path.join(train_dir, "images", stem + ".*"))
        if not imgs:
            continue
        g = cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE).astype(np.float32); H, W = g.shape
        for l in rows:
            x0, y0, x1, y1 = yolo_to_xyxy(l, W, H)
            if np.sqrt(max(x1 - x0, 0) * max(y1 - y0, 0)) < min_crop:
                continue
            x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)
            pad = max(4, int(0.25 * max(x1 - x0, y1 - y0)))
            X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
            patch = g[Y0:Y1, X0:X1]; ring = np.ones_like(patch, bool); ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
            if ring.sum() < 10:
                continue
            bg = np.median(patch[ring]); box = patch[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]; peak = box.max() - bg
            if peak < 25:
                continue
            m = np.clip((box - bg) / peak, 0, 1)
            edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
            if (edge > 0.5).mean() > 0.3:
                continue
            m[m < 0.15] = 0
            if m.sum() < 4:
                continue
            ys, xs = np.nonzero(m)
            out.append((m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32), stem.split("_")[0]))
    return out


def sample_shade(rng):
    r = rng.random()
    if r < 0.45:
        return "dark", -float(rng.uniform(40, 130))
    if r < 0.80:
        return "grey", float(rng.choice([-1, 1]) * rng.uniform(15, 40))
    return "white", float(rng.uniform(40, 150))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size-min", type=float, default=8); ap.add_argument("--size-max", type=float, default=45)
    ap.add_argument("--min-texture", type=float, default=10.0); ap.add_argument("--min-level-dark", type=float, default=70.0)
    ap.add_argument("--max-level-light", type=float, default=200.0); ap.add_argument("--min-delta", type=float, default=15.0)
    ap.add_argument("--tries", type=int, default=60)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    rng = np.random.default_rng(a.seed)
    tr = os.path.join(a.base, "train")
    sprites = real_sprites(tr)
    print(f"{len(sprites)} real bright sprites (shapes) from {len({s for _, s in sprites})} sources", flush=True)
    for s in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, sub))
            for f in os.listdir(os.path.join(a.base, s, sub)):
                link(os.path.join(a.base, s, sub, f), os.path.join(a.out, s, sub, f))
    shutil.copy2(os.path.join(a.base, "data.yaml"), os.path.join(a.out, "data.yaml"))
    srcs = [p for p in sorted(glob.glob(os.path.join(tr, "images", "*"))) if not os.path.basename(p).startswith(PASTED + PUBLIC)]
    rng.shuffle(srcs)
    print(f"{len(srcs)} real-session train frames as backgrounds", flush=True)
    man = open(os.path.join(a.out, "pterr_manifest.csv"), "w", newline=""); mw = csv.writer(man)
    mw.writerow(["image", "source", "shade", "contrast", "size", "bg_level", "bg_texture", "delta", "x0", "y0", "x1", "y1"])
    made, rej = 0, {"texture": 0, "level": 0, "overlap": 0, "invisible": 0}
    for src in srcs:
        if made >= a.n:
            break
        stem = os.path.splitext(os.path.basename(src))[0]
        img = cv2.imread(src); H, W = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab = read_labels(os.path.join(tr, "labels", stem + ".txt"))
        boxes = [yolo_to_xyxy(l, W, H) for l in lab]
        for _ in range(a.tries):
            m0, _src = sprites[int(rng.integers(len(sprites)))]
            size = sample_size(rng, a.size_min, a.size_max)
            m = resize_sprite(m0, size, rng); h, w = m.shape
            if h < 3 or w < 3 or h > H - 20 or w > W - 20:
                continue
            x, y = int(rng.integers(8, W - w - 8)), int(rng.integers(8, H - h - 8))
            nb = (x, y, x + w, y + h)
            if any(boxes_overlap(nb, b, 2 * max(w, h)) for b in boxes):
                rej["overlap"] += 1; continue
            win = g[max(y - 16, 0):y + h + 16, max(x - 16, 0):x + w + 16]
            tex, lvl = float(win.std()), float(np.median(win))
            if tex < a.min_texture:
                rej["texture"] += 1; continue
            shade, c = sample_shade(rng)
            if (c < 0 and lvl < a.min_level_dark) or (c > 0 and lvl > a.max_level_light):
                rej["level"] += 1; continue
            region = img[y:y + h, x:x + w].astype(np.float32)
            sd = max(float((g[y:y + h, x:x + w] - cv2.GaussianBlur(g[y:y + h, x:x + w], (0, 0), 2)).std()), 0.5)
            alpha = m[..., None]
            noise = rng.normal(0, 1, (h, w, 1)).astype(np.float32)
            out = np.clip(region + alpha * c + (alpha > 0.05) * noise * sd * 0.5, 0, 255)
            core = m > 0.5
            delta = float(np.median((out - region)[..., 0][core])) if core.any() else 0.0
            if abs(delta) < max(a.min_delta, 1.5 * sd):
                rej["invisible"] += 1; continue
            new = img.copy(); new[y:y + h, x:x + w] = out.astype(np.uint8)
            name = f"pterr_{stem}_{made}"
            cv2.imwrite(os.path.join(a.out, "train", "images", name + ".jpg"), new, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(os.path.join(a.out, "train", "labels", name + ".txt"), "w").write(
                "".join(" ".join(l) + "\n" for l in lab) + f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
            mw.writerow([name, stem, shade, round(c, 1), round(size, 1), round(lvl, 1), round(tex, 1), round(delta, 1), x, y, x + w, y + h])
            made += 1
            break
    man.close()
    print(f"{made} pasted images -> {a.out}; rejected spots: {rej}")


if __name__ == "__main__":
    main()

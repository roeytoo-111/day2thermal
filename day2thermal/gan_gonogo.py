"""Go/no-go check for a paired RGB->thermal translator, against criteria fixed
BEFORE the run (notebook/2026-09-28.md section 5).

On the held-out val pairs it compares three images per pair:
    real thermal | translator output | linear colour baseline
The baseline is thermal ~ w . [R, G, B] + b, least-squares fitted on the TRAIN
pairs: the best a pure per-pixel brightness/colour map can do. A translator
that cannot clearly beat it has learned nothing thermal.

Criteria (all must hold for GO):
  C1 not a greyscale filter:  r(gen, luminance) < 0.75  and  |r(gen, lum) - r(real, lum)| <= 0.10
  C2 beats the baseline:      L1(gen) <= 0.85 * L1(baseline)  and  SSIM(gen) > SSIM(baseline)
  C3 qualitative (by eye):    cold dark sky with gradient, bright clouds/ground, drones rendered
                              BRIGHT, on >= 3 of 4 strips -- the script writes the strips, you judge.

python -m day2thermal.gan_gonogo --ckpt runs/vid_p2p/checkpoints/latest.pt \
    --pairs data/vid_pairs --out runs/vid_p2p/gonogo
# generalisation to another recording day (the stronger test):
python -m day2thermal.gan_gonogo --ckpt runs/vid_p2p/checkpoints/latest.pt \
    --pairs data/vid_pairs --val-pairs data/vid_pairs_0715 --val-split train --out runs/vid_p2p/gonogo_0715
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from .eval import ssim_pair
from .infer import load_model, translate
from .torch_utils import pad_to_multiple, unpad
from .utils import ensure_dir, list_images, read_rgb, rgb_to_norm


def corr(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def read_pair(root, split, name):
    rgb = read_rgb(os.path.join(root, split, "rgb", name))
    th = cv2.imread(os.path.join(root, split, "thermal", name), cv2.IMREAD_GRAYSCALE)
    return rgb, th.astype(np.float64) / 255.0


def fit_linear_baseline(root, n_px=20000, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for name in list_images(os.path.join(root, "train", "rgb")):
        rgb, th = read_pair(root, "train", name)
        f = rgb.reshape(-1, 3).astype(np.float64) / 255.0
        idx = rng.choice(len(f), min(n_px, len(f)), replace=False)
        X.append(f[idx])
        y.append(th.ravel()[idx])
    X, y = np.concatenate(X), np.concatenate(y)
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef                                           # BGR weights + bias


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", required=True, help="dir with train/ and val/ {rgb,thermal}/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-pairs", default=None,
                    help="evaluate on <val-pairs>/{val-split}/ instead of <pairs>/val/ -- e.g. another session's "
                         "pairs, to test generalisation beyond the training day (baseline still fit on --pairs)")
    ap.add_argument("--val-split", default="val", help="split dir inside --val-pairs (train/val); default val")
    ap.add_argument("--n-strips", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = ensure_dir(a.out)

    coef = fit_linear_baseline(a.pairs)
    print(f"linear colour baseline (fit on train): thermal = {coef[0]:+.3f}B {coef[1]:+.3f}G "
          f"{coef[2]:+.3f}R {coef[3]:+.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    nets, targs, tn, num_downs = load_model(a.ckpt, device)
    vroot, vsplit = (a.val_pairs, a.val_split) if a.val_pairs else (a.pairs, "val")
    names = list_images(os.path.join(vroot, vsplit, "rgb"))
    rows, strips = [], []
    strip_idx = set(np.linspace(0, len(names) - 1, min(a.n_strips, len(names))).astype(int))
    for i, name in enumerate(names):
        rgb, real = read_pair(vroot, vsplit, name)
        lum = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        A = torch.from_numpy(rgb_to_norm(rgb)).unsqueeze(0).to(device)
        A, hw = pad_to_multiple(A, 2 ** num_downs)
        with torch.no_grad():
            gen = unpad(translate(nets, targs, A, 0.0), hw)[0, 0].cpu().numpy().astype(np.float64)
        gen = (gen + 1) / 2
        base = np.clip(rgb.reshape(-1, 3).astype(np.float64) / 255.0 @ coef[:3] + coef[3], 0, 1).reshape(real.shape)
        rows.append({"name": name,
                     "l1_gen": float(np.abs(gen - real).mean()), "l1_base": float(np.abs(base - real).mean()),
                     "ssim_gen": ssim_pair(gen, real), "ssim_base": ssim_pair(base, real),
                     "r_gen_lum": corr(gen, lum), "r_real_lum": corr(real, lum), "r_base_lum": corr(base, lum)})
        if i in strip_idx:
            to8 = lambda z: cv2.cvtColor((np.clip(z, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            tiles = [rgb, to8(real), to8(gen), to8(base)]
            for t, lab in zip(tiles, ["RGB", "real thermal", "translator", "linear baseline"]):
                cv2.putText(t, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            strips.append(np.hstack(tiles))

    m = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "name"}
    c1 = m["r_gen_lum"] < 0.75 and abs(m["r_gen_lum"] - m["r_real_lum"]) <= 0.10
    c2 = m["l1_gen"] <= 0.85 * m["l1_base"] and m["ssim_gen"] > m["ssim_base"]
    print(f"\nval pairs: {len(rows)}")
    print(f"                      translator   linear baseline   real")
    print(f"L1 (0-1)              {m['l1_gen']:.4f}       {m['l1_base']:.4f}")
    print(f"SSIM                  {m['ssim_gen']:.4f}       {m['ssim_base']:.4f}")
    print(f"r(., RGB luminance)   {m['r_gen_lum']:+.3f}       {m['r_base_lum']:+.3f}            {m['r_real_lum']:+.3f}")
    print(f"\nC1 not a greyscale filter : {'PASS' if c1 else 'FAIL'}")
    print(f"C2 beats linear baseline   : {'PASS' if c2 else 'FAIL'}  "
          f"(L1 ratio {m['l1_gen'] / m['l1_base']:.2f}, need <= 0.85)")
    print(f"C3 qualitative             : judge {os.path.join(out, 'strips.png')} "
          f"(dark cold sky, bright clouds/ground, BRIGHT drones)")
    verdict = "NO-GO" if not (c1 and c2) else "GO pending C3"
    print(f"\nVERDICT: {verdict}")
    cv2.imwrite(os.path.join(out, "strips.png"), np.vstack(strips))
    with open(os.path.join(out, "gonogo.json"), "w") as f:
        json.dump({"ckpt": a.ckpt, "pairs": a.pairs, "val_pairs": os.path.join(vroot, vsplit), "n_val": len(rows), "baseline_coef_bgr_bias": coef.tolist(),
                   "means": m, "C1": c1, "C2": c2, "verdict": verdict, "per_pair": rows}, f, indent=2)


if __name__ == "__main__":
    main()

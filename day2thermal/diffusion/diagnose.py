"""Why does a translator fail on a split? Per-pair diagnostics, grouped by scene content.

For every pair (prediction vs real thermal) and, for reference, the linear colour baseline fit on train:
  l1, ssim            raw (as gan_gonogo)
  l1_aff, ssim_aff    after fitting the prediction to the real image with a per-image a*x+b (least
                      squares). Large gain from raw -> affine = the structure is right but the global
                      level/contrast is wrong (AGC: the 8-bit thermal is re-stretched per frame and per day).
  edge_ncc            correlation of gradient magnitudes (does the output put edges where the real image
                      has them? low = invented structure / "painting")
  hf_ratio            high-frequency energy of prediction / real (Laplacian variance). << 1 = too smooth,
                      >> 1 = too much hallucinated texture
  sky_frac            fraction of the RGB that is sky (blue/bright, low texture), for grouping

    python -m day2thermal.diffusion.diagnose --pairs data/diff_pairs --split test \\
        --pred-dir runs/diff_cn/preds_test --out runs/diff_cn/diag_test
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from ..eval import ssim_pair
from ..gan_gonogo import fit_linear_baseline, read_pair
from ..utils import ensure_dir, list_images


def affine_fit(p, t):
    A = np.stack([p.ravel(), np.ones(p.size)], 1)
    coef, *_ = np.linalg.lstsq(A, t.ravel(), rcond=None)
    return np.clip(p * coef[0] + coef[1], 0, 1)


def grad(x):
    return cv2.magnitude(cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def sky_fraction(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float64)
    tex = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3))
    blue = (hsv[..., 0] > 90) & (hsv[..., 0] < 135) & (hsv[..., 1] > 40)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 170)                    # clouds
    return float(((blue | white) & (tex < 6)).mean())


def enhance(p, how):
    """Fixed, camera-like detail enhancement of a prediction in [0,1] (never fitted to the target).
    The real thermal is strongly detail-enhanced (DDE/AGC in the camera); the translator's output is smooth."""
    if how == "none":
        return p
    u8 = (np.clip(p, 0, 1) * 255).astype(np.uint8)
    if how == "clahe":
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8).astype(np.float64) / 255.0
    if how == "unsharp":
        blur = cv2.GaussianBlur(p, (0, 0), 1.5)
        return np.clip(p + 1.5 * (p - blur), 0, 1)
    if how == "clahe+unsharp":
        q = enhance(p, "clahe")
        return np.clip(q + 1.5 * (q - cv2.GaussianBlur(q, (0, 0), 1.5)), 0, 1)
    raise ValueError(how)


def metrics(p, t):
    pa = affine_fit(p, t)
    return {"l1": float(np.abs(p - t).mean()), "ssim": ssim_pair(p, t),
            "l1_aff": float(np.abs(pa - t).mean()), "ssim_aff": ssim_pair(pa, t),
            "edge_ncc": ncc(grad(p), grad(t)),
            "hf_ratio": float(cv2.Laplacian(p, cv2.CV_64F).var() / (cv2.Laplacian(t, cv2.CV_64F).var() + 1e-12))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="dir with train/ (for the baseline) and the evaluated split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enhance", default="none", choices=["none", "clahe", "unsharp", "clahe+unsharp"],
                    help="apply a fixed detail enhancement to the predictions before scoring (texture test)")
    a = ap.parse_args()
    out = ensure_dir(a.out)
    coef = fit_linear_baseline(a.pairs)
    rows = []
    for name in list_images(os.path.join(a.pairs, a.split, "rgb")):
        rgb, real = read_pair(a.pairs, a.split, name)
        pred = cv2.imread(os.path.join(a.pred_dir, os.path.splitext(name)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        pred = enhance(cv2.resize(pred, (real.shape[1], real.shape[0])).astype(np.float64) / 255.0, a.enhance)
        base = np.clip(rgb.reshape(-1, 3).astype(np.float64) / 255.0 @ coef[:3] + coef[3], 0, 1).reshape(real.shape)
        r = {"name": name, "sky_frac": sky_fraction(rgb), "real_mean": float(real.mean()),
             "pred_mean": float(pred.mean())}
        r.update({f"model_{k}": v for k, v in metrics(pred, real).items()})
        r.update({f"base_{k}": v for k, v in metrics(base, real).items()})
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "per_pair.csv"), index=False)
    d["content"] = pd.cut(d.sky_frac, [-0.01, 0.2, 0.5, 1.01], labels=["terrain (sky<20%)", "mixed", "sky (>50%)"])
    cols = ["l1", "l1_aff", "ssim", "ssim_aff", "edge_ncc", "hf_ratio"]
    summ = pd.DataFrame({f"{w}_{c}": d[f"{w}_{c}"] for w in ("model", "base") for c in cols})
    lines = [f"{a.split}: {len(d)} pairs, enhance={a.enhance}  (model vs linear-colour baseline; aff = after per-image a*x+b fit)", ""]
    lines.append(f"{'':<22}" + "".join(f"{c:>10}" for c in cols))
    for w in ("model", "base"):
        lines.append(f"{'ALL ' + w:<22}" + "".join(f"{summ[f'{w}_{c}'].median():>10.3f}" for c in cols))
    lines.append("")
    for grp, g in d.groupby("content", observed=True):
        for w in ("model", "base"):
            lines.append(f"{str(grp).split(' ')[0] + ' ' + w + f' n={len(g)}':<22}" + "".join(f"{g[f'{w}_{c}'].median():>10.3f}" for c in cols))
    lines += ["", f"level: real mean {d.real_mean.median():.3f}, prediction mean {d.pred_mean.median():.3f} "
              f"(median |diff| {np.median(np.abs(d.real_mean - d.pred_mean)):.3f})",
              "medians shown; hf_ratio: 1 = same high-frequency energy as real"]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    # worst / best pairs by model l1 for a visual look
    for tag, sel in (("worst", d.nlargest(6, "model_l1")), ("best", d.nsmallest(6, "model_l1"))):
        strips = []
        for n in sel.name:
            rgb, real = read_pair(a.pairs, a.split, n)
            p = cv2.imread(os.path.join(a.pred_dir, os.path.splitext(n)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
            p = cv2.resize(p, (real.shape[1], real.shape[0]))
            strips.append(np.hstack([rgb, cv2.cvtColor((real * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                     cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)]))
        cv2.imwrite(os.path.join(out, f"{tag}.jpg"),
                    np.vstack([cv2.resize(s, (strips[0].shape[1], strips[0].shape[0])) for s in strips]))
    json.dump({"n": len(d)}, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()

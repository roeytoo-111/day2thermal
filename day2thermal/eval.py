"""Evaluate synthetic thermal images against real ones (matched filenames).

Reports L1, RMSE, PSNR and SSIM on the [0, 1] normalized range; with
``--thermal-mode abs16`` L1 and RMSE are additionally reported in degrees C
(compare against ThermalGAN's observation that ~1 degC accuracy is needed to
keep fine contrasts usable, while generic GANs sit around ~5 degC).

python -m day2thermal.eval --pred runs/p2p_preds --gt data/aligned/val/thermal
"""
import argparse
import json
import os

import cv2
import numpy as np

from .utils import ThermalNorm, list_images, read_thermal


def ssim_pair(a, b):
    """Single-channel SSIM, Gaussian 11x11 window, inputs in [0, 1]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--thermal-mode", choices=["rel8", "abs16"], default="rel8")
    ap.add_argument("--raw-scale", type=float, default=0.04)
    ap.add_argument("--raw-offset", type=float, default=-273.15)
    ap.add_argument("--tmin", type=float, default=-20.0)
    ap.add_argument("--tmax", type=float, default=80.0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    tn = ThermalNorm(args.thermal_mode, args.raw_scale, args.raw_offset,
                     args.tmin, args.tmax)

    def stems(d):
        return {os.path.splitext(f)[0].replace("_16bit", ""): f
                for f in list_images(d)}

    pred_map, gt_map = stems(args.pred), stems(args.gt)
    common = sorted(set(pred_map) & set(gt_map))
    if not common:
        raise SystemExit("no matching filenames between --pred and --gt")

    l1s, rmses, psnrs, ssims = [], [], [], []
    for stem in common:
        p = tn.to_norm(read_thermal(os.path.join(args.pred, pred_map[stem])))
        g = tn.to_norm(read_thermal(os.path.join(args.gt, gt_map[stem])))
        if p.shape != g.shape:
            p = cv2.resize(p, (g.shape[1], g.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        p01, g01 = (p + 1) / 2, (g + 1) / 2
        diff = p01 - g01
        l1s.append(float(np.abs(diff).mean()))
        mse = float((diff ** 2).mean())
        rmses.append(mse ** 0.5)
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        ssims.append(ssim_pair(p01, g01))

    res = {"n": len(common),
           "L1": float(np.mean(l1s)),
           "RMSE": float(np.mean(rmses)),
           "PSNR": float(np.mean(psnrs)),
           "SSIM": float(np.mean(ssims))}
    if tn.mode == "abs16":
        res["L1_degC"] = res["L1"] * tn.celsius_span()
        res["RMSE_degC"] = res["RMSE"] * tn.celsius_span()

    for k, v in res.items():
        print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

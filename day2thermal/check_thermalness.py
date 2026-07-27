"""Is the model actually producing THERMAL, or just a stylised greyscale?

An unpaired translator has a well-known shortcut: emit something close to the
input's luminance. It satisfies cycle-consistency perfectly (the mapping stays
trivially invertible) and partially fools the discriminator, but it is not a
thermal image -- real LWIR is nearly uncorrelated with visible brightness,
because temperature is not albedo.

This measures that directly:

    r_gen  = corr(G(RGB), luminance(RGB))    over a sample of frames
    r_real = corr(real thermal, luminance(RGB))   <- the target ballpark

r_gen close to r_real means the model learned a genuine modality change.
r_gen near 1 means it learned a greyscale filter. Note r_real is computed on
UNREGISTERED pairs here, so it is a rough reference level, not an exact target.

    python -m day2thermal.check_thermalness --ckpt runs/cyc/checkpoints/latest.pt \
        --data-root data/unpaired --n 24
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch

from .infer import load_model, translate
from .torch_utils import pad_to_multiple, unpad
from .utils import list_images, read_rgb, rgb_to_norm


def corr(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data/unpaired")
    ap.add_argument("--n", type=int, default=24, help="frames to sample")
    ap.add_argument("--max-size", type=int, default=640,
                    help="must match the scale the model was trained at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    nets, targs, tn, num_downs = load_model(args.ckpt, device)
    rgb_dir = os.path.join(args.data_root, "rgb")
    th_dir = os.path.join(args.data_root, "thermal")
    names = list_images(rgb_dir)
    random.Random(args.seed).shuffle(names)
    names = names[:args.n]

    r_gen, r_real = [], []
    for name in names:
        bgr = read_rgb(os.path.join(rgb_dir, name))
        h, w = bgr.shape[:2]
        s = args.max_size / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        A = torch.from_numpy(rgb_to_norm(bgr)).unsqueeze(0).to(device)
        A, hw = pad_to_multiple(A, 2 ** num_downs)
        with torch.no_grad():
            gen = unpad(translate(nets, targs, A, 0.0), hw)[0, 0].cpu().numpy()
        r_gen.append(corr(lum, gen))

        tp = os.path.join(th_dir, name)
        if os.path.exists(tp):
            real = cv2.imread(tp, cv2.IMREAD_UNCHANGED)
            if real is not None:
                if real.ndim == 3:
                    real = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
                real = cv2.resize(real, (lum.shape[1], lum.shape[0]))
                r_real.append(corr(lum, real))

    g = np.nanmean(r_gen)
    print(f"frames sampled            : {len(r_gen)}")
    print(f"r(generated, RGB luminance): {g:+.3f}   <- want this LOW")
    if r_real:
        print(f"r(real thermal, RGB lum)   : {np.nanmean(r_real):+.3f}   "
              f"<- reference (unregistered)")
    print()
    if g > 0.75:
        print("VERDICT: greyscale shortcut. The model is restyling luminance, "
              "not changing modality. More epochs will not fix this -- change "
              "the objective (drop --lambda-idt, lower --lambda-cyc, or use a "
              "contrastive/CUT-style loss instead of cycle-consistency).")
    elif g > 0.5:
        print("VERDICT: partial. Some genuine modality change, still a strong "
              "luminance component. Worth training further and re-checking.")
    else:
        print("VERDICT: the output has decoupled from visible brightness, "
              "which is what real thermal does. Judge fidelity visually and "
              "with a detector-in-the-loop test from here.")


if __name__ == "__main__":
    main()

"""Translate a folder of RGB images to thermal with a trained ControlNet (+ the frozen SD base).

Inputs are expected in the THERMAL geometry, i.e. registered RGB crops like the training pairs (the model
learned RGB at ~8x below day-camera resolution, warped into the thermal view). Each output is a single-
channel PNG with the input's name and size. Guidance 1 (one conditional pass per step), fixed seed, so
repeated runs give the same image.

    python -m day2thermal.diffusion.infer --controlnet runs/diff_cn/best --input data/diff_pairs/test/rgb \\
        --out runs/diff_cn/preds_test
    python -m day2thermal.gan_gonogo --pred-dir runs/diff_cn/preds_test --pairs data/diff_pairs \\
        --val-pairs data/diff_pairs --val-split test --out runs/diff_cn/gonogo_test
"""
import argparse
import os

import cv2
import numpy as np
import torch

from ..utils import ensure_dir, list_images
from .train_controlnet import generate, load_models, prompt_embeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controlnet", required=True, help="dir saved by train_controlnet (best/ or last/)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    from diffusers import ControlNetModel
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok, te, vae, unet, _, _ = load_models(a.pretrained, device, dtype)
    emb = prompt_embeds(tok, te, device, dtype)
    cn = ControlNetModel.from_pretrained(a.controlnet).to(device, dtype).eval()
    out = ensure_dir(a.out)
    names = list_images(a.input)
    for i0 in range(0, len(names), a.batch):
        chunk = names[i0:i0 + a.batch]
        imgs = [cv2.imread(os.path.join(a.input, n), cv2.IMREAD_COLOR)[:, :, ::-1] for n in chunk]
        cond = torch.stack([torch.from_numpy(cv2.resize(np.ascontiguousarray(im), (a.width, a.height),
                                                        interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float() / 255
                            for im in imgs]).to(device)
        pred = generate(cn, vae, unet, emb, cond, a.steps, a.seed, a.pretrained).cpu().numpy()
        for n, im, p in zip(chunk, imgs, pred):
            h, w = im.shape[:2]
            cv2.imwrite(os.path.join(out, os.path.splitext(n)[0] + ".png"),
                        (cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) * 255).round().astype(np.uint8))
        print(f"  {min(i0 + a.batch, len(names))}/{len(names)}", flush=True)
    print(f"wrote {len(names)} predictions -> {out}")


if __name__ == "__main__":
    main()

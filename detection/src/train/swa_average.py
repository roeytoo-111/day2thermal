r"""
Stochastic weight averaging for an Ultralytics run: average the per-epoch checkpoints of a constant-learning-rate
continuation (`yolo train model=<best.pt> ... lrf=1 warmup_epochs=0 save_period=1`) and re-estimate the batch-norm
statistics on training images (averaged weights + stale BN statistics break the detector: the 10-13 soup test fired on
65 of 3,396 frames until BN was re-estimated).

Uses the weights stored in each epoch checkpoint (Ultralytics saves the EMA copy there), averages the last
--last-n of them, then runs --bn-images training images through the model in train mode with cumulative BN averaging.

    python3 src/train/swa_average.py --run ../runs/thermal_uav/bosonN2_s0_swa --data data/thermal-7-N2-u0 \
        --last-n 10 --out ../runs/thermal_uav/bosonN2_s0_swa/weights/swa.pt
"""
import os
import re
import glob
import random
import argparse

import cv2
import numpy as np
import torch
from torch import nn


def letterbox(im, size):
    h, w = im.shape[:2]; r = size / max(h, w)
    im = cv2.resize(im, (int(round(w * r)), int(round(h * r))))
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    out[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return torch.from_numpy(out[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--last-n", type=int, default=10); ap.add_argument("--bn-images", type=int, default=1500)
    ap.add_argument("--imgsz", type=int, default=512); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    eps = sorted(glob.glob(os.path.join(a.run, "weights", "epoch*.pt")), key=lambda p: int(re.findall(r"epoch(\d+)", p)[0]))
    eps = eps[-a.last_n:]
    if len(eps) < 2:
        raise SystemExit(f"need >= 2 epoch checkpoints in {a.run}/weights (train with save_period=1)")
    print("averaging", [os.path.basename(e) for e in eps])
    base = torch.load(eps[-1], map_location="cpu", weights_only=False)
    acc = None
    for e in eps:
        ck = torch.load(e, map_location="cpu", weights_only=False)
        sd = (ck.get("ema") or ck["model"]).float().state_dict()
        if acc is None:
            acc = {k: v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    acc[k] += v
    for k, v in acc.items():
        if v.dtype.is_floating_point:
            acc[k] = v / len(eps)
    m = (base.get("ema") or base["model"]).float()
    m.load_state_dict(acc)
    imgs = sorted(glob.glob(os.path.join(a.data, "train", "images", "*")))
    random.seed(0); imgs = random.sample(imgs, min(a.bn_images, len(imgs)))
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats(); mod.momentum = None
    m.to(a.device).train()
    with torch.no_grad():
        for i in range(0, len(imgs), 32):
            x = torch.stack([letterbox(cv2.imread(p), a.imgsz) for p in imgs[i:i + 32]]).to(a.device)
            m(x)
    m.eval().cpu()
    base["model"] = m.half(); base["ema"] = None
    base["train_args"] = dict(base.get("train_args", {}), swa_of=[os.path.basename(e) for e in eps])
    torch.save(base, a.out)
    print(f"saved {a.out} (SWA of {len(eps)} epochs, BN re-estimated on {len(imgs)} training images)")


if __name__ == "__main__":
    main()

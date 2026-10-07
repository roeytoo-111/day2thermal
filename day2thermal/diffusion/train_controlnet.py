"""ControlNet (RGB condition) on a frozen, pretrained Stable Diffusion 1.5 -> thermal images.

The base model stays frozen; only the ControlNet (initialised from the SD UNet encoder) is trained, which
is what makes ~700 pairs workable. Targets are the registered thermal crops, replicated to 3 channels.

Augmentation (training only):
  geometric, IDENTICAL on RGB and thermal: horizontal flip, random crop of 70-100% of the area,
      rotation within +-5 deg (the rig rolls ~1 deg; 15 deg would invent corner content)
  photometric, RGB ONLY: brightness / contrast / saturation jitter
  thermal target: untouched. (It is 8-bit AGC video, i.e. already a relative scale, not radiometric
      temperature -- but perturbing it would only add unpredictable target noise.)

Validation every --val-every steps on data/.../val: DDIM sampling at a fixed seed, then the same metrics
as gan_gonogo.py (L1, SSIM, r with RGB luminance). The ControlNet with the best val L1 is kept in
<out>/best; the last one in <out>/last.

    python -m day2thermal.diffusion.train_controlnet --data data/diff_pairs --out runs/diff_cn \\
        --pretrained stable-diffusion-v1-5/stable-diffusion-v1-5 --steps 10000 --batch 4 --grad-accum 2
"""
import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..eval import ssim_pair
from ..utils import ensure_dir, list_images

PROMPT = "a thermal infrared image, white hot"


# ------------------------------------------------------------------ data
def load_pair(root, split, name):
    rgb = cv2.imread(os.path.join(root, split, "rgb", name), cv2.IMREAD_COLOR)[:, :, ::-1]
    th = cv2.imread(os.path.join(root, split, "thermal", name), cv2.IMREAD_GRAYSCALE)
    return rgb, th


def augment(rgb, th, rot_deg, rng):
    h, w = th.shape
    s = math.sqrt(rng.uniform(0.7, 1.0))
    ch, cw = max(int(h * s), 16), max(int(w * s), 16)
    y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
    rgb, th = rgb[y0:y0 + ch, x0:x0 + cw], th[y0:y0 + ch, x0:x0 + cw]
    if rot_deg > 0:
        ang = rng.uniform(-rot_deg, rot_deg)
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
        rgb = cv2.warpAffine(rgb, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        th = cv2.warpAffine(th, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        rgb, th = rgb[:, ::-1], th[:, ::-1]
    # RGB-only photometric jitter
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    c = rng.uniform(0.85, 1.15)
    rgb = np.clip((rgb.astype(np.float32) - 128) * c + 128, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(th)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, size, train, rot_deg=5.0, seed=0):
        self.root, self.split, self.size, self.train, self.rot = root, split, size, train, rot_deg
        self.names = list_images(os.path.join(root, split, "rgb"))
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rgb, th = load_pair(self.root, self.split, self.names[i])
        if self.train:
            rgb, th = augment(rgb, th, self.rot, self.rng)
        W, H = self.size
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        th = cv2.resize(th, (W, H), interpolation=cv2.INTER_AREA)
        cond = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0              # ControlNet: [0, 1]
        tgt = torch.from_numpy(th).float().div(127.5).sub(1.0)[None].repeat(3, 1, 1)  # VAE: [-1, 1]
        return {"cond": cond, "target": tgt, "name": self.names[i]}


# ------------------------------------------------------------------ model
def load_models(pretrained, device, dtype):
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder").to(device, dtype)
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae").to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet").to(device, dtype)
    sched = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
    for m in (te, vae, unet):
        m.requires_grad_(False)
        m.eval()
    return tok, te, vae, unet, sched, ControlNetModel


@torch.no_grad()
def prompt_embeds(tok, te, device, dtype, prompt=PROMPT):
    ids = tok([prompt], padding="max_length", max_length=tok.model_max_length, truncation=True,
              return_tensors="pt").input_ids.to(device)
    return te(ids)[0].to(dtype)


@torch.no_grad()
def generate(controlnet, vae, unet, emb, cond, steps, seed, pretrained):
    """cond: B,3,H,W in [0,1] -> B,H,W thermal in [0,1] (guidance 1: a single conditional pass per step)."""
    from diffusers import DDIMScheduler
    sch = DDIMScheduler.from_pretrained(pretrained, subfolder="scheduler")
    sch.set_timesteps(steps)
    dev, dt = cond.device, emb.dtype
    B, _, H, W = cond.shape
    f = 2 ** (len(vae.config.block_out_channels) - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn((B, unet.config.in_channels, H // f, W // f), generator=g).to(dev, dt) * sch.init_noise_sigma
    e = emb.expand(B, -1, -1)
    # same mixed precision as the training step: the trainable ControlNet keeps fp32 weights while the
    # frozen UNet / VAE and the inputs are bf16/fp16 -- without autocast the matmuls see mixed dtypes
    amp = torch.autocast(dev.type, dtype=dt, enabled=dt != torch.float32)
    with amp:
        for t in sch.timesteps:
            inp = sch.scale_model_input(lat, t)
            down, mid = controlnet(inp, t, encoder_hidden_states=e, controlnet_cond=cond.to(dt), return_dict=False)
            eps = unet(inp, t, encoder_hidden_states=e, down_block_additional_residuals=[d.to(dt) for d in down],
                       mid_block_additional_residual=mid.to(dt)).sample
            lat = sch.step(eps.to(lat.dtype), t, lat).prev_sample
        img = vae.decode((lat / vae.config.scaling_factor).to(dt)).sample.float()
    return ((img.mean(1) + 1) / 2).clamp(0, 1)


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")


@torch.no_grad()
def validate(controlnet, vae, unet, emb, root, split, size, steps, device, pretrained, max_n=64, strips_to=None):
    ds = PairDataset(root, split, size, train=False)
    rows, strips = [], []
    controlnet.eval()
    for i0 in range(0, min(len(ds), max_n), 8):
        batch = [ds[i] for i in range(i0, min(i0 + 8, len(ds), max_n))]
        cond = torch.stack([b["cond"] for b in batch]).to(device)
        pred = generate(controlnet, vae, unet, emb, cond, steps, 0, pretrained).cpu().numpy()
        for b, p in zip(batch, pred):
            rgb, th = load_pair(root, split, b["name"])
            h, w = th.shape
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
            t = th.astype(np.float64) / 255.0
            lum = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            rows.append({"l1": float(np.abs(p - t).mean()), "ssim": ssim_pair(p, t), "r_lum": corr(p, lum)})
            if strips_to is not None and len(strips) < 6:
                to8 = lambda z: cv2.cvtColor((np.clip(z, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                strips.append(np.hstack([np.ascontiguousarray(rgb[:, :, ::-1]), to8(t), to8(p)]))
    controlnet.train()
    if strips_to and strips:
        cv2.imwrite(strips_to, np.vstack([cv2.resize(s, (strips[0].shape[1], strips[0].shape[0])) for s in strips]))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prep.py output (train/, val/, test/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10000, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rot-deg", type=float, default=5.0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-steps", type=int, default=20, help="DDIM steps for validation sampling")
    ap.add_argument("--val-max", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6, help="stop after this many validations without improvement")
    ap.add_argument("--mixed", default="bf16", choices=["bf16", "fp16", "no"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--init-controlnet", default=None,
                    help="start from a saved ControlNet (e.g. runs/diff_cn_all/best) instead of the SD UNet copy; "
                         "the optimizer restarts, so consider a lower --lr")
    ap.add_argument("--snr-gamma", type=float, default=0.0,
                    help="Min-SNR loss weighting min(SNR, gamma)/SNR per sample (e.g. 5); 0 = plain MSE")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    frozen_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[a.mixed]
    if device.type == "cpu":
        frozen_dtype = torch.float32
    out = ensure_dir(a.out)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=2)

    tok, te, vae, unet, sched, ControlNetModel = load_models(a.pretrained, device, frozen_dtype)
    emb = prompt_embeds(tok, te, device, frozen_dtype)
    # ControlNet's condition encoder must downsample like the VAE (SD1.5: 8x -> the default 4 stages)
    f = 2 ** (len(vae.config.block_out_channels) - 1)
    ch = (16, 32, 96, 256)[:int(math.log2(f)) + 1]
    if a.init_controlnet:                    # continue from a saved ControlNet (fresh optimizer and lr)
        controlnet = ControlNetModel.from_pretrained(a.init_controlnet, torch_dtype=torch.float32).to(device)
        print(f"initialised ControlNet from {a.init_controlnet}")
    else:
        controlnet = ControlNetModel.from_unet(unet.float(), conditioning_embedding_out_channels=ch).to(device)  # fp32 master
    unet.to(frozen_dtype)
    controlnet.train()
    if device.type == "cuda":
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(controlnet.parameters(), lr=a.lr, weight_decay=1e-2)
    ds = PairDataset(a.data, "train", (a.width, a.height), train=True, rot_deg=a.rot_deg, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     num_workers=a.workers if device.type == "cuda" else 0)
    scale = vae.config.scaling_factor
    log = open(os.path.join(out, "log.jsonl"), "a")
    best, bad, step, t0 = float("inf"), 0, 0, time.time()
    it = iter(dl)
    while step < a.steps:
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            cond, tgt = b["cond"].to(device), b["target"].to(device)
            with torch.no_grad():
                lat = vae.encode(tgt.to(frozen_dtype)).latent_dist.sample() * scale
            noise = torch.randn_like(lat)
            t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=device).long()
            noisy = sched.add_noise(lat, noise, t)
            with torch.autocast(device.type, dtype=frozen_dtype, enabled=device.type == "cuda" and a.mixed != "no"):
                e = emb.expand(lat.shape[0], -1, -1)
                down, mid = controlnet(noisy, t, encoder_hidden_states=e, controlnet_cond=cond, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=e,
                            down_block_additional_residuals=[d.to(frozen_dtype) for d in down],
                            mid_block_additional_residual=mid.to(frozen_dtype)).sample
            if a.snr_gamma > 0:                                         # Min-SNR-gamma (epsilon target)
                ac = sched.alphas_cumprod.to(device)[t].float()
                snr = ac / (1 - ac)
                w = torch.clamp(snr, max=a.snr_gamma) / snr
                per = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
                loss = (per * w).mean() / a.grad_accum
            else:
                loss = F.mse_loss(pred.float(), noise.float()) / a.grad_accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{a.steps}  loss {tot:.4f}  {(time.time() - t0) / step:.2f} s/step", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            controlnet.save_pretrained(os.path.join(out, "last"))       # before validating: never lose training
            m = validate(controlnet, vae, unet, emb, a.data, "val", (a.width, a.height), a.val_steps, device,
                         a.pretrained, a.val_max, strips_to=os.path.join(out, f"val_{step:06d}.jpg"))
            m.update({"step": step, "loss": tot})
            log.write(json.dumps(m) + "\n")
            log.flush()
            print(f"  VAL step {step}: L1 {m['l1']:.4f}  SSIM {m['ssim']:.4f}  r(lum) {m['r_lum']:+.3f}", flush=True)
            if m["l1"] < best:
                best, bad = m["l1"], 0
                controlnet.save_pretrained(os.path.join(out, "best"))
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"early stop: no val improvement in {a.patience} validations")
                    break
    print(f"done. best val L1 {best:.4f} -> {os.path.join(out, 'best')}")


if __name__ == "__main__":
    main()

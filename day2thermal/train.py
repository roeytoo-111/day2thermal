"""Train a day->thermal translation model on aligned RGB/thermal pairs.

Two modes
---------
pix2pix (default)
    Single generator G: RGB -> thermal, conditional PatchGAN + L1.
    Fast, strong baseline. Start here.

two_stage (ThermalGAN-inspired)
    G1: {RGB, temperature plane [, z]} -> S  (low-frequency thermal base)
    G2: {RGB, S}                       -> R  (relative thermal contrasts)
    B = S + R.
    S is an annotation-free surrogate of ThermalGAN's "thermal segmentation":
    a Gaussian low-pass of the ground-truth thermal frame (--lowpass-sigma).
    The temperature condition T defaults to the frame's mean thermal value; if
    you log ambient temperature per frame, pass --temps-csv for a physically
    meaningful, controllable condition. At inference you can sweep T to obtain
    a multimodal thermal set per RGB image (like ThermalGAN's probe sets).

Examples
--------
python -m day2thermal.train --data-root data/flight01/aligned --out runs/p2p \
    --mode pix2pix --crop-size 256 --batch-size 8

python -m day2thermal.train --data-root data/flight01/aligned --out runs/2stage \
    --mode two_stage --crop-size 256 --lambda-temp 10
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (PairedDayThermalDataset, UnpairedDayThermalDataset,
                      temporal_chunk_split)
from .losses import GANLoss, kl_loss
from .networks import define_D, define_E, define_G
from .torch_utils import (ImagePool, gaussian_blur, set_requires_grad, set_seed,
                          t_rgb_to_uint8, t_thermal_to_uint8)
from .utils import ThermalNorm, ensure_dir, list_images, make_grid_rows


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["pix2pix", "two_stage", "cyclegan"],
                    default="pix2pix")
    # data
    ap.add_argument("--load-size", type=int, default=286,
                    help="squash-resize before cropping; 0 = keep native resolution")
    # unpaired (cyclegan) data options -- --data-root then points at the RAW
    # extractor output (root/rgb, root/thermal); no registration is used.
    ap.add_argument("--rgb-width", type=int, default=0,
                    help="unpaired: downscale RGB to this width before cropping "
                         "so crops match the thermal angular scale (0 = thermal width)")
    ap.add_argument("--skip-first", type=int, default=0,
                    help="unpaired: drop the first N frames (e.g. pre-launch "
                         "ground footage that is not representative)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="unpaired: fraction of temporal chunks held out")
    ap.add_argument("--chunk", type=int, default=200,
                    help="unpaired: frames per contiguous split chunk")
    ap.add_argument("--lambda-cyc", type=float, default=10.0,
                    help="cyclegan: cycle-consistency weight")
    ap.add_argument("--lambda-idt", type=float, default=5.0,
                    help="cyclegan: identity weight (0 disables)")
    ap.add_argument("--pool-size", type=int, default=50,
                    help="cyclegan: discriminator history buffer (0 disables)")
    ap.add_argument("--crop-size", type=int, default=256,
                    help="training crop (power of two, >= 32)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--temps-csv", default=None,
                    help="CSV 'filename,celsius' with ambient temperature per frame")
    # thermal normalization
    ap.add_argument("--thermal-mode", choices=["rel8", "abs16"], default="rel8")
    ap.add_argument("--raw-scale", type=float, default=0.04)
    ap.add_argument("--raw-offset", type=float, default=-273.15)
    ap.add_argument("--tmin", type=float, default=-20.0)
    ap.add_argument("--tmax", type=float, default=80.0)
    # model
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--norm", choices=["instance", "batch", "none"], default="instance")
    ap.add_argument("--no-extra-head", action="store_true",
                    help="disable the ThermalGAN-style extra output conv")
    ap.add_argument("--dropout", action="store_true")
    ap.add_argument("--n-layers-d", type=int, default=3)
    ap.add_argument("--num-scales-d", type=int, default=1,
                    help="2 enables a pix2pixHD-style multi-scale discriminator")
    # two-stage specifics
    ap.add_argument("--lowpass-sigma", type=float, default=8.0,
                    help="Gaussian sigma (px at crop scale) of the thermal base S")
    ap.add_argument("--lambda-temp", type=float, default=10.0,
                    help="weight of |mean(S_hat) - T| (latent temperature loss)")
    ap.add_argument("--use-vae", action="store_true",
                    help="add a BicycleGAN-style VAE latent (encoder E on S + KL)")
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--lambda-kl", type=float, default=0.01)
    # optimization (ThermalGAN / pix2pix defaults)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--gan-mode", choices=["lsgan", "vanilla"], default="lsgan")
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--n-epochs-decay", type=int, default=100)
    # bookkeeping
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-freq", type=int, default=10, help="epochs between checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-iters", type=int, default=None,
                    help="debug: cap iterations per epoch")
    ap.add_argument("--val-max-batches", type=int, default=50,
                    help="cap validation batches per epoch (0 = all)")
    return ap


def make_temperature_plane(T, h, w):
    return T.view(-1, 1, 1, 1).expand(-1, 1, h, w)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        set_seed(args.seed)

        assert args.crop_size >= 32 and (args.crop_size & (args.crop_size - 1)) == 0, \
            "--crop-size must be a power of two >= 32"
        self.num_downs = int(math.log2(args.crop_size))
        self.extra_head = not args.no_extra_head

        self.tn = ThermalNorm(args.thermal_mode, args.raw_scale,
                              args.raw_offset, args.tmin, args.tmax)
        if args.mode == "cyclegan":
            self.train_ds, self.val_ds = self.build_unpaired_datasets()
        else:
            self.train_ds = PairedDayThermalDataset(
                args.data_root, "train", self.tn, args.load_size, args.crop_size,
                augment=True, temps_csv=args.temps_csv)
            self.val_ds = PairedDayThermalDataset(
                args.data_root, "val", self.tn, args.load_size, args.crop_size,
                augment=False, temps_csv=args.temps_csv)
        self.train_loader = DataLoader(self.train_ds, args.batch_size,
                                       shuffle=True, num_workers=args.num_workers,
                                       drop_last=True)
        self.val_loader = DataLoader(self.val_ds, args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)

        self.gan = GANLoss(args.gan_mode).to(self.device)
        self.l1 = torch.nn.L1Loss()
        self.build_nets()
        self.build_optim()

        self.ckpt_dir = ensure_dir(os.path.join(args.out, "checkpoints"))
        self.sample_dir = ensure_dir(os.path.join(args.out, "samples"))
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump({"args": vars(args), "thermal_norm": self.tn.to_dict()},
                      f, indent=2)

        self.start_epoch = 0
        if args.resume:
            self.load(args.resume)

        n_params = sum(sum(p.numel() for p in n.parameters())
                       for n in self.nets.values())
        print(f"device={self.device} | train={len(self.train_ds)} "
              f"val={len(self.val_ds)} | params={n_params/1e6:.1f}M | "
              f"mode={args.mode}")

    # -------------------------------------------------------- unpaired data
    def build_unpaired_datasets(self):
        """--data-root is the RAW extractor output here (root/rgb, root/thermal).
        The two domains are split independently but with the same temporal-chunk
        rule, so a val RGB frame never has its own thermal twin in train."""
        a = self.args
        rgb_dir = os.path.join(a.data_root, "rgb")
        th_dir = os.path.join(a.data_root, "thermal")
        for d in (rgb_dir, th_dir):
            if not os.path.isdir(d):
                raise SystemExit(
                    f"--mode cyclegan expects raw frames at {d}. Point "
                    f"--data-root at the extract_frames output (containing "
                    f"rgb/ and thermal/), not the registered 'aligned' folder.")
        rgb_names = list_images(rgb_dir)[a.skip_first:]
        th_names = list_images(th_dir)[a.skip_first:]
        rgb_tr, rgb_va = temporal_chunk_split(rgb_names, a.val_frac, a.chunk)
        th_tr, th_va = temporal_chunk_split(th_names, a.val_frac, a.chunk)
        mk = lambda rn, tn_, aug: UnpairedDayThermalDataset(  # noqa: E731
            rgb_dir, th_dir, self.tn, a.crop_size, rn, tn_,
            rgb_width=a.rgb_width, augment=aug)
        print(f"unpaired: rgb {len(rgb_tr)}/{len(rgb_va)} train/val, "
              f"thermal {len(th_tr)}/{len(th_va)} train/val"
              + (f" (skipped first {a.skip_first})" if a.skip_first else ""))
        return mk(rgb_tr, th_tr, True), mk(rgb_va, th_va, False)

    # ------------------------------------------------------------------ nets
    def build_nets(self):
        a = self.args
        common = dict(ngf=a.ngf, num_downs=self.num_downs, norm=a.norm,
                      use_dropout=a.dropout, extra_head=self.extra_head)
        self.nets = {}
        if a.mode == "pix2pix":
            self.nets["G"] = define_G(3, 1, **common)
            self.nets["D"] = define_D(4, a.ndf, a.n_layers_d, a.norm, a.num_scales_d)
        elif a.mode == "cyclegan":
            # G: RGB -> thermal (the one infer.py uses), F: thermal -> RGB.
            # Discriminators are UNconditional here: with no correspondence
            # there is nothing to condition on, so D sees only its own domain.
            self.nets["G"] = define_G(3, 1, **common)
            self.nets["F"] = define_G(1, 3, **common)
            self.nets["D_B"] = define_D(1, a.ndf, a.n_layers_d, a.norm, a.num_scales_d)
            self.nets["D_A"] = define_D(3, a.ndf, a.n_layers_d, a.norm, a.num_scales_d)
        else:
            g1_in = 4 + (a.nz if a.use_vae else 0)  # RGB + T plane [+ z planes]
            self.nets["G1"] = define_G(g1_in, 1, **common)
            self.nets["G2"] = define_G(4, 1, **common)          # RGB + S
            self.nets["D1"] = define_D(5, a.ndf, a.n_layers_d, a.norm, a.num_scales_d)
            self.nets["D2"] = define_D(4, a.ndf, a.n_layers_d, a.norm, a.num_scales_d)
            if a.use_vae:
                self.nets["E"] = define_E(1, a.nz)
        for n in self.nets.values():
            n.to(self.device)

    def build_optim(self):
        a = self.args
        betas = (a.beta1, a.beta2)
        self.opts = {}
        if a.mode == "pix2pix":
            self.opts["G"] = torch.optim.Adam(self.nets["G"].parameters(), a.lr, betas)
            self.opts["D"] = torch.optim.Adam(self.nets["D"].parameters(), a.lr, betas)
        elif a.mode == "cyclegan":
            self.opts["G"] = torch.optim.Adam(
                list(self.nets["G"].parameters()) + list(self.nets["F"].parameters()),
                a.lr, betas)
            self.opts["D"] = torch.optim.Adam(
                list(self.nets["D_A"].parameters()) + list(self.nets["D_B"].parameters()),
                a.lr, betas)
            self.pool_A = ImagePool(a.pool_size)
            self.pool_B = ImagePool(a.pool_size)
        else:
            g1_params = list(self.nets["G1"].parameters())
            if a.use_vae:
                g1_params += list(self.nets["E"].parameters())
            self.opts["G1"] = torch.optim.Adam(g1_params, a.lr, betas)
            self.opts["G2"] = torch.optim.Adam(self.nets["G2"].parameters(), a.lr, betas)
            self.opts["D1"] = torch.optim.Adam(self.nets["D1"].parameters(), a.lr, betas)
            self.opts["D2"] = torch.optim.Adam(self.nets["D2"].parameters(), a.lr, betas)

        def lr_lambda(epoch):
            a_ = self.args
            return max(0.0, 1.0 - max(0, epoch + 1 - a_.n_epochs)
                       / max(1, a_.n_epochs_decay))
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda)
                       for o in self.opts.values()]

    # ------------------------------------------------------------------ steps
    def step_pix2pix(self, batch):
        A = batch["A"].to(self.device)
        B = batch["B"].to(self.device)
        G, D = self.nets["G"], self.nets["D"]
        fake = G(A)

        set_requires_grad(D, True)
        self.opts["D"].zero_grad()
        loss_d = 0.5 * (self.gan(D(torch.cat([A, fake.detach()], 1)), False) +
                        self.gan(D(torch.cat([A, B], 1)), True))
        loss_d.backward()
        self.opts["D"].step()

        set_requires_grad(D, False)
        self.opts["G"].zero_grad()
        loss_g_gan = self.gan(D(torch.cat([A, fake], 1)), True)
        loss_g_l1 = self.l1(fake, B) * self.args.lambda_l1
        (loss_g_gan + loss_g_l1).backward()
        self.opts["G"].step()

        return {"D": loss_d.item(), "G_gan": loss_g_gan.item(),
                "G_L1": loss_g_l1.item()}, \
               {"A": A, "B": B, "fake": fake.detach()}

    def step_cyclegan(self, batch):
        """CycleGAN. No pixel correspondence is used anywhere: the only signals
        are (a) does the output look like the target domain, and (b) can the
        round trip be inverted. A and B in a batch are unrelated scenes."""
        a = self.args
        A = batch["A"].to(self.device)          # RGB    3xHxW
        B = batch["B"].to(self.device)          # thermal 1xHxW
        G, F_, D_A, D_B = (self.nets[k] for k in ("G", "F", "D_A", "D_B"))

        fake_B = G(A)                 # RGB -> thermal
        rec_A = F_(fake_B)            # and back
        fake_A = F_(B)                # thermal -> RGB
        rec_B = G(fake_A)

        # ---- generators
        set_requires_grad([D_A, D_B], False)
        self.opts["G"].zero_grad()
        g_gan = self.gan(D_B(fake_B), True) + self.gan(D_A(fake_A), True)
        g_cyc = (self.l1(rec_A, A) + self.l1(rec_B, B)) * a.lambda_cyc
        g_total = g_gan + g_cyc
        losses = {}
        if a.lambda_idt > 0:
            # Classical CycleGAN identity (G(B)~B, F(A)~A) does not typecheck
            # here: G is 3->1 and F is 1->3. The channel-adapted equivalent
            # keeps the intent -- feeding a generator something already in its
            # output domain should change it as little as possible, which
            # anchors global intensity and prevents contrast inversion.
            idt = (self.l1(G(B.expand(-1, 3, -1, -1)), B) +
                   self.l1(F_(A.mean(1, keepdim=True)), A)) * a.lambda_idt
            g_total = g_total + idt
            losses["G_idt"] = idt.item()
        g_total.backward()
        self.opts["G"].step()

        # ---- discriminators (history buffer to damp oscillation)
        set_requires_grad([D_A, D_B], True)
        self.opts["D"].zero_grad()
        fb = self.pool_B.query(fake_B.detach())
        fa = self.pool_A.query(fake_A.detach())
        loss_d = 0.5 * (self.gan(D_B(B), True) + self.gan(D_B(fb), False)) + \
                 0.5 * (self.gan(D_A(A), True) + self.gan(D_A(fa), False))
        loss_d.backward()
        self.opts["D"].step()

        losses.update({"D": loss_d.item(), "G_gan": g_gan.item(),
                       "G_cyc": g_cyc.item()})
        vis = {"A": A, "B": B, "fake_B": fake_B.detach(),
               "rec_A": rec_A.detach(), "fake_A": fake_A.detach()}
        return losses, vis

    def step_two_stage(self, batch):
        a = self.args
        A = batch["A"].to(self.device)
        B = batch["B"].to(self.device)
        T = batch["T"].to(self.device)
        h, w = B.shape[-2:]

        S = gaussian_blur(B, a.lowpass_sigma)
        R = B - S
        Tpl = make_temperature_plane(T, h, w)

        G1, G2, D1, D2 = (self.nets[k] for k in ("G1", "G2", "D1", "D2"))

        in1 = torch.cat([A, Tpl], 1)
        losses = {}
        mu = logvar = None
        if a.use_vae:
            mu, logvar = self.nets["E"](S)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            zpl = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, h, w)
            in1 = torch.cat([in1, zpl], 1)
        S_hat = G1(in1)

        # ---- stage 1 discriminator (conditioned on A and T)
        set_requires_grad(D1, True)
        self.opts["D1"].zero_grad()
        loss_d1 = 0.5 * (self.gan(D1(torch.cat([A, Tpl, S_hat.detach()], 1)), False) +
                         self.gan(D1(torch.cat([A, Tpl, S], 1)), True))
        loss_d1.backward()
        self.opts["D1"].step()

        # ---- stage 1 generator (+ encoder)
        set_requires_grad(D1, False)
        self.opts["G1"].zero_grad()
        g1_gan = self.gan(D1(torch.cat([A, Tpl, S_hat], 1)), True)
        g1_l1 = self.l1(S_hat, S) * a.lambda_l1
        g1_temp = self.l1(S_hat.mean(dim=(1, 2, 3)), T) * a.lambda_temp
        g1_total = g1_gan + g1_l1 + g1_temp
        if a.use_vae:
            g1_kl = kl_loss(mu, logvar) * a.lambda_kl
            g1_total = g1_total + g1_kl
            losses["G1_kl"] = g1_kl.item()
        g1_total.backward()
        self.opts["G1"].step()

        # ---- stage 2 (trained independently; S_hat detached, as in the paper)
        S_in = S_hat.detach()
        R_hat = G2(torch.cat([A, S_in], 1))

        set_requires_grad(D2, True)
        self.opts["D2"].zero_grad()
        loss_d2 = 0.5 * (self.gan(D2(torch.cat([A, R_hat.detach()], 1)), False) +
                         self.gan(D2(torch.cat([A, R], 1)), True))
        loss_d2.backward()
        self.opts["D2"].step()

        set_requires_grad(D2, False)
        self.opts["G2"].zero_grad()
        g2_gan = self.gan(D2(torch.cat([A, R_hat], 1)), True)
        g2_l1 = self.l1(R_hat, R) * a.lambda_l1
        (g2_gan + g2_l1).backward()
        self.opts["G2"].step()

        B_hat = (S_in + R_hat.detach()).clamp(-1, 1)
        losses.update({"D1": loss_d1.item(), "G1_gan": g1_gan.item(),
                       "G1_L1": g1_l1.item(), "G1_T": g1_temp.item(),
                       "D2": loss_d2.item(), "G2_gan": g2_gan.item(),
                       "G2_L1": g2_l1.item()})
        vis = {"A": A, "B": B, "S": S, "S_hat": S_hat.detach(),
               "R": R, "R_hat": R_hat.detach(), "B_hat": B_hat}
        return losses, vis

    # ------------------------------------------------------------ validation
    @torch.no_grad()
    def forward_full(self, A, T):
        a = self.args
        if a.mode == "pix2pix":
            return self.nets["G"](A)
        h, w = A.shape[-2:]
        Tpl = make_temperature_plane(T, h, w)
        in1 = torch.cat([A, Tpl], 1)
        if a.use_vae:
            z = torch.zeros(A.size(0), a.nz, device=A.device)
            in1 = torch.cat([in1, z.view(-1, a.nz, 1, 1).expand(-1, -1, h, w)], 1)
        S_hat = self.nets["G1"](in1)
        R_hat = self.nets["G2"](torch.cat([A, S_hat], 1))
        return (S_hat + R_hat).clamp(-1, 1)

    @torch.no_grad()
    def validate(self):
        for n in self.nets.values():
            n.eval()
        total, count = 0.0, 0
        for vi, batch in enumerate(self.val_loader):
            if self.args.val_max_batches and vi >= self.args.val_max_batches:
                break
            A = batch["A"].to(self.device)
            B = batch["B"].to(self.device)
            T = batch["T"].to(self.device)
            if self.args.mode == "cyclegan":
                # There is NO paired ground truth in the unpaired setting, so a
                # val L1 against B would be comparing unrelated scenes. Report
                # cycle reconstruction ||F(G(A)) - A|| instead: it measures how
                # invertible the mapping is, NOT how thermal-accurate it is.
                total += torch.abs(self.nets["F"](self.nets["G"](A)) - A
                                   ).mean().item() * A.size(0)
            else:
                total += torch.abs(self.forward_full(A, T) - B
                                   ).mean().item() * A.size(0)
            count += A.size(0)
        for n in self.nets.values():
            n.train()
        l1 = total / max(1, count)
        if self.args.mode == "cyclegan":
            return l1, f"val cycle-L1 = {l1:.4f}  (not a fidelity metric)"
        msg = f"val L1 = {l1:.4f}"
        if self.tn.mode == "abs16":
            msg += f" ({l1 * self.tn.celsius_span() / 2:.2f} degC)"
        return l1, msg

    # ------------------------------------------------------------- samples/io
    def save_samples(self, vis, epoch, it):
        rows = []
        n = min(4, vis["A"].size(0))
        for i in range(n):
            row = [t_rgb_to_uint8(vis["A"][i]), t_thermal_to_uint8(vis["B"][i])]
            if self.args.mode == "pix2pix":
                row.append(t_thermal_to_uint8(vis["fake"][i]))
            elif self.args.mode == "cyclegan":
                # NB: column 2 (B) is an UNRELATED real thermal frame, shown as
                # a domain reference -- it is not the target for column 1.
                row += [t_thermal_to_uint8(vis["fake_B"][i]),
                        t_rgb_to_uint8(vis["rec_A"][i]),
                        t_rgb_to_uint8(vis["fake_A"][i])]
            else:
                row += [t_thermal_to_uint8(vis["S"][i]),
                        t_thermal_to_uint8(vis["S_hat"][i]),
                        t_thermal_to_uint8(vis["B_hat"][i])]
            rows.append(row)
        grid = make_grid_rows(rows)
        cv2.imwrite(os.path.join(self.sample_dir,
                                 f"e{epoch:03d}_i{it:05d}.png"), grid)

    def save(self, epoch, tag):
        state = {"epoch": epoch,
                 "args": vars(self.args),
                 "thermal_norm": self.tn.to_dict(),
                 "nets": {k: n.state_dict() for k, n in self.nets.items()},
                 "opts": {k: o.state_dict() for k, o in self.opts.items()}}
        torch.save(state, os.path.join(self.ckpt_dir, f"{tag}.pt"))

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        for k, sd in state["nets"].items():
            self.nets[k].load_state_dict(sd)
        for k, sd in state.get("opts", {}).items():
            if k in self.opts:
                self.opts[k].load_state_dict(sd)
        self.start_epoch = state.get("epoch", 0)
        # Fast-forward the LR schedulers. They are constructed fresh in
        # build_optim() with last_epoch=-1, so without this they would restart
        # their decay from zero and a decay scheduled for a later epoch would
        # never fire on a resumed run.
        for s in self.scheds:
            for _ in range(self.start_epoch):
                s.step()
        lr_now = self.opts[next(iter(self.opts))].param_groups[0]["lr"]
        print(f"resumed from {path} at epoch {self.start_epoch} (lr={lr_now:.3e})")

    # ---------------------------------------------------------------- driver
    def fit(self):
        a = self.args
        step_fn = {"pix2pix": self.step_pix2pix,
                   "cyclegan": self.step_cyclegan,
                   "two_stage": self.step_two_stage}[a.mode]
        total_epochs = a.n_epochs + a.n_epochs_decay
        for epoch in range(self.start_epoch, total_epochs):
            t0 = time.time()
            for it, batch in enumerate(self.train_loader):
                if a.max_iters and it >= a.max_iters:
                    break
                losses, vis = step_fn(batch)
                if it % a.log_every == 0:
                    msg = " ".join(f"{k}={v:.3f}" for k, v in losses.items())
                    print(f"[e{epoch:03d} i{it:05d}] {msg}", flush=True)
                    self.save_samples(vis, epoch, it)
            for s in self.scheds:
                s.step()
            _, vmsg = self.validate()
            print(f"[e{epoch:03d}] {vmsg} | {time.time() - t0:.1f}s", flush=True)
            self.save(epoch + 1, "latest")
            if (epoch + 1) % a.save_freq == 0:
                self.save(epoch + 1, f"epoch_{epoch + 1:03d}")
        print("training done.")


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out)
    Trainer(args).fit()


if __name__ == "__main__":
    main()

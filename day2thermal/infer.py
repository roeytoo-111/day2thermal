"""Translate a folder of RGB images into synthetic thermal images.

For a two-stage checkpoint you can sweep the temperature condition to obtain a
*multimodal* thermal set per input image (ThermalGAN's key trick, useful to
augment a detector's training data across weather conditions):

    python -m day2thermal.infer --ckpt runs/2stage/checkpoints/latest.pt \
        --input my_day_dataset/images --out my_thermal_dataset \
        --temps 0,15,30

``--temps`` values are degrees Celsius when the model was trained with
``--thermal-mode abs16``; with ``rel8`` they are relative levels in [0, 1].
Outputs are 8-bit PNG previews; add ``--save-16bit`` to also write 16-bit
frames (calibrated counts in abs16 mode).
"""
import argparse
import math
import os

import cv2
import numpy as np
import torch

from .networks import define_G
from .torch_utils import pad_to_multiple, unpad
from .utils import ThermalNorm, ensure_dir, list_images, read_rgb, rgb_to_norm


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    args = state["args"]
    tn = ThermalNorm.from_dict(state["thermal_norm"])
    num_downs = int(math.log2(args["crop_size"]))
    extra_head = not args.get("no_extra_head", False)
    common = dict(ngf=args["ngf"], num_downs=num_downs, norm=args["norm"],
                  use_dropout=False, extra_head=extra_head)
    nets = {}
    if args["mode"] in ("pix2pix", "cyclegan"):
        # cyclegan stores the RGB->thermal direction under the same key, so
        # inference is identical; the reverse generator F is not needed here.
        nets["G"] = define_G(3, 1, **common)
    else:
        g1_in = 4 + (args["nz"] if args.get("use_vae") else 0)
        nets["G1"] = define_G(g1_in, 1, **common)
        nets["G2"] = define_G(4, 1, **common)
    for k, net in nets.items():
        net.load_state_dict(state["nets"][k])
        net.to(device).eval()
    return nets, args, tn, num_downs


@torch.no_grad()
def translate(nets, args, A, t_norm=None):
    if args["mode"] in ("pix2pix", "cyclegan"):
        return nets["G"](A)
    h, w = A.shape[-2:]
    Tpl = torch.full((A.size(0), 1, h, w), float(t_norm),
                     device=A.device, dtype=A.dtype)
    in1 = torch.cat([A, Tpl], 1)
    if args.get("use_vae"):
        nz = args["nz"]
        z = torch.zeros(A.size(0), nz, 1, 1, device=A.device, dtype=A.dtype)
        in1 = torch.cat([in1, z.expand(-1, -1, h, w)], 1)
    S = nets["G1"](in1)
    R = nets["G2"](torch.cat([A, S], 1))
    return (S + R).clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="folder of RGB images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", default=None,
                    help="comma-separated temperature conditions (two_stage only)")
    ap.add_argument("--max-size", type=int, default=None,
                    help="cap the longer image side before translation")
    ap.add_argument("--save-16bit", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    nets, targs, tn, num_downs = load_model(args.ckpt, device)
    multiple = 2 ** num_downs

    if targs["mode"] == "two_stage":
        raw_temps = [float(t) for t in (args.temps or "").split(",") if t != ""] \
            or [None]
    else:
        raw_temps = [None]

    names = list_images(args.input)
    if not names:
        raise SystemExit(f"no images found in {args.input}")
    print(f"{len(names)} images | mode={targs['mode']} | device={device}")

    for name in names:
        bgr = read_rgb(os.path.join(args.input, name))
        if args.max_size:
            h, w = bgr.shape[:2]
            s = args.max_size / max(h, w)
            if s < 1:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
        A = torch.from_numpy(rgb_to_norm(bgr)).unsqueeze(0).to(device)
        A, hw = pad_to_multiple(A, multiple)

        for t in raw_temps:
            if t is None:
                t_norm, sub = 0.0, ""
            elif tn.mode == "abs16":
                t_norm, sub = float(tn.celsius_to_norm(t)), f"T{t:g}C"
            else:
                t_norm, sub = float(t) * 2.0 - 1.0, f"T{t:g}"
            pred = unpad(translate(nets, targs, A, t_norm), hw)
            arr = pred[0, 0].cpu().numpy()

            out_dir = ensure_dir(os.path.join(args.out, sub) if sub else args.out)
            stem = os.path.splitext(name)[0]
            cv2.imwrite(os.path.join(out_dir, stem + ".png"), tn.norm_to_uint8(arr))
            if args.save_16bit:
                cv2.imwrite(os.path.join(out_dir, stem + "_16bit.png"),
                            tn.norm_to_uint16(arr))
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

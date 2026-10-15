r"""
Train an RGB detector with EXACTLY the recipe of yolo26s_fw4_20261011: every train_arg stored in the fw4 checkpoint
(epochs 100, patience 5, batch 10, imgsz 640, optimizer auto, seed 0, close_mosaic 10, ...) is reused; only model
(the same base, yolo26s_small_07072026), data, project, name and device change. So a difference to fw4 comes from the
data (and the GPU / ultralytics patch version, which are logged).

    python3 scripts/train_rgb_like_fw4.py --fw4 ~/models/yolo26s_fw4_20261011.pt --base ~/models/yolo26s_small_07072026.pt \
        --data ~/day2thermal/data/rgb_v5a/data.yaml --name v5a_s0
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fw4", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="../runs/rgb"); ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=None, help="override fw4's seed (0)")
    ap.add_argument("--arch", default=None, help="model yaml (e.g. yolo26s-p2.yaml, extra stride-4 head); --base then only "
                    "initialises the transferable layers (backbone), as the thermal P2 runs did")
    a = ap.parse_args()
    import ultralytics
    from ultralytics import YOLO
    args = dict(torch.load(a.fw4, map_location="cpu", weights_only=False)["train_args"])
    drop = {"model", "data", "project", "name", "device", "save_dir", "resume", "pretrained", "mode", "task", "exist_ok"}  # pretrained: set below
    args = {k: v for k, v in args.items() if k not in drop}
    if a.seed is not None:
        args["seed"] = a.seed
    print(f"ultralytics {ultralytics.__version__} (fw4: 8.4.155); recipe from {a.fw4}:",
          {k: args[k] for k in ("epochs", "patience", "batch", "imgsz", "optimizer", "seed", "close_mosaic")}, flush=True)
    if a.arch:
        YOLO(a.arch).train(data=a.data, project=a.project, name=a.name, device=a.device, exist_ok=True, pretrained=a.base, **args)
    else:
        YOLO(a.base).train(data=a.data, project=a.project, name=a.name, device=a.device, exist_ok=True, **args)


if __name__ == "__main__":
    main()

"""One-time prep for unpaired (cyclegan) training: downscale the RGB frames.

Unpaired training crops RGB at roughly the thermal frame's angular scale, so
every sample otherwise decodes a full 3840x2160 PNG and throws away ~97% of the
pixels -- the data loader, not the GPU, ends up being the bottleneck. Doing the
resize once turns each epoch's decode cost into a fraction of what it was.

Thermal is hard-linked (or copied) unchanged: it is the target domain and must
not be resampled.

    python -m day2thermal.prep_unpaired --raw data/raw --out data/unpaired \
        --width 640 --skip-first 900
"""
import argparse
import os
import shutil

import cv2

from .utils import ensure_dir, list_images

try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional here
    def tqdm(it, desc="", **_):
        it = list(it)
        n = len(it)
        for i, x in enumerate(it):
            if i % 200 == 0 or i == n - 1:
                print(f"\r{desc}: {i + 1}/{n}", end="", flush=True)
            yield x
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="dir with rgb/ and thermal/")
    ap.add_argument("--out", default="data/unpaired")
    ap.add_argument("--width", type=int, default=0,
                    help="target RGB width (0 = match the thermal width)")
    ap.add_argument("--skip-first", type=int, default=0,
                    help="drop the first N frames of each stream")
    ap.add_argument("--copy-thermal", action="store_true",
                    help="copy thermal instead of hard-linking")
    args = ap.parse_args()

    rgb_dir = os.path.join(args.raw, "rgb")
    th_dir = os.path.join(args.raw, "thermal")
    rgb_names = list_images(rgb_dir)[args.skip_first:]
    th_names = list_images(th_dir)[args.skip_first:]
    if not rgb_names or not th_names:
        raise SystemExit(f"nothing to do under {args.raw}")

    width = args.width
    if width <= 0:
        probe = cv2.imread(os.path.join(th_dir, th_names[0]), cv2.IMREAD_UNCHANGED)
        width = probe.shape[1]

    out_rgb = ensure_dir(os.path.join(args.out, "rgb"))
    out_th = ensure_dir(os.path.join(args.out, "thermal"))

    for n in tqdm(rgb_names, desc=f"rgb -> {width}px"):
        dst = os.path.join(out_rgb, n)
        if os.path.exists(dst):
            continue
        img = cv2.imread(os.path.join(rgb_dir, n), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (width, max(1, round(h * width / w))),
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img)

    for n in tqdm(th_names, desc="thermal (unmodified)"):
        src, dst = os.path.join(th_dir, n), os.path.join(out_th, n)
        if os.path.exists(dst):
            continue
        if args.copy_thermal:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    print(f"prepared {len(rgb_names)} rgb / {len(th_names)} thermal -> {args.out}\n"
          f"train with:  --mode cyclegan --data-root {args.out}")


if __name__ == "__main__":
    main()

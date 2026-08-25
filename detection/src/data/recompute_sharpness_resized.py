"""
Recompute sharpness on already-extracted ground-truth frames, downsized to
match the training set's actual resolution, before concluding a domain
mismatch from a raw sharpness comparison. Laplacian-variance sharpness is
NOT scale-invariant: a native 640x512 frame will generically show higher
variance than the same scene resized to 512x512, independent of any real
difference in optical/processing sharpness. Run this first.

Usage:
    python3 recompute_sharpness_resized.py --frames_dir gt_audit_out/frames --target_size 512
"""

import argparse
import glob
import os

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--target_size", type=int, default=512,
                    help="Square size to resize to before computing sharpness (default: 512, matching the training set).")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not paths:
        print(f"No frames found in {args.frames_dir}")
        return

    sharp_native, sharp_resized = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sharp_native.append(cv2.Laplacian(img, cv2.CV_64F).var())
        resized = cv2.resize(img, (args.target_size, args.target_size), interpolation=cv2.INTER_AREA)
        sharp_resized.append(cv2.Laplacian(resized, cv2.CV_64F).var())

    print(f"Frames processed: {len(sharp_native)}")
    print(f"Native-resolution sharpness median: {np.median(sharp_native):.2f}  "
          f"(matches extract_gt_frames.py's earlier number)")
    print(f"Resized-to-{args.target_size}x{args.target_size} sharpness median: {np.median(sharp_resized):.2f}  "
          f"(apples-to-apples vs. the training set's 420.93)")
    print("\nIf the resized number is still far above 420.93, that's a real gap worth taking seriously.")
    print("If it drops close to 420.93, most of the earlier gap was the resolution confound, not a real domain mismatch.")


if __name__ == "__main__":
    main()
"""
Build a leak-free copy of a YOLO dataset by removing an entire recording
session that overlaps the evaluation video -- not just the frames the
near-duplicate check happened to confirm.

Why whole-session: in thermal-1-filtered, every confirmed leak (from
verify_leakage_ssim.py) belongs to the "<index>_png" filename family, and
that family's index maps linearly onto thermal.mp4 frame numbers
(frame ~= 5.0 * index + 14, i.e. the video sampled every 5th frame). The
unconfirmed members of the family are the neighbours of confirmed leaks,
5 frames apart -- removing only the confirmed ones would leave the leak in.

The script checks that claim before acting: every confirmed leak must match
--session_regex, and the index->frame fit must be tight (--max_resid). It
then writes the dataset (hard links, so no extra disk) plus leak_report.json
recording the fit and the last video frame the session covers -- frames
after that (plus a buffer) are a clean temporal holdout for the ORIGINAL
dataset's models.

Usage:
    python3 src/data/remove_session_leak.py \
        --dataset_dir data/thermal-1-filtered --output_dir data/thermal-1-noleak \
        --leakage_csv leakage_verified.csv --video data/videos/thermal.mp4
"""

import os
import re
import json
import glob
import shutil
import argparse

import numpy as np
import pandas as pd
import yaml

SPLITS = ["train", "valid", "test"]


def parse_args():
    p = argparse.ArgumentParser(description="Remove a leaked recording session from a YOLO dataset.")
    p.add_argument("--dataset_dir", required=True, help="Source dataset dir (with data.yaml + split subdirs).")
    p.add_argument("--output_dir", required=True, help="Where to write the leak-free dataset.")
    p.add_argument("--leakage_csv", required=True, help="verify_leakage_ssim.py output (confirmed_leak column).")
    p.add_argument("--session_regex", default=r"^(\d+)_png\.rf\.",
                   help="Filename regex identifying the leaked session; group 1 = source frame index.")
    p.add_argument("--video", default=None, help="Eval video, only used to report its frame count.")
    p.add_argument("--max_resid", type=float, default=30.0,
                   help="Abort if the index->frame linear fit misses any confirmed leak by more frames than this.")
    p.add_argument("--dry_run", action="store_true", help="Report what would be removed, write nothing.")
    return p.parse_args()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def label_path(dataset_dir, split, image_name):
    return os.path.join(dataset_dir, split, "labels", os.path.splitext(image_name)[0] + ".txt")


def main():
    args = parse_args()
    session_re = re.compile(args.session_regex)

    # ---- evidence: all confirmed leaks in the session, tight index->frame fit
    leaks = pd.read_csv(args.leakage_csv)
    leaks = leaks[leaks["confirmed_leak"].astype(str) == "True"]
    per_image = leaks.groupby("path_1")["frame_id"].median()
    names = [os.path.basename(p) for p in per_image.index]
    outside = [n for n in names if not session_re.match(n)]
    if outside:
        raise SystemExit(f"{len(outside)} confirmed leaks do not match --session_regex, e.g. {outside[:3]} "
                         "-- the leak is not a single session; refusing to proceed.")
    idx = np.array([int(session_re.match(n).group(1)) for n in names])
    frames = per_image.values.astype(float)
    slope, intercept = np.polyfit(idx, frames, 1)
    resid = np.abs(frames - (slope * idx + intercept))
    print(f"Confirmed leaks: {len(names)} images, all in session. "
          f"frame = {slope:.3f} * index + {intercept:.1f} (max resid {resid.max():.1f}, median {np.median(resid):.1f})")
    if resid.max() > args.max_resid:
        raise SystemExit(f"Fit residual {resid.max():.1f} > --max_resid {args.max_resid}; mapping not linear, check by hand.")

    # ---- plan
    plan, removed = {}, {}
    for split in SPLITS:
        images = sorted(os.listdir(os.path.join(args.dataset_dir, split, "images")))
        drop = [n for n in images if session_re.match(n)]
        plan[split] = [n for n in images if not session_re.match(n)]
        removed[split] = drop
        print(f"  {split:<6} {len(images):>5} images -> keep {len(plan[split]):>5}, remove {len(drop):>5}")

    session_idx = np.array([int(session_re.match(n).group(1)) for s in SPLITS for n in removed[s]])
    first_frame = int(round(slope * session_idx.min() + intercept))
    last_frame = int(round(slope * session_idx.max() + intercept))
    n_video = None
    if args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    print(f"Removed session covers video frames ~{first_frame}..{last_frame}"
          + (f" of {n_video}" if n_video else "") + f" (confirmed leaks up to frame {int(frames.max())}).")

    if args.dry_run:
        return

    # ---- write
    if os.path.exists(args.output_dir):
        raise SystemExit(f"{args.output_dir} already exists; remove it first.")
    for split in SPLITS:
        os.makedirs(os.path.join(args.output_dir, split, "images"))
        os.makedirs(os.path.join(args.output_dir, split, "labels"))
        for name in plan[split]:
            link_or_copy(os.path.join(args.dataset_dir, split, "images", name),
                         os.path.join(args.output_dir, split, "images", name))
            lbl = label_path(args.dataset_dir, split, name)
            if os.path.exists(lbl):
                link_or_copy(lbl, label_path(args.output_dir, split, name))

    with open(os.path.join(args.dataset_dir, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.output_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    report = {
        "source_dataset": args.dataset_dir,
        "session_regex": args.session_regex,
        "n_confirmed_leak_images": len(names),
        "index_to_frame_fit": {"slope": slope, "intercept": intercept,
                               "max_resid": float(resid.max()), "median_resid": float(np.median(resid))},
        "removed_session_video_frames": [first_frame, last_frame],
        "video": args.video, "video_frame_count": n_video,
        "counts": {s: {"kept": len(plan[s]), "removed": len(removed[s])} for s in SPLITS},
        "removed_images": removed,
    }
    with open(os.path.join(args.output_dir, "leak_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.output_dir} (+ leak_report.json).")


if __name__ == "__main__":
    main()

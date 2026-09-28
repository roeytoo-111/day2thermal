"""
Check for train/test leakage by hashing images two ways:
  - SHA256: catches literal byte-identical duplicate files (cheap, exact).
  - Perceptual hash (pHash): catches the SAME underlying frame even after
    resizing / re-encoding / mild compression -- this is the check that
    actually matters here, since training images went through Roboflow's
    export pipeline (resized from the camera's native 640x512 down to
    512x512) and video frames go through ffmpeg's own decode/encode path.
    Two images can show the identical real-world instant and still be
    completely different byte-for-byte -- SHA256 alone would silently miss
    that. Compared via Hamming distance (near-match), not exact equality.

Two modes, composable -- hash once, compare many times:

  hash mode: compute + save hashes for a directory of images (a training
  folder, or a directory of frames already extracted from a video via
  extract_gt_frames.py --extract_all).

  compare mode: cross-reference two precomputed hash CSVs, report both
  exact SHA256 matches and near-duplicate pHash matches.

Usage:
    # Step 1: hash the training folder
    python3 check_leakage.py hash --dir data/thermal_1_filtered --out train_hashes.csv

    # Step 2: extract ALL frames from the eval video (reuses extract_gt_frames.py),
    # then hash them
    python3 extract_gt_frames.py --video_path data/videos/thermal.mp4 \
        --output_dir leak_check_frames --extract_all
    python3 check_leakage.py hash --dir leak_check_frames/frames --out video_hashes.csv

    # Step 3: compare
    python3 check_leakage.py compare --hashes1 train_hashes.csv --hashes2 video_hashes.csv \
        --phash_threshold 8 --out leakage_matches.csv
"""

import os
import glob
import hashlib
import argparse

import pandas as pd
from PIL import Image
import imagehash

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(image_dir, recursive=True):
    pattern = "**/*" if recursive else "*"
    paths = [p for p in glob.glob(os.path.join(image_dir, pattern), recursive=recursive)
             if p.lower().endswith(IMG_EXTS)]
    print(f"Found {len(paths)} images under {image_dir}")

    rows = []
    for i, p in enumerate(paths):
        try:
            sha = sha256_of_file(p)
            with Image.open(p) as im:
                phash = str(imagehash.phash(im))
        except Exception as e:
            print(f"  skipping {p}: {e}")
            continue
        rows.append({"path": p, "sha256": sha, "phash": phash})
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(paths)}...")

    return pd.DataFrame(rows)


def cmd_hash(args):
    df = hash_directory(args.dir, recursive=not args.no_recursive)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} hashes to {args.out}")


def cmd_compare(args):
    h1 = pd.read_csv(args.hashes1)
    h2 = pd.read_csv(args.hashes2)
    print(f"Comparing {len(h1)} images (set 1) against {len(h2)} images (set 2)")

    # Exact SHA256 matches -- literal duplicate files.
    exact = h1.merge(h2, on="sha256", suffixes=("_1", "_2"))
    print(f"\nExact SHA256 matches (byte-identical files): {len(exact)}")
    if not exact.empty:
        print(exact[["path_1", "path_2"]].head(20).to_string(index=False))

    # Near-duplicate pHash matches -- same underlying scene, different
    # resize/encoding. This is the check that matters most here.
    print(f"\nComputing pHash Hamming distances (threshold <= {args.phash_threshold})...")
    h1_hashes = [imagehash.hex_to_hash(h) for h in h1["phash"]]
    h2_hashes = [imagehash.hex_to_hash(h) for h in h2["phash"]]

    near_matches = []
    for i, (p1, hash1) in enumerate(zip(h1["path"], h1_hashes)):
        for p2, hash2 in zip(h2["path"], h2_hashes):
            dist = hash1 - hash2  # Hamming distance, imagehash overloads '-' for this
            if dist <= args.phash_threshold:
                near_matches.append({"path_1": p1, "path_2": p2, "phash_distance": dist})
        if (i + 1) % 1000 == 0:
            print(f"  checked {i + 1}/{len(h1)} against full set 2...")

    near_df = pd.DataFrame(near_matches).sort_values("phash_distance") if near_matches else pd.DataFrame(
        columns=["path_1", "path_2", "phash_distance"])
    print(f"\nNear-duplicate pHash matches (distance <= {args.phash_threshold}): {len(near_df)}")
    if not near_df.empty:
        print(near_df.head(30).to_string(index=False))
        print("\nManually eyeball a few of these pairs before concluding real leakage --")
        print("very similar-looking but genuinely distinct frames (e.g. static sky/horizon)")
        print("can land at low Hamming distance without being the same instant.")

    combined = pd.concat([
        exact[["path_1", "path_2"]].assign(match_type="exact_sha256", phash_distance=0),
        near_df.assign(match_type="near_duplicate_phash"),
    ], ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"\nSaved {len(combined)} total flagged pairs to {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Check for train/test leakage via SHA256 + perceptual hashing.")
    sub = p.add_subparsers(dest="mode", required=True)

    p_hash = sub.add_parser("hash", help="Compute and save hashes for a directory of images.")
    p_hash.add_argument("--dir", required=True)
    p_hash.add_argument("--out", required=True)
    p_hash.add_argument("--no-recursive", action="store_true", dest="no_recursive")

    p_cmp = sub.add_parser("compare", help="Compare two precomputed hash CSVs.")
    p_cmp.add_argument("--hashes1", required=True)
    p_cmp.add_argument("--hashes2", required=True)
    p_cmp.add_argument("--phash_threshold", type=int, default=8,
                        help="Max Hamming distance (out of 64 bits) to flag as a near-duplicate (default: 8). "
                             "Lower = stricter/fewer false flags, higher = more permissive/more false flags. "
                             "Worth eyeballing actual flagged pairs to calibrate for your footage.")
    p_cmp.add_argument("--out", required=True)

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "hash":
        cmd_hash(args)
    elif args.mode == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
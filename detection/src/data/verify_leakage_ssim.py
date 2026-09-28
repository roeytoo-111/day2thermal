"""
Refine raw pHash leakage matches into a trustworthy verdict, since pHash
alone is known to over-flag on low-texture content (smooth sky/cloud/horizon
frames have little of the mid/high-frequency detail pHash actually
discriminates on). Two additions:

  1. Temporal clustering: for each training image, are its matched video
     frame_ids tightly clustered in time, or scattered across the whole
     video? A wide scatter across thousands of unrelated frames is the
     signature of coincidental low-texture collision, not real overlap.

  2. SSIM secondary confirmation: pHash is deliberately low-frequency
     (that's what makes it survive resize/recompression). SSIM is sensitive
     to exactly the fine structural detail pHash ignores -- a genuine
     near-duplicate frame should score very high; a coincidentally
     similar-looking but genuinely distinct sky frame should score
     noticeably lower once fine detail is actually compared.

Usage:
    python3 verify_leakage_ssim.py --matches leakage_matches.csv --out leakage_verified.csv
"""

import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

FRAME_ID_RE = re.compile(r"frame_(\d+)")


def extract_frame_id(path):
    m = FRAME_ID_RE.search(path)
    return int(m.group(1)) if m else None


def compute_ssim(path_a, path_b, size=256, grid=8):
    """Resize both to a common size (removes the resolution confound between
    a 512x512 training export and a native-resolution extracted frame) and
    compare in grayscale. Returns (global_ssim, worst_block_ssim) -- global
    SSIM is a whole-frame average, which on sky-dominated content can stay
    high even when a small target sits in a different position (diluted by
    all the agreeing sky pixels). worst_block_ssim splits the frame into a
    grid and reports the MINIMUM block score -- much more sensitive to a
    localized difference, which is exactly the failure mode that matters
    here."""
    try:
        a = np.asarray(Image.open(path_a).convert("L").resize((size, size))).astype(np.float64)
        b = np.asarray(Image.open(path_b).convert("L").resize((size, size))).astype(np.float64)
        global_score, ssim_map = ssim(a, b, full=True, data_range=255.0)

        bh, bw = size // grid, size // grid
        block_scores = []
        for by in range(grid):
            for bx in range(grid):
                block = ssim_map[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                if block.size:
                    block_scores.append(block.mean())
        worst_block = min(block_scores) if block_scores else global_score
        return float(global_score), float(worst_block)
    except Exception:
        return None, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", required=True, help="leakage_matches.csv from check_leakage.py compare.")
    p.add_argument("--out", required=True)
    p.add_argument("--ssim_threshold", type=float, default=0.85,
                    help="Worst-block SSIM above this = confirmed likely leak (default: 0.85, calibrated "
                         "against a genuine-vs-coincidental test case -- 0.92 vs 0.84 in that test). Treat "
                         "this as a starting point, not gospel: sort your real results by ssim_worst_block "
                         "and look for a natural gap/cliff in the distribution rather than trusting a fixed "
                         "cutoff blindly -- the right value depends on your footage's actual noise character.")
    p.add_argument("--max_pairs", type=int, default=None,
                    help="Cap the number of pairs to SSIM-check, if the full set is too slow (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.matches)
    if args.max_pairs:
        df = df.head(args.max_pairs)
    print(f"Computing SSIM for {len(df)} flagged pairs...")

    ssim_global, ssim_worst_block = [], []
    for i, row in df.iterrows():
        g, w = compute_ssim(row["path_1"], row["path_2"])
        ssim_global.append(g)
        ssim_worst_block.append(w)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(df)}...")

    df["ssim_global"] = ssim_global
    df["ssim_worst_block"] = ssim_worst_block
    df["frame_id"] = df["path_2"].apply(extract_frame_id)
    df["confirmed_leak"] = df["ssim_worst_block"] >= args.ssim_threshold

    n_confirmed = df["confirmed_leak"].sum()
    print(f"\n{n_confirmed}/{len(df)} pairs confirmed as likely real leaks (worst-block SSIM >= {args.ssim_threshold}).")
    print(f"{len(df) - n_confirmed} were pHash matches but SSIM disagrees -- likely coincidental low-texture collision.")

    # Temporal clustering per training image, CONFIRMED matches only.
    print("\n" + "=" * 70)
    print("TEMPORAL CLUSTERING (confirmed leaks only)")
    print("=" * 70)
    confirmed = df[df["confirmed_leak"]]
    clusters = defaultdict(list)
    for _, row in confirmed.iterrows():
        if row["frame_id"] is not None:
            clusters[row["path_1"]].append(row["frame_id"])

    summary_rows = []
    for train_img, frame_ids in clusters.items():
        frame_ids = sorted(frame_ids)
        summary_rows.append({
            "train_image": train_img,
            "n_matches": len(frame_ids),
            "min_frame": frame_ids[0],
            "max_frame": frame_ids[-1],
            "spread": frame_ids[-1] - frame_ids[0],
        })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("n_matches", ascending=False)
        print(summary_df.head(20).to_string(index=False))
        wide_spread = summary_df[summary_df["spread"] > 5000]
        if not wide_spread.empty:
            print(f"\n⚠ {len(wide_spread)} training images have confirmed matches spread over >5000 frames --")
            print("  worth a manual look; could be a genuinely long static video segment, or SSIM threshold too loose.")
    else:
        print("No confirmed leaks after SSIM filtering.")

    df.to_csv(args.out, index=False)
    print(f"\nSaved full results (with ssim, frame_id, confirmed_leak columns) to {args.out}")
    if not summary_df.empty:
        summary_path = args.out.replace(".csv", "_per_train_image_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Per-training-image summary saved to {summary_path}")


if __name__ == "__main__":
    main()
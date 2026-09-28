import os
import argparse
import pandas as pd
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Verify pHash leakage matches and export side-by-side visual pairs.")
    p.add_argument("--csv", default="leakage_matches.csv", help="Path to leakage_matches.csv output.")
    p.add_argument("--out_dir", default="leakage_vis", help="Output folder to save visual comparisons.")
    p.add_argument("--max_vis", type=int, default=20, help="Max number of distance-0 visual pairs to render.")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Matches file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print("=== LEAKAGE SUMMARY REPORT ===")
    print(f"Total flagged pairs (dist <= 8): {len(df)}")
    
    # Filter by exact pHash match
    exact_phash = df[df["phash_distance"] == 0]
    unique_train_exact = exact_phash["path_1"].nunique()
    unique_video_exact = exact_phash["path_2"].nunique()
    
    print(f"\nExact pHash matches (distance == 0): {len(exact_phash)} pairs")
    print(f"  Unique training images involved: {unique_train_exact}")
    print(f"  Unique video frames matched:     {unique_video_exact}")

    # Display top leaked training images
    print("\nTop 10 Training Images with Most Match Hits:")
    top_train = df["path_1"].value_counts().head(10)
    for img_path, count in top_train.items():
        min_dist = df[df["path_1"] == img_path]["phash_distance"].min()
        print(f"  {os.path.basename(img_path)} -> {count} matching frames (Min Dist: {min_dist})")

    # Render side-by-side verification images for distance 0
    if len(exact_phash) > 0 and args.max_vis > 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"\nSaving up to {args.max_vis} side-by-side comparison images to '{args.out_dir}/'...")
        
        sample_pairs = exact_phash.head(args.max_vis)
        for idx, row in sample_pairs.iterrows():
            img1_path = row["path_1"]
            img2_path = row["path_2"]

            im1 = cv2.imread(img1_path)
            im2 = cv2.imread(img2_path)

            if im1 is None or im2 is None:
                continue

            # Resize to match height for visualization
            h = min(im1.shape[0], im2.shape[0])
            w1 = int(im1.shape[1] * (h / im1.shape[0]))
            w2 = int(im2.shape[1] * (h / im2.shape[0]))
            
            im1_res = cv2.resize(im1, (w1, h))
            im2_res = cv2.resize(im2, (w2, h))

            # Add labels
            cv2.putText(im1_res, "Train Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(im2_res, f"Video Frame (Dist: {row['phash_distance']})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            canvas = np.hstack([im1_res, im2_res])
            out_name = f"match_{idx:04d}_dist{row['phash_distance']}.jpg"
            cv2.imwrite(os.path.join(args.out_dir, out_name), canvas)

        print(f"Visualizations written to {args.out_dir}/")

if __name__ == "__main__":
    main()
"""
label_review.py

Fast keypress labeling of candidate crops produced by pipeline_audit.py /
detect_only.py / jetson_benchmark.py.

Controls:
  't' -> True Positive (real UAV, box is a reasonably tight fit)
  'l' -> True Positive but box is LOOSE/degenerate (drone is present in frame,
         but the box is way oversized / doesn't tightly localize it)
  'f' -> False Positive (generic -- not a UAV)
  'h' -> False Positive, SHADOW specifically (tracked separately for analysis)
  'g' -> False Positive, GROUND TRAIL specifically (tracked separately)
  's' -> skip / unsure
  'q' -> quit and save progress

Ground-truth labeling guide:
  - If the crop contains a REAL DRONE: ALWAYS press 't' or 'l', regardless of whether 
    gate_status says 'confirmed' or 'rejected'.
    - 'confirmed' + 't'/'l' = Stage 2 True Positive (TP)
    - 'rejected'  + 't'/'l' = Stage 2 False Negative (FN)
  - If the crop DOES NOT contain a drone: ALWAYS press 'f', 'h', or 'g'.
    - 'confirmed' + 'f'/'h'/'g' = Stage 2 False Positive (FP)
    - 'rejected'  + 'f'/'h'/'g' = Stage 2 True Negative (TN)
"""

import os
import csv
import shutil
import argparse
from collections import defaultdict

import cv2

DISPLAY_SIZE = 480

VALID_LABELS = ("TP", "TP_loose", "FP", "FP_shadow", "FP_ground_trail")


def letterbox(img, size=DISPLAY_SIZE, bg=(40, 40, 40)):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(
        resized,
        (size - nh) // 2, size - nh - (size - nh) // 2,
        (size - nw) // 2, size - nw - (size - nw) // 2,
        cv2.BORDER_CONSTANT, value=bg,
    )
    return canvas


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_manifest(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def review(audit_dir, manifest_path=None):
    manifest_path = manifest_path or os.path.join(audit_dir, "manifest.csv")
    rows = load_manifest(manifest_path)
    fieldnames = list(rows[0].keys()) if rows else [
        "filename", "frame_id", "conf", "bbox", "gate_status", "label"]

    tp_dir = os.path.join(audit_dir, "tp")
    tp_loose_dir = os.path.join(audit_dir, "tp_loose")
    fp_dir = os.path.join(audit_dir, "fp")
    os.makedirs(tp_dir, exist_ok=True)
    os.makedirs(tp_loose_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    unlabeled = [r for r in rows if not r.get("label")]
    print(f"{len(rows)} total crops, {len(unlabeled)} unlabeled.")
    print("Controls: t=TP  l=TP-loose-box  f=FP  h=FP-shadow  g=FP-ground-trail  "
          "s=skip  q=quit+save\n")

    for i, row in enumerate(unlabeled):
        # 1. Resolve path first before attempting cv2.imread
        img_path = os.path.join(audit_dir, row["filename"])
        if not os.path.exists(img_path) and row.get("video"):
            candidate = os.path.join(audit_dir, row["video"], row["filename"])
            if os.path.exists(candidate):
                img_path = candidate
            else:
                for sub in os.listdir(audit_dir):
                    sub_path = os.path.join(audit_dir, sub)
                    if os.path.isdir(sub_path) and row["video"] in sub:
                        check_path = os.path.join(sub_path, row["filename"])
                        if os.path.exists(check_path):
                            img_path = check_path
                            break

        img = cv2.imread(img_path)
        if img is None:
            continue

        disp = letterbox(img)
        label_text = f"[{i+1}/{len(unlabeled)}] conf={row['conf']} gate={row['gate_status']}"
        cv2.putText(disp, label_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        cv2.imshow("Candidate Review", disp)
        key = cv2.waitKey(0) & 0xFF

        base_name = os.path.basename(row["filename"])
        if key == ord('t'):
            row["label"] = "TP"
            shutil.copy(img_path, os.path.join(tp_dir, base_name))
        elif key == ord('l'):
            row["label"] = "TP_loose"
            shutil.copy(img_path, os.path.join(tp_loose_dir, base_name))
        elif key == ord('f'):
            row["label"] = "FP"
            shutil.copy(img_path, os.path.join(fp_dir, base_name))
        elif key == ord('h'):
            row["label"] = "FP_shadow"
            shutil.copy(img_path, os.path.join(fp_dir, base_name))
        elif key == ord('g'):
            row["label"] = "FP_ground_trail"
            shutil.copy(img_path, os.path.join(fp_dir, base_name))
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    save_manifest(manifest_path, rows, fieldnames)

    # -- Summary --
    print("\n" + "=" * 50)
    print("FAILURE-MODE SUMMARY")
    print("=" * 50)

    by_gate = defaultdict(lambda: {k: 0 for k in VALID_LABELS})
    conf_buckets = defaultdict(lambda: {k: 0 for k in VALID_LABELS})
    label_totals = defaultdict(int)

    for r in rows:
        if r["label"] not in VALID_LABELS:
            continue
        label_totals[r["label"]] += 1
        by_gate[r["gate_status"]][r["label"]] += 1
        try:
            c = float(r["conf"])
        except ValueError:
            continue
        bucket = f"{int(c*10)/10:.1f}-{int(c*10)/10 + 0.1:.1f}"
        conf_buckets[bucket][r["label"]] += 1

    def fmt_row(counts):
        total = sum(counts.values())
        prec = (counts["TP"] + counts["TP_loose"]) / total if total else 0
        return (f"TP={counts['TP']:4d}  TP_loose={counts['TP_loose']:4d}  "
                f"FP={counts['FP']:4d}  FP_shadow={counts['FP_shadow']:4d}  "
                f"FP_ground_trail={counts['FP_ground_trail']:4d}  presence_rate={prec:.2%}")

    print("\nBy gate_status:")
    for gate, counts in sorted(by_gate.items()):
        print(f"  {gate:25s} {fmt_row(counts)}")

    print("\nBy confidence bucket:")
    for bucket, counts in sorted(conf_buckets.items()):
        print(f"  conf {bucket}  {fmt_row(counts)}")

    # Stage 2 Gate Performance Evaluation
    s2_confirmed = by_gate["confirmed"]
    s2_rejected = by_gate["rejected"]
    
    s2_tp = s2_confirmed["TP"] + s2_confirmed["TP_loose"]
    s2_fp = s2_confirmed["FP"] + s2_confirmed["FP_shadow"] + s2_confirmed["FP_ground_trail"]
    s2_fn = s2_rejected["TP"] + s2_rejected["TP_loose"]
    s2_tn = s2_rejected["FP"] + s2_rejected["FP_shadow"] + s2_rejected["FP_ground_trail"]
    
    print("\n" + "-" * 50)
    print("STAGE 2 CLASSIFIER / GATE ACCURACY")
    print("-" * 50)
    print(f"  Stage-2 True Positives  (Confirmed Drone)  : {s2_tp}")
    print(f"  Stage-2 False Positives (Confirmed Non-Drone): {s2_fp}")
    print(f"  Stage-2 False Negatives (Rejected Drone)   : {s2_fn}")
    print(f"  Stage-2 True Negatives  (Rejected Non-Drone): {s2_tn}")
    if (s2_tp + s2_fp) > 0:
        print(f"  Stage 2 Precision : {s2_tp / (s2_tp + s2_fp):.2%}")
    if (s2_tp + s2_fn) > 0:
        print(f"  Stage 2 Recall    : {s2_tp / (s2_tp + s2_fn):.2%}")

    n_fp_total = (label_totals["FP"] + label_totals["FP_shadow"]
                  + label_totals["FP_ground_trail"])
    if n_fp_total:
        shadow_frac = label_totals["FP_shadow"] / n_fp_total
        trail_frac = label_totals["FP_ground_trail"] / n_fp_total
        print(f"\nShadow FPs are {shadow_frac:.1%} of all false positives "
              f"({label_totals['FP_shadow']}/{n_fp_total}).")
        print(f"Ground-trail FPs are {trail_frac:.1%} of all false positives "
              f"({label_totals['FP_ground_trail']}/{n_fp_total}).")

    n_tp_total = label_totals["TP"] + label_totals["TP_loose"]
    if n_tp_total:
        loose_frac = label_totals["TP_loose"] / n_tp_total
        print(f"Loose/degenerate boxes are {loose_frac:.1%} of all true detections "
              f"({label_totals['TP_loose']}/{n_tp_total}).")

    print(f"\nLabeled crops copied to:\n  {tp_dir}\n  {fp_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--audit_dir', required=True)
    p.add_argument('--manifest', default=None)
    args = p.parse_args()
    review(args.audit_dir, args.manifest)
r"""
Ensemble pseudo-labels for distillation into one model (the ensemble itself is too slow for the Jetson).

Teacher = several thermal models fused with ensemble_dets.fuse (weighted boxes; score = sum of each model's best
confidence / K). Runs over UNLABELLED frames of TRAINING sessions only (never 08_58_15 / 10_15_59 / 10_05_11 /
07-02 / 07-30) and writes a YOLO set (uav = 0):
  * fused box >= --hi            -> uav box
  * frame with every fused score < --lo  -> empty image (negative)
  * anything in between (a box in [lo, hi)) -> frame skipped: the teacher is unsure there
Frames are sampled every --step frames, at most --max-per-video per video.

    python3 src/data/make_pseudo_labels.py --models a.pt b.pt c.pt --videos ../data/boson_work_thermal/*_thermal.mp4 \
        ../data/new_videos/clipped/2026-06-23_scenario2_thermal_clipped.mp4 --out data/boson_work/pseudo_ensA
"""
import os
import sys
import json
import argparse

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from ensemble_dets import fuse   # noqa: E402

EXCLUDE = ("08_58_15", "10_15_59", "10_05_11", "07-02", "0702", "07-30", "07_30", "th_output")
EXCLUDE_EXACT = ("thermal.mp4", "day.mp4")   # the 07-02 test video; matched exactly (Boson files end in _thermal.mp4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--max-per-video", type=int, default=600); ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.15); ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--max-t", default=None, help="video:seconds limits, e.g. 2026-07-15:380 (train part only)")
    a = ap.parse_args()
    from ultralytics import YOLO
    models = [YOLO(p) for p in a.models]
    ids = [{int(k) for k, v in m.names.items() if "uav" in str(v).lower()} for m in models]
    limits = dict(x.split(":") for x in a.max_t.split(",")) if a.max_t else {}
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    stats = {}
    for v in a.videos:
        name = os.path.basename(v).replace("_thermal", "").rsplit(".", 1)[0]
        if any(x in os.path.basename(v) for x in EXCLUDE) or os.path.basename(v) in EXCLUDE_EXACT:
            print("skip (held out / test):", v); continue
        cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tmax = next((float(s) for k, s in limits.items() if k in v), None)
        k, n_pos, n_neg, n_skip, written = -1, 0, 0, 0, 0
        while written < a.max_per_video:
            ok, im = cap.read(); k += 1
            if not ok or (tmax is not None and k / fps > tmax):
                break
            if k % a.step:
                continue
            per = []
            for m, idset in zip(models, ids):
                r = m.predict(im, imgsz=a.imgsz, conf=0.05, verbose=False)[0]
                per.append([b + [s] for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(c) in idset])
            fused = fuse(per, len(models), 0.3, 6.0)
            pos = [b for b in fused if b[4] >= a.hi]
            unsure = [b for b in fused if a.lo <= b[4] < a.hi]
            if unsure:
                n_skip += 1; continue
            H, W = im.shape[:2]; stem = f"pl_{name}_{k:06d}"
            cv2.imwrite(os.path.join(a.out, "images", stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(os.path.join(a.out, "labels", stem + ".txt"), "w").write("".join(
                f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n" for b in pos))
            n_pos += bool(pos); n_neg += not pos; written += 1
        stats[name] = {"positive_frames": n_pos, "empty_frames": n_neg, "skipped_unsure": n_skip}
        print(name, stats[name], flush=True)
    json.dump({"args": vars(a), "stats": stats}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

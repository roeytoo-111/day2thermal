r"""
Mine THERMAL hard examples where the current models fire on something that is not the drone.

A. New Boson frames (training sessions only). Frames whose label status is `pos` (the drone was confirmed by the day
   AND the thermal model when the labels were built; box in the label CSV) and that are not already in the training
   set. Every --step-th such frame is run through the given models; if a uav detection >= --conf lies farther than
   max(--min-dist, 2 x drone size) from the confirmed drone, the frame is a hard example: it is written with the
   DRONE's box (uav = 0), so the false-alarm area becomes negative in context (one drone per flight, the same rule
   as mine_rgb_fp.py). Held-out sessions and 10_05_11 (misregistered) are excluded.
B. Existing REAL training images (not pastes): a uav detection >= --conf that overlaps no label (IoU < 0.1) and is
   not near a uav label -> the image is listed for duplication (hard-example oversampling).

Outputs in --out: boson/{images,labels} (A), hard_existing.txt (B, one image file name per line), summary.json.

    python3 src/data/mine_thermal_hard.py --labels-dir data/boson_work/labels --videos ../data/boson_work_thermal \
        --train data/thermal-5-bosonP-u0 --models ../runs/thermal_uav/bosonM/weights/best.pt \
        ../runs/thermal_uav/bosonM0u_s1/weights/best.pt --out data/boson_work/hardneg
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np
import pandas as pd

PASTE_PREFIXES = ("bgpaste_", "bgboson_")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def uav_ids(model):
    return {int(k) for k, v in model.names.items() if "uav" in str(v).lower()}


def detect(models, ims, conf):
    out = [[] for _ in ims]
    for m in models:
        ids = uav_ids(m)
        for i, r in enumerate(m.predict(ims, imgsz=512, conf=conf, verbose=False)):
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(c) in ids:
                    out[i].append(b + [s])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True); ap.add_argument("--videos", required=True)
    ap.add_argument("--train", required=True); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="08_58_15,10_15_59,10_05_11")
    ap.add_argument("--conf", type=float, default=0.35); ap.add_argument("--min-dist", type=float, default=40)
    ap.add_argument("--step", type=int, default=3); ap.add_argument("--min-gap", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    from ultralytics import YOLO
    models = [YOLO(p) for p in a.models]
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out, "boson", sub), exist_ok=True)
    have = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(a.train, "train", "images", "*"))}
    excl = set(a.exclude.split(","))
    summary = {"A": {}, "B": {}}
    # ---------------- A: new Boson frames
    for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))):
        s = os.path.basename(f)[:-4]
        if s in excl:
            continue
        L = pd.read_csv(f)
        P = L[(L.status == "pos") & L.x0.notna()].drop_duplicates("frame")
        P = P[~P.frame.apply(lambda k: f"b{s}_{int(k):06d}" in have)].iloc[::a.step]
        want = {int(r.frame): (r.x0, r.y0, r.x1, r.y1) for r in P.itertuples()}
        if not want:
            continue
        cap = cv2.VideoCapture(os.path.join(a.videos, f"{s}_thermal.mp4"))
        buf, k, kept, last_kept, fps_seen = [], -1, 0, -10 ** 9, 0
        def flush():
            nonlocal kept, last_kept, fps_seen
            dets = detect(models, [im for _, im in buf], 0.25)
            for (fi, im), D in zip(buf, dets):
                x0, y0, x1, y1 = want[fi]; dc = ((x0 + x1) / 2, (y0 + y1) / 2); size = max(x1 - x0, y1 - y0)
                far = [d for d in D if d[4] >= a.conf and
                       np.hypot((d[0] + d[2]) / 2 - dc[0], (d[1] + d[3]) / 2 - dc[1]) > max(a.min_dist, 2 * size)
                       and iou(d, (x0, y0, x1, y1)) < 0.05]
                if far:
                    fps_seen += 1
                    if fi - last_kept >= a.min_gap:
                        H, W = im.shape[:2]; stem = f"hn_b{s}_{fi:06d}"
                        cv2.imwrite(os.path.join(a.out, "boson", "images", stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        open(os.path.join(a.out, "boson", "labels", stem + ".txt"), "w").write(
                            f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
                        kept += 1; last_kept = fi
            buf.clear()
        last = max(want)
        while k < last:
            ok, im = cap.read(); k += 1
            if not ok:
                break
            if k in want:
                buf.append((k, im))
                if len(buf) >= a.batch:
                    flush()
        if buf:
            flush()
        summary["A"][s] = {"candidates": len(want), "with_far_detection": fps_seen, "kept": kept}
        print(f"A {s}: {len(want)} candidate frames, {fps_seen} with a far uav detection >= {a.conf}, kept {kept}", flush=True)
    # ---------------- B: existing real training images
    imgs = [f for f in sorted(glob.glob(os.path.join(a.train, "train", "images", "*")))
            if not os.path.basename(f).startswith(PASTE_PREFIXES)]
    hard = []
    for i in range(0, len(imgs), a.batch):
        chunk = imgs[i:i + a.batch]; ims = [cv2.imread(p) for p in chunk]
        for p, im, D in zip(chunk, ims, detect(models, ims, 0.25)):
            H, W = im.shape[:2]
            lab = []
            lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
            if os.path.exists(lp):
                for l in open(lp):
                    q = l.split()
                    if len(q) >= 5:
                        c, cx, cy, w, h = int(q[0]), *map(float, q[1:5])
                        lab.append((c, ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
            bad = [d for d in D if d[4] >= a.conf and all(iou(d, b) < 0.1 for _, b in lab) and
                   all(np.hypot((d[0] + d[2] - b[0] - b[2]) / 2, (d[1] + d[3] - b[1] - b[3]) / 2) > max(20, 1.5 * max(b[2] - b[0], b[3] - b[1]))
                       for c, b in lab if c == 0)]
            if bad:
                hard.append(os.path.basename(p))
        if i % (a.batch * 40) == 0:
            print(f"B {i}/{len(imgs)} real training images, {len(hard)} hard so far", flush=True)
    open(os.path.join(a.out, "hard_existing.txt"), "w").write("\n".join(hard) + "\n")
    fam = pd.Series([h.split("_")[0] if not h[0].isdigit() else "num" for h in hard]).value_counts().head(8).to_dict()
    summary["B"] = {"real_train_images": len(imgs), "hard": len(hard), "families": fam}
    summary["args"] = vars(a)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"B: {len(hard)}/{len(imgs)} real training images have an off-label uav detection >= {a.conf}; {fam}")
    print(f"A total kept: {sum(v['kept'] for v in summary['A'].values())}")


if __name__ == "__main__":
    main()

r"""
Hard negatives for the RGB detector's cloud / sky false alarms, mined from Open Images (images CC BY 2.0, labels
CC BY 4.0: commercial use allowed with attribution).

Candidates: images with a human-verified positive label Cloud or Sky and NO positive label for anything that flies
(aircraft, airliner, airplane, fixed-wing, jet, glider, helicopter, drone, bird, insect, butterfly, kite, balloon,
hot-air balloon, parachute). Of 18.8k such val+test images, 15,649 qualify (all CC BY 2.0).
Only sky-dominated images are used (>= --min-smooth smooth thumbnail blocks, or the Cloud label): "Sky" also tags car
and boat close-ups. The detector runs on each whole image (imgsz 640); every SMALL uav / airplane box (long side <=
--max-box-frac of the image) >= --conf is a false alarm candidate. A
640 tile is cut around it at native resolution (the scale the tiled 4K pipeline sees; padded grey if the image is
smaller) and queued in <out>/review/tiles.csv, the format of review_rgb_tiles.py (d = there IS a drone after all,
f = false alarm -> negative, b = bird). Unlabelled flying things are rare but possible, hence the review.
attribution.csv lists ImageID, author, license and source URL of every tile (CC BY requires attribution).

    python3 src/data/mine_open_images_negatives.py --images ../data/openimages/images \
        --meta ../data/openimages/validation-images-with-rotation.csv ../data/openimages/test-images-with-rotation.csv \
        --model ../../ANN-Detection/models/yolo26s_rgb_20261012/yolo26s_fw4_20261011.pt --out ../data/oi_negatives
"""
import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd

TILE = 640


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True); ap.add_argument("--meta", nargs="+", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.10); ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-smooth", type=float, default=0.5, help="share of smooth 8x8 thumbnail blocks (sky) an image "
                    "needs, unless it carries the Cloud label")
    ap.add_argument("--max-box-frac", type=float, default=0.3, help="false box long side <= this x image long side "
                    "(the 'Sky' label also covers car / boat close-ups that fw4 calls airplane: not our scene)")
    ap.add_argument("--labels", nargs="*", default=[], help="Open Images human image-label csvs (to read the Cloud label)")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    alarm = {k for k, v in model.names.items() if v in ("uav", "airplane")}
    meta = pd.concat([pd.read_csv(m) for m in a.meta]).set_index("ImageID")
    tdir = os.path.join(a.out, "review", "tiles"); os.makedirs(tdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    cloud = set()
    for lp in a.labels:
        L = pd.read_csv(lp); cloud |= set(L[(L.Confidence == 1) & (L.LabelName == "/m/0csby")].ImageID)

    def smooth_frac(im):
        th = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        return float((th.reshape(8, 8, 8, 8).std(axis=(1, 3)) < 6).mean())
    rows, attr, n_img_fp, stats = [], [], 0, []
    for b0 in range(0, len(paths), a.batch):
        chunk = paths[b0:b0 + a.batch]
        res = model.predict(chunk, imgsz=a.imgsz, conf=0.05, verbose=False, device=a.device)
        for p, r in zip(chunk, res):
            iid = os.path.basename(p)[:-4]
            dets = [(b, float(c), model.names[int(k)]) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            im = r.orig_img; H, W = im.shape[:2]
            sf = smooth_frac(im); skyish = sf >= a.min_smooth or iid in cloud
            small = [d for d in dets if d[2] in ("uav", "airplane") and max(d[0][2] - d[0][0], d[0][3] - d[0][1]) <= a.max_box_frac * max(H, W)]
            mx = max([c for b, c, n in small], default=0.0)
            stats.append({"image": iid, "smooth_frac": round(sf, 3), "cloud_label": iid in cloud, "skyish": skyish,
                          "max_alarm_conf_small": mx})
            if not skyish:
                continue
            fps = sorted([d for d in small if d[1] >= a.conf], key=lambda d: -d[1])[:a.per_image]
            if not fps:
                continue
            n_img_fp += 1
            for j, (b, c, nm) in enumerate(fps):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                tx, ty = int(np.clip(cx - TILE / 2, 0, max(W - TILE, 0))), int(np.clip(cy - TILE / 2, 0, max(H - TILE, 0)))
                tile = np.full((TILE, TILE, 3), 114, np.uint8)
                crop = im[ty:ty + TILE, tx:tx + TILE]; tile[:crop.shape[0], :crop.shape[1]] = crop
                stem = f"oi_{iid}_{j}"
                cv2.imwrite(os.path.join(tdir, stem + ".jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"tile": stem + ".jpg", "session": "openimages", "day_frame": 0, "rgb_conf": round(c, 3),
                             "x0": round(b[0] - tx, 1), "y0": round(b[1] - ty, 1), "x1": round(b[2] - tx, 1), "y1": round(b[3] - ty, 1),
                             "decision": "", "pred": nm})
            if iid in meta.index:
                m = meta.loc[iid]
                attr.append({"ImageID": iid, "Author": m.Author, "AuthorProfileURL": m.AuthorProfileURL, "License": m.License,
                             "OriginalLandingURL": m.OriginalLandingURL, "Title": m.Title})
        if (b0 // a.batch) % 50 == 0:
            print(f"  {b0 + len(chunk)}/{len(paths)} images, {n_img_fp} with a false alarm >= {a.conf}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "review", "tiles.csv"), index=False)
    pd.DataFrame(attr).to_csv(os.path.join(a.out, "attribution.csv"), index=False)
    S = pd.DataFrame(stats); S.to_csv(os.path.join(a.out, "per_image.csv"), index=False)
    K = S[S.skyish]
    print(f"{len(paths)} images, {len(K)} sky-dominated or Cloud-labelled: {n_img_fp} ({100 * n_img_fp / max(len(K), 1):.1f}%) with a "
          f"small uav/airplane box >= {a.conf}; >= 0.15: {(K.max_alarm_conf_small >= 0.15).sum()}, >= 0.25: "
          f"{(K.max_alarm_conf_small >= 0.25).sum()}; {len(rows)} review tiles")


if __name__ == "__main__":
    main()

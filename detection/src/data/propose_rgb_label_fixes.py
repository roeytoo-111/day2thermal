r"""
Propose label corrections for an RGB detection dataset by comparing its labels with a detector's predictions, for
a human to decide in src/eval/review_label_fixes.py. Nothing is changed here.

Why: the fw4 review found that ALL 450 "background -> bird" false positives in v3 valid sit in images that already
have labelled birds (flocks boxed partially), plus birds in images labelled empty. That noise trains the model to
suppress real birds. The same comparison also surfaces doubtful uav labels (stands, hands, 7 px boxes in crowds).

Proposal kinds, one row per box (queue.csv):
  missing      model box (conf >= --miss-conf, per class) that no label overlaps (IoU < 0.3 and centre outside
               every label box): probably an unlabelled object. Proposed class = model class.
  wrong_class  label whose best overlapping model box (IoU >= 0.5, conf >= --class-conf) has another class while
               no same-class box >= 0.1 overlaps it.
  unsupported  label with no model box of any class at conf >= 0.05 overlapping it (IoU >= 0.1): doubtful label,
               out-of-scope drone (stand, hand) or a hard true positive the model misses. Only for --unsup-classes.
Images are ordered so that images with many proposals come first (flocks); --max-images caps the queue.

    python3 src/data/propose_rgb_label_fixes.py --data ../ANN-Detection/data/datasets/Fixed_wing_v3 \
        --model ../ANN-Detection/models/yolo26s_rgb_20261012/yolo26s_fw4_20261011.pt --out data/rgb_fixes/v3
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgb_ds as R   # noqa: E402


def iou_m(a, b):
    """IoU matrix of xyxy arrays a (n,4), b (m,4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,valid,test"); ap.add_argument("--names", default=None)
    ap.add_argument("--miss-conf", default="uav:0.30,airplane:0.30,bird:0.25")
    ap.add_argument("--class-conf", type=float, default=0.5)
    ap.add_argument("--unsup-classes", default="uav,airplane")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    root = R.find_root(a.data)
    names = a.names.split(",") if a.names else R.read_names(root)
    model = YOLO(a.model); mnames = {int(k): v for k, v in model.names.items()}
    m2d = {k: names.index(v) for k, v in mnames.items() if v in names}   # model class -> dataset class
    miss = {names.index(k): float(v) for k, v in (x.split(":") for x in a.miss_conf.split(",")) if k in names}
    unsup = {names.index(x) for x in a.unsup_classes.split(",") if x in names}
    print(f"dataset names {names}; model names {mnames}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    imgs = [(s, p) for s in a.splits.split(",") for p in R.list_split(root, s)]
    rows, done = [], 0
    for b0 in range(0, len(imgs), a.batch):
        chunk = imgs[b0:b0 + a.batch]
        res = model.predict([p for _, p in chunk], imgsz=a.imgsz, conf=0.05, verbose=False, device=a.device)
        for (split, p), r in zip(chunk, res):
            H, W = r.orig_shape
            P = r.boxes.xyxy.cpu().numpy(); Pc = np.array([m2d.get(int(c), -1) for c in r.boxes.cls.tolist()]); Ps = r.boxes.conf.cpu().numpy()
            L = R.read_labels(R.label_path(p))
            G = np.array([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] for _, cx, cy, w, h in L]).reshape(-1, 4)
            Gc = np.array([c for c, *_ in L])
            M = iou_m(P, G)
            base = {"split": split, "image": p, "W": W, "H": H}
            for i in range(len(P)):
                c = Pc[i]
                if c < 0 or Ps[i] < miss.get(c, 1.1):
                    continue
                cx, cy = (P[i, 0] + P[i, 2]) / 2, (P[i, 1] + P[i, 3]) / 2
                inside = len(G) and ((G[:, 0] <= cx) & (cx <= G[:, 2]) & (G[:, 1] <= cy) & (cy <= G[:, 3])).any()
                if (not len(G) or M[i].max() < 0.3) and not inside:
                    rows.append({**base, "kind": "missing", "label_idx": -1, "cls": names[c], "proposed": names[c],
                                 "conf": round(float(Ps[i]), 3), "x0": P[i, 0], "y0": P[i, 1], "x1": P[i, 2], "y1": P[i, 3]})
            for j in range(len(G)):
                col = M[:, j] if len(P) else np.zeros(0)
                if not len(col) or col.max() < 0.1:
                    if Gc[j] in unsup:
                        rows.append({**base, "kind": "unsupported", "label_idx": j, "cls": names[Gc[j]], "proposed": "",
                                     "conf": 0.0, "x0": G[j, 0], "y0": G[j, 1], "x1": G[j, 2], "y1": G[j, 3]})
                    continue
                same = (Pc == Gc[j]) & (col >= 0.1)
                k = int(np.argmax(np.where(col >= 0.5, Ps, -1)))
                if col[k] >= 0.5 and Ps[k] >= a.class_conf and Pc[k] != Gc[j] and Pc[k] >= 0 and not same.any():
                    rows.append({**base, "kind": "wrong_class", "label_idx": j, "cls": names[Gc[j]], "proposed": names[Pc[k]],
                                 "conf": round(float(Ps[k]), 3), "x0": G[j, 0], "y0": G[j, 1], "x1": G[j, 2], "y1": G[j, 3]})
        done += len(chunk)
        if done % 2000 < a.batch:
            print(f"  {done}/{len(imgs)} images, {len(rows)} proposals", flush=True)
    Q = pd.DataFrame(rows)
    if Q.empty:
        print("no proposals"); return
    for c in ("x0", "y0", "x1", "y1"):
        Q[c] = Q[c].round(1)
    n_per = Q.groupby("image").size().rename("n_img")
    Q = Q.join(n_per, on="image").sort_values(["n_img", "image", "kind", "x0"], ascending=[False, True, True, True])
    if a.max_images:
        keep = Q.image.drop_duplicates().head(a.max_images)
        Q = Q[Q.image.isin(keep)]
    Q["decision"] = ""; Q["nx0"] = np.nan; Q["ny0"] = np.nan; Q["nx1"] = np.nan; Q["ny1"] = np.nan
    Q.to_csv(os.path.join(a.out, "queue.csv"), index=False)
    open(os.path.join(a.out, "dataset.txt"), "w").write(f"{root}\n{','.join(names)}\n")
    s = Q.groupby(["split", "kind", "cls"]).size()
    print(s.to_string()); print(f"{len(Q)} proposals in {Q.image.nunique()} images -> {a.out}/queue.csv")


if __name__ == "__main__":
    main()

r"""
Compare RGB detectors on YOLO-format test sets: Ultralytics val (mAP50, mAP50-95, per class) plus the deployment view at
conf --thr (0.15): uav recall (IoU >= 0.5, same class and class-agnostic), false uav boxes on background (uav
prediction with IoU < 0.45 to every ground-truth box of any class, as in the fw4 README "Background predicted as uav"),
and the share of images with no uav ground truth that get a uav box.

    python3 src/eval/score_rgb_models.py --model fw4 fw4.pt --model v5a best.pt \
        --set v5_test ../data/rgb_eval/v5_test.yaml --set unseen ../data/rgb_eval/unseen_test/data.yaml \
        --set boson ../data/rgb_eval/boson_heldout/data.yaml --out results/scores_rgb_v5a.md
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import rgb_ds as R   # noqa: E402
from propose_rgb_label_fixes import iou_m   # noqa: E402


def md_table(T):
    fmt = lambda v: (str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() and abs(v) >= 1 else
                     f"{v:.3f}" if isinstance(v, float) and abs(v) < 1.5 else f"{v:.1f}" if isinstance(v, float) else str(v))
    lines = ["| | " + " | ".join(T.columns) + " |", "|---|" + "---|" * len(T.columns)]
    lines += ["| " + idx + " | " + " | ".join(fmt(T.at[idx, c]) for c in T.columns) + " |" for idx in T.index]
    return "\n".join(lines)


def deploy_view(model, img_dir, names, thr, imgsz, device):
    uav = names.index("uav")
    imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(R.IMG_EXT))
    n_gt = tp_same = tp_any = fp_bg = neg_imgs = neg_fired = 0
    for b0 in range(0, len(imgs), 32):
        res = model.predict(imgs[b0:b0 + 32], imgsz=imgsz, conf=thr, verbose=False, device=device)
        for p, r in zip(imgs[b0:b0 + 32], res):
            H, W = r.orig_shape
            L = R.read_labels(R.label_path(p))
            G = np.array([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] for _, cx, cy, w, h in L]).reshape(-1, 4)
            Gc = np.array([c for c, *_ in L])
            P = r.boxes.xyxy.cpu().numpy(); Pc = r.boxes.cls.cpu().numpy().astype(int)
            mnames = r.names
            Pc = np.array([names.index(mnames[c]) if mnames[c] in names else -1 for c in Pc])
            M = iou_m(G, P)
            gu = np.nonzero(Gc == uav)[0] if len(Gc) else []
            n_gt += len(gu)
            for j in gu:
                tp_any += bool(len(P) and (M[j] >= 0.5).any())
                tp_same += bool(len(P) and ((M[j] >= 0.5) & (Pc == uav)).any())
            pu = np.nonzero(Pc == uav)[0]
            fp_bg += int(sum(1 for k in pu if not len(G) or M[:, k].max() < 0.45))
            if not len(gu):
                neg_imgs += 1; neg_fired += bool(len(pu))
    return {"uav GT": n_gt, f"uav recall@{thr}": 100 * tp_same / max(n_gt, 1), f"uav found any class@{thr}": 100 * tp_any / max(n_gt, 1),
            f"background->uav boxes@{thr}": fp_bg, f"no-uav images with a uav box@{thr} (%)": 100 * neg_fired / max(neg_imgs, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs=2, action="append", required=True, metavar=("NAME", "PT"))
    ap.add_argument("--set", nargs=2, action="append", required=True, metavar=("NAME", "YAML"))
    ap.add_argument("--thr", type=float, default=0.15); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    blocks = []
    for sname, y in a.set:
        cfg = yaml.safe_load(open(y)); names = [str(x) for x in cfg["names"]]
        img_dir = os.path.join(cfg.get("path", os.path.dirname(y)), cfg["test"])
        rows = {}
        for mname, pt in a.model:
            m = YOLO(pt)
            v = m.val(data=y, split="test", imgsz=a.imgsz, batch=16, conf=0.001, device=a.device, verbose=False, plots=False)
            r = {"mAP50": v.box.map50, "mAP50-95": v.box.map}
            for i, n in enumerate(names):
                if i < len(v.box.ap50):
                    r[f"{n} AP50"] = v.box.ap50[i]; r[f"{n} AP50-95"] = v.box.ap[i]
            r.update(deploy_view(m, img_dir, names, a.thr, a.imgsz, a.device))
            rows[mname] = r
        T = pd.DataFrame(rows)
        n_img = len([f for f in os.listdir(img_dir) if f.lower().endswith(R.IMG_EXT)])
        blocks.append(f"### {sname} ({n_img} images)\n\n" + md_table(T))
        print(blocks[-1], flush=True)
    if a.out:
        open(a.out, "w").write("# RGB model comparison\n\n" + "\n\n".join(blocks) + "\n")


if __name__ == "__main__":
    main()

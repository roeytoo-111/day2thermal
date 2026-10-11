r"""
Svanström et al. Drone-detection-dataset (CC0; github.com/DroneDetectionThesis/Drone-detection-dataset), IR part:
365 clips of 320x256 (~10 s each; 157 drone, 79 bird, 74 airplane, 55 helicopter), one labelled target per frame
(MATLAB groundTruth .mat, read with mcos-decoder). Ground camera looking up at sky above a treeline -- the
interceptor's main case.

Output: a YOLO dataset in the uav = 0 convention ([uav, bird]):
  drone -> class 0, bird -> class 1, airplane / helicopter -> empty label (hard negative: a real object that is not
  a drone). Frames are upscaled x2 to 640x512 (our cameras' frame size; the box scales with it).
Split BY CLIP (neighbouring frames are near-duplicates): every clip goes wholly to train / valid / test, chosen
by a seeded shuffle within each class. --every N keeps one frame in N (default 30 = 1 per second); frames
where the labelled target is absent are kept only in negative classes.

    python3 src/data/svanstrom_to_yolo.py --src data/svanstrom/Video_IR --out data/svanstrom_yolo
"""
import os
import glob
import json
import argparse
import random

import cv2

CLS = {"DRONE": 0, "BIRD": 1, "AIRPLANE": None, "HELICOPTER": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=30); ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--valid-frac", type=float, default=0.1); ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from mcos_decoder import load_groundtruth
    rng = random.Random(a.seed)
    split_of, stats = {}, {}
    for c in CLS:
        vs = sorted(glob.glob(os.path.join(a.src, f"IR_{c}_*.mp4")))
        rng.shuffle(vs)
        nt, nv = round(len(vs) * a.test_frac), round(len(vs) * a.valid_frac)
        for i, v in enumerate(vs):
            split_of[v] = "test" if i < nt else "valid" if i < nt + nv else "train"
    for s in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, sub), exist_ok=True)
    for v, split in sorted(split_of.items()):
        c = os.path.basename(v).split("_")[1]
        boxes = load_groundtruth(v.replace(".mp4", "_LABELS.mat"))
        cap, i = cv2.VideoCapture(v), -1
        while True:
            ok, im = cap.read()
            i += 1
            if not ok:
                break
            if i % a.every:
                continue
            b = boxes[i] if i < len(boxes) else None
            if CLS[c] is not None and b is None:
                continue                                        # target out of view: no reliable label
            im = cv2.resize(im, (im.shape[1] * a.scale, im.shape[0] * a.scale), interpolation=cv2.INTER_LINEAR)
            H, W = im.shape[:2]
            stem = f"svan_{os.path.basename(v)[3:-4]}_{i:04d}"
            cv2.imwrite(os.path.join(a.out, split, "images", stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            line = ""
            if CLS[c] is not None:
                x, y, w, h = (val * a.scale for val in b)
                line = f"{CLS[c]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n"
            open(os.path.join(a.out, split, "labels", stem + ".txt"), "w").write(line)
            k = (split, c)
            stats[k] = stats.get(k, 0) + 1
    open(os.path.join(a.out, "data.yaml"), "w").write(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\nnames:\n- uav\n- bird\n")
    json.dump({"args": vars(a), "clips": {os.path.basename(k): s for k, s in split_of.items()},
               "frames": {f"{s}/{c}": n for (s, c), n in sorted(stats.items())}},
              open(os.path.join(a.out, "build_config.json"), "w"), indent=1)
    for (s, c), n in sorted(stats.items()):
        print(f"{s:5s} {c:10s} {n}")


if __name__ == "__main__":
    main()

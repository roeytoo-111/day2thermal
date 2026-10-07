r"""
RGB training set for fine-tuning the production day model (yolo26s_20260921_145030_rgbcurrent.pt, classes
{0: uav, 1: airplane, 2: bird}) on Fixed_wing_v3 (names [bird, airplane, uav]) + the 2026-10-07 Boson addon
(336 confirmed uav tiles, 659 mined false-positive tiles, labels in v3 order). Labels are REWRITTEN to the model's
order (bird 0->2, uav 2->0) so fine-tuning does not scramble the classes; images are hard links.

  train      v3 train + addon train          valid  v3 valid          test  v3 test
  test_boson addon test: confirmed uav tiles from the two held-out Boson sessions (new camera rig)

    python3 make_rgb_dataset.py --v3 ~/rgbdata/Fixed_wing_v3 --addon ~/rgbdata/Fixed_wing_v3_boson --out ~/rgbdata/fw_v3boson
"""
import os, glob, shutil, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--v3", required=True); ap.add_argument("--addon", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
REMAP = {0: 2, 1: 1, 2: 0}
def put(src_split_dir, out_split, prefix=""):
    n = 0
    for sub in ("images", "labels"): os.makedirs(os.path.join(a.out, out_split, sub), exist_ok=True)
    for img in glob.glob(os.path.join(src_split_dir, "images", "*")):
        stem = os.path.splitext(os.path.basename(img))[0]
        dst = os.path.join(a.out, out_split, "images", prefix + os.path.basename(img))
        try: os.link(img, dst)
        except OSError: shutil.copy2(img, dst)
        lbl = os.path.join(src_split_dir, "labels", stem + ".txt"); lines = []
        if os.path.exists(lbl):
            for l in open(lbl):
                p = l.split()
                # boxes (5 fields) AND Roboflow polygons (class + x y pairs; ultralytics turns them into boxes)
                if len(p) >= 5 and len(p) % 2 == 1 and p[0].isdigit() and int(p[0]) in REMAP:
                    lines.append(" ".join([str(REMAP[int(p[0])])] + p[1:]))
        open(os.path.join(a.out, out_split, "labels", prefix + stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n
c = {"train_v3": put(os.path.join(a.v3, "train"), "train"), "train_addon": put(os.path.join(a.addon, "train"), "train"),
     "valid": put(os.path.join(a.v3, "valid"), "valid"), "test": put(os.path.join(a.v3, "test"), "test"),
     "test_boson": put(os.path.join(a.addon, "test"), "test_boson")}
o = os.path.abspath(a.out)
open(os.path.join(a.out, "data.yaml"), "w").write(f"path: {o}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 3\nnames: ['uav', 'airplane', 'bird']\n")
open(os.path.join(a.out, "data_boson_test.yaml"), "w").write(f"path: {o}\ntrain: train/images\nval: test_boson/images\nnc: 3\nnames: ['uav', 'airplane', 'bird']\n")
print(c)

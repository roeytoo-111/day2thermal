r"""
Real drone-free thermal backgrounds as EMPTY training images (hard negatives), from paste_on_backgrounds.py work
dirs (`select` output: real thermal crops from TRAINING sessions + the detector boxes to remove in backgrounds.csv).

Same cleaning as paste_on_backgrounds build: the real drone(s) (detections >= --remove-conf, <= --remove-max-size px)
are inpainted, and a crop whose removal site lies on texture (ring high-pass std > --drop-textured) is dropped (an
inpaint there leaves a smear, and the box may be a real dark drone). No paste is added: in M0's data these
backgrounds only ever appear WITH a pasted drone, so the model never sees this sky empty.

    python3 src/data/make_clean_negatives.py --work data/bgpaste data/boson_work/bg_boson --out data/boson_work/clean_neg
"""
import os
import csv
import json
import argparse

import cv2

from paste_on_backgrounds import remove_real_drones, removal_on_texture


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--remove-conf", type=float, default=0.25); ap.add_argument("--remove-max-size", type=float, default=40)
    ap.add_argument("--drop-textured", type=float, default=12.0)
    a = ap.parse_args()
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    stats = {}
    for w in a.work:
        tag = os.path.basename(os.path.normpath(w)); n = kept = 0
        for r in csv.DictReader(open(os.path.join(w, "backgrounds.csv"))):
            n += 1
            g = cv2.imread(os.path.join(w, "real", r["name"]), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            boxes = json.loads(r["remove_boxes"])
            if removal_on_texture(g, boxes, a.remove_conf, a.remove_max_size) > a.drop_textured:
                continue
            clean = remove_real_drones(g, boxes, a.remove_conf, a.remove_max_size)
            stem = f"neg_{tag}_{os.path.splitext(r['name'])[0]}"
            cv2.imwrite(os.path.join(a.out, "images", stem + ".jpg"), cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(os.path.join(a.out, "labels", stem + ".txt"), "w").close()
            kept += 1
        stats[tag] = {"crops": n, "kept_as_negatives": kept}
        print(f"{tag}: {kept} of {n} crops kept as empty negatives")
    json.dump({"args": vars(a), "stats": stats}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

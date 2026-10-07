r"""
Pool of NEW real thermal backgrounds for paste_on_backgrounds.py: the clean-negative frames (build_boson_labels.py
status 'neg': neither the RGB nor the thermal model sees anything) of the TRAIN sessions, thinned to --step frames.
Writes <out>/real/<name>.png (the thermal frame, gray) and <out>/backgrounds.csv (name, remove_boxes="[]") -- the
format `paste_on_backgrounds.py build --work <out>` reads (its `select` step is not needed: no drone to erase).

    python3 src/data/make_boson_backgrounds.py --labels-dir data/boson_work/labels --mp4-dir data/boson_work/mp4 \
        --holdout 08_58_15,10_15_59 --out data/boson_work/bg_boson
"""
import os, glob, argparse
import cv2, pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--labels-dir", required=True)
ap.add_argument("--mp4-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--holdout", default="")
ap.add_argument("--step", type=int, default=45, help="keep every N-th thermal frame among negatives")
a = ap.parse_args()
hold = {s for s in a.holdout.split(",") if s}
os.makedirs(os.path.join(a.out, "real"), exist_ok=True)
rows = []
for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))):
    s = os.path.basename(f)[:-4]
    if s in hold:
        continue
    d = pd.read_csv(f)
    want = set(d[(d.status == "neg") & (d.frame % a.step == 0)].frame.astype(int))
    if not want:
        continue
    cap = cv2.VideoCapture(os.path.join(a.mp4_dir, f"{s}_thermal.mp4"))
    i = 0
    while i <= max(want):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            name = f"b{s}_{i:06d}.png"
            cv2.imwrite(os.path.join(a.out, "real", name), cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            rows.append({"name": name, "session": s, "frame": i, "n_removed": 0, "remove_boxes": "[]"})
        i += 1
pd.DataFrame(rows).to_csv(os.path.join(a.out, "backgrounds.csv"), index=False)
print(f"{len(rows)} backgrounds from {pd.DataFrame(rows).session.nunique() if rows else 0} sessions -> {a.out}")

r"""
Daniel's review decisions -> YOLO training frames (native 640x512), for the M4 run:
  review/tneg_<s>.csv   thermal-only detections (day model silent): y = a drone the day model missed (positive, box),
                        n = a verified thermal false alarm (clouds etc.): the frame becomes a hard negative (empty label)
  yolo_boson/hard_<s>.csv  day-only detections: y = positive (box); n is NOT used (the drone may be elsewhere)
Held-out sessions (08_58_15, 10_15_59) are never exported.
    python3 src/data/export_reviewed.py --work data/boson_work --out data/boson_work/yolo_boson_reviewed
"""
import os, glob, argparse
import cv2, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
HOLD = {"08_58_15", "10_15_59"}
want = {}
for f in glob.glob(os.path.join(a.work, "review", "tneg_*.csv")) + glob.glob(os.path.join(a.work, "yolo_boson", "hard_*.csv")):
    s = os.path.basename(f).split("_", 1)[1][:-4]
    if s in HOLD: continue
    d = pd.read_csv(f); d["reviewed"] = d["reviewed"].fillna("")
    d = d[d.reviewed == "y"]
    neg_ok = os.path.basename(f).startswith("tneg_")
    for r in d.itertuples():
        if r.verdict == "drone":
            want.setdefault(s, {})[int(r.frame_id)] = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
        elif r.verdict == "nothing" and neg_ok:
            want.setdefault(s, {})[int(r.frame_id)] = None
counts = {"pos": 0, "neg": 0}
for sub in ("images", "labels"): os.makedirs(os.path.join(a.out, "train", sub), exist_ok=True)
for s, frames in want.items():
    cap = cv2.VideoCapture(os.path.join(a.work, "mp4", f"{s}_thermal.mp4")); i = 0
    while i <= max(frames):
        ok, im = cap.read()
        if not ok: break
        if i in frames:
            b = frames[i]; stem = f"rv{s}_{i:06d}"
            cv2.imwrite(os.path.join(a.out, "train", "images", stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            line = "" if b is None else f"1 {(b[0] + b[2]) / 1280:.6f} {(b[1] + b[3]) / 1024:.6f} {(b[2] - b[0]) / 640:.6f} {(b[3] - b[1]) / 512:.6f}\n"
            open(os.path.join(a.out, "train", "labels", stem + ".txt"), "w").write(line)
            counts["neg" if b is None else "pos"] += 1
        i += 1
print("exported reviewed frames:", counts, "->", a.out)

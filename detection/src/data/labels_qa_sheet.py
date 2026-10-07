"""Contact sheet of automatic labels for human spot-checking: N per session of status 'pos' (green box) and optionally
'hard' (red box), cropped around the box on the thermal frame.
    python3 src/data/labels_qa_sheet.py --labels-dir data/boson_work/labels --mp4-dir data/boson_work/mp4 --out data/boson_work/qa"""
import os, glob, argparse
import cv2, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--labels-dir", required=True); ap.add_argument("--mp4-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per", type=int, default=6); ap.add_argument("--cols", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for status, color, fname in (("pos", (0, 255, 0), "labels_pos_qa.jpg"), ("hard", (0, 0, 255), "labels_hard_qa.jpg")):
    tiles = []
    for f in sorted(glob.glob(os.path.join(a.labels_dir, "*.csv"))):
        s = os.path.basename(f)[:-4]
        d = pd.read_csv(f); d = d[d.status == status]
        if d.empty: continue
        d = d.iloc[np.linspace(0, len(d) - 1, min(a.per, len(d))).astype(int)]
        cap = cv2.VideoCapture(os.path.join(a.mp4_dir, f"{s}_thermal.mp4")); want = {int(r.frame): r for r in d.itertuples()}
        i = 0
        while i <= max(want):
            ok, im = cap.read()
            if not ok: break
            if i in want:
                r = want[i]; x0, y0, x1, y1 = [int(float(v)) for v in (r.x0, r.y0, r.x1, r.y1)]; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                ox, oy = max(cx - 40, 0), max(cy - 40, 0); c = cv2.resize(im[oy:cy + 40, ox:cx + 40], (160, 160), interpolation=cv2.INTER_NEAREST); k = 2.0
                cv2.rectangle(c, (int((x0 - ox) * k) - 2, int((y0 - oy) * k) - 2), (int((x1 - ox) * k) + 2, int((y1 - oy) * k) + 2), color, 1)
                cv2.putText(c, f"{s} {i}", (2, 11), 0, 0.33, (0, 255, 255), 1); tiles.append(c)
            i += 1
    while len(tiles) % a.cols: tiles.append(np.zeros((160, 160, 3), np.uint8))
    cv2.imwrite(os.path.join(a.out, fname), np.vstack([np.hstack(tiles[k:k + a.cols]) for k in range(0, len(tiles), a.cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(status, len(tiles), "tiles ->", fname)

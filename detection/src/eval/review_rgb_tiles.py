r"""
Decide the RGB-only detections from build_rgb_addon.py (tiles in <addon>/review/tiles, boxes in tiles.csv):

  d   a real DRONE (the thermal camera missed it)         -> tile joins train/ with a uav label
  f   a FALSE ALARM (cloud, contrail, wire, terrain, ...) -> tile joins train/ as a hard negative (empty label file)
  space  skip if the tile holds several flying objects (only the boxed one would get a label)
  b   bird: keep the tile with a bird label on this box
  a   airplane (manned aircraft): keep the tile with an airplane label on this box
  drag  redraw the box (then d)
  u   undo (previous tile)      space  skip      q  save and quit (resumable)

    python3 src/eval/review_rgb_tiles.py --addon data/boson_work/Fixed_wing_v3_boson
    python3 src/eval/review_rgb_tiles.py --addon data/boson_work/Fixed_wing_v3_boson --finalize   # write the labels
"""
import os, sys, argparse, shutil
import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--addon", required=True)
ap.add_argument("--zoom", type=float, default=1.0, help="tile magnification (tiles are 640 px; capped so the window is <= --max-h tall)")
ap.add_argument("--max-h", type=int, default=700)
ap.add_argument("--finalize", action="store_true")
ap.add_argument("--names", default="bird,airplane,uav", help="class order of the labels written by --finalize "
                "(default: Fixed_wing_v3's; use uav,airplane,bird for v4 and later)")
a = ap.parse_args()
csv = os.path.join(a.addon, "review", "tiles.csv")
df = pd.read_csv(csv)
df["decision"] = df["decision"].fillna("").astype(str)

if a.finalize:
    n = {"d": 0, "f": 0, "b": 0, "a": 0}
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.addon, "train", sub), exist_ok=True)
    for r in df.itertuples():
        if r.decision not in n:
            continue
        stem = r.tile[:-4]
        shutil.copy2(os.path.join(a.addon, "review", "tiles", r.tile), os.path.join(a.addon, "train", "images", "rv_" + r.tile))
        line = ""
        if r.decision in ("d", "b", "a"):
            names = a.names.split(",")
            cls = names.index({"d": "uav", "b": "bird", "a": "airplane"}[r.decision])
            line = f"{cls} {(r.x0 + r.x1) / 2 / 640:.6f} {(r.y0 + r.y1) / 2 / 640:.6f} {(r.x1 - r.x0) / 640:.6f} {(r.y1 - r.y0) / 640:.6f}\n"
        open(os.path.join(a.addon, "train", "labels", "rv_" + stem + ".txt"), "w").write(line)
        n[r.decision] += 1
    print("finalized:", n, "(d = drone tiles, f = hard negatives, b = bird tiles)")
    sys.exit(0)

todo = [i for i in df.index if df.at[i, "decision"] == ""]
state = {"down": None, "cur": None, "box": None}
Z = min(a.zoom, a.max_h / 640)
def mouse(ev, x, y, flags, p):
    if ev == cv2.EVENT_LBUTTONDOWN: state["down"] = (x / Z, y / Z)
    elif ev == cv2.EVENT_MOUSEMOVE and state["down"] is not None: state["cur"] = (x / Z, y / Z)
    elif ev == cv2.EVENT_LBUTTONUP and state["down"] is not None:
        (x0, y0), (x1, y1) = state["down"], (x / Z, y / Z)
        if abs(x1 - x0) > 2 and abs(y1 - y0) > 2: state["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        state["down"] = None
cv2.namedWindow("rgb tile"); cv2.setMouseCallback("rgb tile", mouse)
i = 0
while 0 <= i < len(todo):
    k = todo[i]; r = df.loc[k]
    im = cv2.imread(os.path.join(a.addon, "review", "tiles", r.tile))
    state["box"] = [r.x0, r.y0, r.x1, r.y1]
    while True:
        v = cv2.resize(im, None, fx=Z, fy=Z, interpolation=cv2.INTER_LINEAR)
        b = state["box"]
        cv2.rectangle(v, (int(b[0] * Z) - 3, int(b[1] * Z) - 3), (int(b[2] * Z) + 3, int(b[3] * Z) + 3), (0, 255, 255), 1)
        done = int((df.decision != "").sum())
        cv2.putText(v, f"{r.session} f{r.day_frame} rgb {r.rgb_conf}  [{done}/{len(df)}]  d=drone a=airplane b=bird f=false alarm u=undo space=skip q=quit",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("rgb tile", v)
        c = cv2.waitKey(30) & 0xFF
        if c in (ord("d"), ord("f"), ord("b"), ord("a")):
            df.at[k, "decision"] = chr(c)
            if c in (ord("d"), ord("b"), ord("a")): df.loc[k, ["x0", "y0", "x1", "y1"]] = [round(float(x), 1) for x in state["box"]]
            i += 1; break
        if c == ord(" "): i += 1; break
        if c == ord("u"): i = max(i - 1, 0); df.at[todo[i], "decision"] = ""; break
        if c == ord("q"):
            df.to_csv(csv, index=False); print("saved", csv); sys.exit(0)
df.to_csv(csv, index=False); print("all tiles decided; now run with --finalize")

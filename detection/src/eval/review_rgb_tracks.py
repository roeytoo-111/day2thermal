r"""
One decision per object track from mine_rgb_tracks.py mine (<W>/tracks/<session>.json). Top: up to 8 zoomed crops spread
over the track (magenta box = the detection, green = its confidence). Bottom: the whole frame at the track's best
detection (magenta circle).

Track decision:
  u   drone (uav)          p   airplane          b   bird
  f   false alarm (cloud, terrain, glare, ...): its tiles become negatives
  x   unsure / mixed (track jumps between objects): never used
  space  skip (asked again)      z  back      q  save and quit (resumable)

Drawing your own box (an object the model missed, or a detection box that is off):
  drag on a crop or on the whole frame, then press u / p / b to label it (c clears an unlabelled drawing,
  r removes your last labelled box on this track).
  The box is stored for THAT frame and exported as a labelled object; the screen stays on the track (cyan boxes =
  your boxes), so decide the track afterwards with u / p / b / f / x as usual. A false-alarm track with a drawn drone
  next to it is fine: press f for the track after labelling the drone.
No Enter key.

    python3 src/eval/review_rgb_tracks.py --W ../data/flight_20261014
    python3 src/eval/review_rgb_tracks.py --W ../data/flight_20261014 --min-n 3     # long tracks first pass
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np

KEYS = {ord("u"): "uav", ord("p"): "airplane", ord("b"): "bird", ord("f"): "f", ord("x"): "x"}
HDR, CW, CS = 52, 165, 170          # header height, crop slot width, strip height


def crop_geo(b, W, H, out=160, minside=48):
    """Same geometry as mine_rgb_tracks.crop: (x0, y0, zoom) of the crop around box b."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(3 * max(b[2] - b[0], b[3] - b[1]), minside)
    x0, y0 = int(np.clip(cx - s / 2, 0, max(W - s, 0))), int(np.clip(cy - s / 2, 0, max(H - s, 0)))
    h, w = min(int(s), H - y0), min(int(s), W - x0)
    return x0, y0, out / max(h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--W", required=True); ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--session", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.W, "tracks", "*.json")))
    if a.session:
        files = [f for f in files if os.path.basename(f)[:-5] == a.session]
    if not files:
        raise SystemExit(f"no tracks in {a.W}/tracks yet: run detection (rgb_label_videos.py) and "
                         "mine_rgb_tracks.py mine first")
    docs = {f: json.load(open(f)) for f in files}
    todo = [(f, i) for f in files for i, t in enumerate(docs[f]["tracks"]) if t["decision"] == "" and t["n"] >= a.min_n]
    n_all = sum(len(d["tracks"]) for d in docs.values())
    save = lambda: [json.dump(d, open(f, "w")) for f, d in docs.items()]
    st = {"down": None, "cur": None, "geo": [], "drawn": None}

    def to_frame(x, y):
        """canvas px -> (frame k, frame x, frame y) via the panel under the cursor, or None."""
        for (px0, py0, px1, py1, k, fx0, fy0, z) in st["geo"]:
            if px0 <= x < px1 and py0 <= y < py1:
                return k, fx0 + (x - px0) / z, fy0 + (y - py0) / z, (px0, py0, px1, py1)
        return None

    def mouse(ev, x, y, flags, p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["down"] = (x, y); st["cur"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["down"]:
            st["cur"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["down"]:
            a_, b_ = to_frame(*st["down"]), to_frame(x, y)
            if a_ and b_ and a_[3] == b_[3] and abs(x - st["down"][0]) > 2 and abs(y - st["down"][1]) > 2:
                st["drawn"] = {"k": a_[0], "box": [round(min(a_[1], b_[1]), 1), round(min(a_[2], b_[2]), 1),
                                                   round(max(a_[1], b_[1]), 1), round(max(a_[2], b_[2]), 1)]}
            st["down"] = None; st["cur"] = None

    cv2.namedWindow("tracks"); cv2.setMouseCallback("tracks", mouse)
    hist, j = [], 0
    while 0 <= j < len(todo):
        f, i = todo[j]; d = docs[f]; t = d["tracks"][i]; s = d["session"]
        FW, FH = d.get("frame_size", [3840, 2160])
        t.setdefault("manual", [])
        cdir = os.path.join(a.W, "tracks", "crops")
        crops = [cv2.imread(os.path.join(cdir, f"{s}_{t['id']}_{n}.jpg")) for n in range(len(t["crop_idx"]))]
        ctx = cv2.imread(os.path.join(cdir, f"{s}_{t['id']}_ctx.jpg"))
        ctx = ctx if ctx is not None else np.zeros((540, 960, 3), np.uint8)
        jb = int(np.argmax(t["c"])); zc = ctx.shape[1] / FW
        st["geo"] = []
        for n, c in enumerate(crops[:8]):
            if c is None:
                continue
            x0, y0, z = crop_geo(t["b"][t["crop_idx"][n]], FW, FH)
            st["geo"].append((n * CW, HDR + 5, n * CW + c.shape[1], HDR + 5 + c.shape[0], t["k"][t["crop_idx"][n]], x0, y0, z))
        st["geo"].append((0, HDR + CS, ctx.shape[1], HDR + CS + ctx.shape[0], t["k"][jb], 0, 0, zc))
        st["drawn"] = None
        moved = False
        while not moved:
            Wc = max(8 * CW, ctx.shape[1])
            canvas = np.zeros((HDR + CS + ctx.shape[0], Wc, 3), np.uint8)
            for n, c in enumerate(crops[:8]):
                if c is not None:
                    canvas[HDR + 5:HDR + 5 + c.shape[0], n * CW:n * CW + c.shape[1]] = c
            canvas[HDR + CS:, :ctx.shape[1]] = ctx
            for m in t["manual"] + ([st["drawn"]] if st["drawn"] else []):     # your boxes on every panel of that frame
                for (px0, py0, px1, py1, k, fx0, fy0, z) in st["geo"]:
                    if k == m["k"]:
                        b = m["box"]
                        cv2.rectangle(canvas, (int(px0 + (b[0] - fx0) * z), int(py0 + (b[1] - fy0) * z)),
                                      (int(px0 + (b[2] - fx0) * z), int(py0 + (b[3] - fy0) * z)),
                                      (255, 255, 0) if m is not st["drawn"] else (0, 255, 0), 1)
            if st["down"] and st["cur"]:
                cv2.rectangle(canvas, st["down"], st["cur"], (0, 255, 0), 1)
            done = sum(1 for dd in docs.values() for tt in dd["tracks"] if tt["decision"])
            dur = (t["k"][-1] - t["k"][0]) / 30
            votes = " ".join(f"{k}:{v}" for k, v in sorted(t["cls_votes"].items(), key=lambda x: -x[1]))
            mine = f"  your boxes: {len(t['manual'])}" if t["manual"] else ""
            cv2.putText(canvas, f"[{done}/{n_all}] {s} track {t['id']}: {t['n']} dets over {dur:.1f}s, max conf {t['max_conf']:.2f}, model says {votes}{mine}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            help_ = ("BOX DRAWN: u / p / b = label it as drone / airplane / bird,  c = clear" if st["drawn"] else
                     "TRACK: u=drone p=airplane b=bird f=false alarm x=unsure   drag=draw a box (r=remove last)   space=skip z=back q=quit")
            cv2.putText(canvas, help_, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if st["drawn"] else (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("tracks", canvas)
            c = cv2.waitKey(30) & 0xFF
            if c == 255:
                continue
            if st["drawn"]:
                if c in (ord("u"), ord("p"), ord("b")):
                    t["manual"].append({**st["drawn"], "cls": KEYS[c]}); st["drawn"] = None
                elif c == ord("c"):
                    st["drawn"] = None
                continue
            if c in KEYS:
                t["decision"] = KEYS[c]; hist.append(j); j += 1; moved = True
            elif c == ord("r") and t["manual"]:
                t["manual"].pop()
            elif c == ord(" "):
                hist.append(j); j += 1; moved = True
            elif c == ord("z") and hist:
                j = hist.pop(); docs[todo[j][0]]["tracks"][todo[j][1]]["decision"] = ""; moved = True
            elif c == ord("q"):
                save(); print("saved"); return
        if len(hist) % 25 == 0:
            save()
    save(); print("all tracks decided; run mine_rgb_tracks.py export")


if __name__ == "__main__":
    main()

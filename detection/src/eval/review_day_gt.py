r"""
Ground-truth review for the 4K DAY camera of a .ts recording, one frame per timestamp (pipeline_eval convention:
frames are identified by pts_time, never by a count).

Window (fits ~1200x520): left = the whole frame downscaled, right = a zoom on the 4K pixels. Candidates (model
detections at that timestamp, numbered) are drawn on both; green = the current box of this frame.

Keys
  Space / k / Enter  keep the current decision (a pre-filled box becomes a confirmed drone) and go on
                 (on a frame with no box and no decision it does nothing: press n for "nothing")
  1..9           candidate k is the drone              n   nothing in this frame (empty)
  u              unsure / difficult (keeps the box if there is one: too small / ambiguous to demand)
  b              the current box is a bird          o   the current box is an airplane / other object
  x              clear the box                      a / d   previous / next frame without changing it
  + / -          zoom in / out                     c   toggle candidates            q   save and quit
Mouse: click the overview to move the zoom there; drag in the zoom to draw a box (4K pixels).
A search circle (constant velocity from the nearest confirmed drone boxes, as gt_boxes --gate) is drawn in red,
and the zoom centres on it when the frame has no box.

Every decision is written to --out at once (CSV: t,verdict,cls,x0,y0,x1,y1,difficult,source), so you can quit and
resume. --only: unlabelled (default: frames without a human decision), all, prefill (frames whose box is still the
machine pre-fill).

    python3 src/eval/review_day_gt.py --video ../data/arsuf_1005/day_output.ts \
        --frames ../data/arsuf_1005/gt_day/frames.csv --dets ../data/arsuf_1005/day_rgbcurrent.json \
        --out ../data/arsuf_1005/gt_day/boxes.csv
"""
import os
import sys
import csv
import json
import bisect
import argparse
import threading
from collections import OrderedDict

import cv2
import numpy as np

sys.path.insert(0, "/shared/interceptor/tools/pipeline_eval")
from frames import FrameExtractor   # noqa: E402

OV_W, OV_H, ZW = 800, 450, 420
COLS = {0: (0, 220, 255), 1: (255, 255, 0), 2: (255, 0, 255)}     # uav yellow, airplane cyan, bird magenta (BGR)
FIELDS = ["t", "verdict", "cls", "x0", "y0", "x1", "y1", "difficult", "source"]


class Frames:
    """Exact 4K frame by timestamp (interceptor FrameExtractor decoding), small in-memory LRU, prefetch."""
    def __init__(self, path, cache_dir, keep=24):
        self.fx, self.path, self.keep = FrameExtractor(cache_dir), path, keep
        self.mem, self.lock = OrderedDict(), threading.Lock()
        self.idx = self.fx.index(path)

    def _load(self, t):
        tt = self.idx.nearest(t)
        good = self.fx._good_keys.setdefault(self.path, set())
        cands = [k for k in self.idx.keys if k[0] <= tt + 1e-6][::-1]
        for kt, pos in ([k for k in cands if k[1] in good] + [k for k in cands if k[1] not in good])[:12]:
            img, got = self.fx._decode_at(self.path, pos, tt)
            if got is not None and abs(got - tt) < 0.0015:
                good.add(pos)
                return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return None

    def get(self, t):
        key = round(t, 6)
        with self.lock:
            if key in self.mem:
                self.mem.move_to_end(key)
                return self.mem[key]
        im = self._load(t)
        with self.lock:
            self.mem[key] = im
            while len(self.mem) > self.keep:
                self.mem.popitem(last=False)
        return im

    def prefetch(self, ts):
        def run():
            for t in ts:
                with self.lock:
                    if round(t, 6) in self.mem:
                        continue
                self.get(t)
        threading.Thread(target=run, daemon=True).start()


def load_rows(path):
    rows = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            rows[round(float(r["t"]), 6)] = r
    return rows


def save_rows(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader()
        for t in sorted(rows):
            w.writerow({k: rows[t].get(k, "") for k in FIELDS})
    os.replace(tmp, path)


def gate(rows, ts, t, max_s=3.0):
    """(cx, cy, r) from the nearest confirmed drone boxes before/after t (human decisions only)."""
    conf = sorted(x for x in rows if rows[x]["verdict"] == "drone" and rows[x]["source"] == "human" and x != t)
    i = bisect.bisect_left(conf, t)
    nb = [conf[j] for j in (i - 2, i - 1, i, i + 1) if 0 <= j < len(conf) and abs(conf[j] - t) <= max_s]
    if not nb:
        return None
    def c(x):
        r = rows[x]; return np.array([(float(r["x0"]) + float(r["x1"])) / 2, (float(r["y0"]) + float(r["y1"])) / 2])
    before = [x for x in nb if x < t]; after = [x for x in nb if x > t]
    if before and after:
        a, b = before[-1], after[0]; w = (t - a) / (b - a); p = c(a) * (1 - w) + c(b) * w; dt = min(t - a, b - t)
    elif len(before) >= 2 or len(after) >= 2:
        a, b = (before[-2], before[-1]) if len(before) >= 2 else (after[0], after[1])
        v = (c(b) - c(a)) / (b - a); ref = b if before else a; p = c(ref) + v * (t - ref); dt = abs(t - ref)
    else:
        ref = (before or after)[0]; p = c(ref); dt = abs(t - ref)
    return float(p[0]), float(p[1]), 60 + 400 * dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--frames", required=True, help="CSV with column t")
    ap.add_argument("--dets", required=True, help="detect_ts.py JSON of the day video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="unlabelled", choices=["unlabelled", "all", "prefill"])
    ap.add_argument("--start", type=float, default=None, help="start at this t (seconds, file timeline)")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/pipeline_eval"))
    a = ap.parse_args()
    ts = [round(float(r["t"]), 6) for r in csv.DictReader(open(a.frames))]
    D = json.load(open(a.dets)); dets = {round(f["t"], 6): f["dets"] for f in D["frames"]}
    rows = load_rows(a.out)
    def todo_ok(t):
        r = rows.get(t)
        if a.only == "all":
            return True
        if a.only == "prefill":
            return r is not None and r["source"].startswith("prefill")
        return r is None or r["source"].startswith("prefill")
    order = [t for t in ts if todo_ok(t)]
    if a.start is not None:
        order = [t for t in order if t >= a.start - 1e-6]
    if not order:
        print("nothing to review"); return
    print(f"{len(order)} frames to review ({a.only}); {sum(1 for r in rows.values() if r['source'] == 'human')} "
          f"human decisions so far")
    fr = Frames(a.video, a.cache_dir)
    st = {"i": 0, "zoom": 160, "center": None, "drag": None, "cands": True}
    cv2.namedWindow("day GT", cv2.WINDOW_AUTOSIZE)

    def cur():
        return order[st["i"]]

    def cands(t):
        return sorted([d for d in dets.get(t, []) if d[4] >= 0.05], key=lambda d: -d[4])[:9]

    def box_of(t):
        r = rows.get(t)
        if r and r["x0"] not in ("", None):
            return [float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])]
        return None

    def zoom_rect(im, t):
        H, W = im.shape[:2]
        b = box_of(t)
        if st["center"] is not None:
            cx, cy = st["center"]
        elif b:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        else:
            g = gate(rows, ts, t)
            if g:
                cx, cy = g[0], g[1]
            elif cands(t):
                c0 = cands(t)[0]; cx, cy = (c0[0] + c0[2]) / 2, (c0[1] + c0[3]) / 2
            else:
                cx, cy = W / 2, H / 2
        s = st["zoom"]
        x0 = int(np.clip(cx - s / 2, 0, W - s)); y0 = int(np.clip(cy - s / 2, 0, H - s))
        return x0, y0, s

    def draw():
        t = cur(); im = fr.get(t)
        if im is None:
            canvas = np.zeros((OV_H + 70, OV_W + ZW, 3), np.uint8)
            cv2.putText(canvas, f"frame t={t} could not be decoded (d = next)", (20, 40), 0, 0.7, (0, 0, 255), 2)
            cv2.imshow("day GT", canvas); return None
        H, W = im.shape[:2]; sx = OV_W / W
        ov = cv2.resize(im, (OV_W, OV_H), interpolation=cv2.INTER_AREA)
        zx, zy, s = zoom_rect(im, t)
        zm = cv2.resize(im[zy:zy + s, zx:zx + s], (ZW, ZW), interpolation=cv2.INTER_NEAREST if s < ZW else cv2.INTER_AREA)
        zs = ZW / s
        cv2.rectangle(ov, (int(zx * sx), int(zy * sx)), (int((zx + s) * sx), int((zy + s) * sx)), (200, 200, 200), 1)
        g = gate(rows, ts, t)
        if g:
            cv2.circle(ov, (int(g[0] * sx), int(g[1] * sx)), max(4, int(g[2] * sx)), (0, 0, 255), 1)
            cv2.circle(zm, (int((g[0] - zx) * zs), int((g[1] - zy) * zs)), int(g[2] * zs), (0, 0, 255), 1)
        if st["cands"]:
            for k, d in enumerate(cands(t), 1):
                col = COLS.get(d[5], (200, 200, 200))
                cv2.rectangle(ov, (int(d[0] * sx) - 3, int(d[1] * sx) - 3), (int(d[2] * sx) + 3, int(d[3] * sx) + 3), col, 1)
                cv2.putText(ov, str(k), (int(d[2] * sx) + 4, int(d[1] * sx) + 2), 0, 0.4, col, 1)
                cv2.rectangle(zm, (int((d[0] - zx) * zs) - 2, int((d[1] - zy) * zs) - 2),
                              (int((d[2] - zx) * zs) + 2, int((d[3] - zy) * zs) + 2), col, 1)
                cv2.putText(zm, f"{k} {D['names'].get(str(d[5]), d[5])} {d[4]:.2f}",
                            (int((d[0] - zx) * zs), int((d[1] - zy) * zs) - 5), 0, 0.4, col, 1)
        b = box_of(t); r = rows.get(t)
        if b:
            cv2.rectangle(ov, (int(b[0] * sx) - 4, int(b[1] * sx) - 4), (int(b[2] * sx) + 4, int(b[3] * sx) + 4), (0, 255, 0), 1)
            cv2.rectangle(zm, (int((b[0] - zx) * zs), int((b[1] - zy) * zs)), (int((b[2] - zx) * zs), int((b[3] - zy) * zs)),
                          (0, 255, 0), 1)
        if st["drag"] and st["drag"][1]:
            (x0, y0), (x1, y1) = st["drag"]
            cv2.rectangle(zm, (x0, y0), (x1, y1), (0, 255, 0), 1)
        canvas = np.zeros((OV_H + 70, OV_W + ZW, 3), np.uint8)
        canvas[:OV_H, :OV_W] = ov; canvas[:ZW, OV_W:] = zm
        dec = "-" if r is None else f"{r['verdict']}{' ' + r['cls'] if r.get('cls') else ''}" \
              f"{' (difficult)' if r.get('difficult') == '1' else ''} [{r['source']}]"
        done = sum(1 for x in rows.values() if x["source"] == "human")
        lines = [f"{st['i'] + 1}/{len(order)}  t={t:.3f}  now: {dec}   human-reviewed {done}/{len(ts)}   zoom {s}px",
                 "Space or k keep | 1-9 candidate | n nothing | u unsure | b bird | o other | x clear | a/d prev/next | "
                 "+/- zoom | c cands | q quit"]
        for j, l in enumerate(lines):
            cv2.putText(canvas, l, (8, OV_H + 22 + 22 * j), 0, 0.45, (230, 230, 230), 1)
        cv2.imshow("day GT", canvas)
        fr.prefetch(order[st["i"] + 1: st["i"] + 6])
        return (zx, zy, s, sx)

    geo = {}

    def on_mouse(ev, x, y, flags, _):
        g = geo.get("g")
        if not g:
            return
        zx, zy, s, sx = g
        if x < OV_W and y < OV_H and ev == cv2.EVENT_LBUTTONDOWN:
            st["center"] = (x / sx, y / sx); refresh()
        elif x >= OV_W and y < ZW:
            p = (x - OV_W, y)
            if ev == cv2.EVENT_LBUTTONDOWN:
                st["drag"] = [p, None]
            elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
                st["drag"][1] = p; refresh()
            elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
                (x0, y0), p1 = st["drag"][0], (p if st["drag"][1] is None else st["drag"][1])
                st["drag"] = None
                zs = ZW / s
                bx = sorted([zx + x0 / zs, zx + p1[0] / zs]); by = sorted([zy + y0 / zs, zy + p1[1] / zs])
                if bx[1] - bx[0] >= 2 and by[1] - by[0] >= 2:
                    rows[cur()] = {"t": f"{cur():.6f}", "verdict": "drone", "cls": "uav",
                                   "x0": f"{bx[0]:.1f}", "y0": f"{by[0]:.1f}", "x1": f"{bx[1]:.1f}", "y1": f"{by[1]:.1f}",
                                   "difficult": "0", "source": "human"}
                    save_rows(a.out, rows)
                refresh()

    def refresh():
        geo["g"] = draw()

    cv2.setMouseCallback("day GT", on_mouse)
    refresh()
    while True:
        k = cv2.waitKey(0) & 0xFF
        t = cur(); r = rows.get(t); b = box_of(t); moved = False
        def decide(verdict, cls="", box=None, difficult="0"):
            row = {"t": f"{t:.6f}", "verdict": verdict, "cls": cls, "difficult": difficult, "source": "human",
                   "x0": "", "y0": "", "x1": "", "y1": ""}
            if box is not None:
                row.update({"x0": f"{box[0]:.1f}", "y0": f"{box[1]:.1f}", "x1": f"{box[2]:.1f}", "y1": f"{box[3]:.1f}"})
            rows[t] = row; save_rows(a.out, rows)
        if k == ord("q"):
            save_rows(a.out, rows); break
        elif k in (13, 32, ord("k")):
            if r is not None:
                decide(r["verdict"], r.get("cls", "") or ("uav" if r["verdict"] == "drone" else ""), b, r.get("difficult", "0"))
                moved = True
        elif ord("1") <= k <= ord("9"):
            c = cands(t)
            if k - ord("1") < len(c):
                decide("drone", "uav", c[k - ord("1")][:4]); moved = True
        elif k == ord("n"):
            decide("nothing"); moved = True
        elif k == ord("u"):
            decide("unsure", "uav" if b else "", b, "1"); moved = True
        elif k == ord("b") and b:
            decide("bird_or_other", "bird", b); moved = True
        elif k == ord("o") and b:
            decide("bird_or_other", "airplane", b); moved = True
        elif k == ord("x"):
            rows.pop(t, None); save_rows(a.out, rows)
        elif k == ord("d"):
            moved = True
        elif k == ord("a"):
            st["i"] = max(0, st["i"] - 1); st["center"] = None
        elif k in (ord("+"), ord("=")):
            st["zoom"] = max(40, int(st["zoom"] / 1.5))
        elif k in (ord("-"), ord("_")):
            st["zoom"] = min(2160, int(st["zoom"] * 1.5))
        elif k == ord("c"):
            st["cands"] = not st["cands"]
        if moved:
            if st["i"] + 1 >= len(order):
                save_rows(a.out, rows); print("end of list"); break
            st["i"] += 1; st["center"] = None
        refresh()
    cv2.destroyAllWindows()
    print(f"saved {a.out}: {sum(1 for x in rows.values() if x['source'] == 'human')} human decisions")


if __name__ == "__main__":
    main()

"""
Box-level ground truth for the recall eval set, from MODEL-INDEPENDENT proposals.

Why: compute_recall_from_gt.py counted a positive frame as "caught" if the model
fired anywhere in the frame. A model that fires everywhere gets recall for free
(p2_noleak_synthA: 94.6% recall with 79% false-fire). With one drone box per
positive frame, a hit must land on the drone.

Fairness rule: proposals must not come from any model being scored (its own
guesses would become the ground truth it is measured against). They come from
a classical small-target detector only:
  * temporal saliency: |frame - median of +-2/4/6 neighbours|, neighbours first
    aligned to the frame by global phase correlation (cancels camera pan),
  * spatial saliency: white + black top-hat (small bright or dark blobs).

The same tool audits EVERY frame labelled empty (not only frames where some model
fired): tiny airborne objects were found in "empty" frames (notebook 2026-09-29).

Stages:
  sample       (new sessions) every N-th frame -> unlabelled manifest with a contiguous train/val split
  propose      one sequential pass over the video -> proposals.json (top candidates per frame)
  review       keypress UI, writes boxes.csv (resumable)
  export-yolo  reviewed frames -> YOLO images/labels at native resolution

Review keys (window "GT boxes"):
  positive frames (labelled TP):
    1-6    the numbered candidate is the drone
    click  on the drone (either panel) -> polarity-aware auto-fit; if it cannot fit, it asks for a drag
    drag   press-drag-release around the drone -> exactly that box
           (after click/drag: Enter accept, +/- grow/shrink, drag again to redraw, Esc cancel)
    n      no drone visible in this frame (label error)
  empty frames (labelled FP), all of them:
    Enter  nothing airborne (stays empty)          -- the common case
    d      a DRONE is present: then 1-6 or click to give its box
    w      a BIRD / other airborne object: then 1-6 or click
  unlabelled frames (new sessions):
    1-6 / click  drone     Enter  nothing     w  bird / other airborne object
  all:
    With --registration/--day, the synced day frame is shown next to the IR, WARPED INTO THE IR GEOMETRY
    (same pixel coordinates: candidate boxes are drawn on both; clicking either panel works). Frames outside
    a verified registration segment use the nearest segment and are marked "approx".
    t      switch the zoom tiles between IR and full-resolution RGB (same area, ~7x more day pixels)
    p      play the zoomed region over +-12 frames (motion tells drone from speck)
    v      show the matching 4K day-camera crop (full resolution)
    s      skip / unsure     b  back     q  save and quit

    python3 src/eval/gt_boxes.py propose --video data/videos/thermal.mp4 \
        --manifest data/recall_ground_truth/manifest.csv --out data/recall_ground_truth/proposals.json
    python3 src/eval/gt_boxes.py review --video data/videos/thermal.mp4 \
        --manifest data/recall_ground_truth/manifest.csv --proposals data/recall_ground_truth/proposals.json \
        --out data/recall_ground_truth/boxes.csv [--registration ../data/vid_pairs/registration.json --day data/videos/day.mp4]
"""

import os
import csv
import sys
import json
import argparse

import cv2
import numpy as np
import pandas as pd

NEIGH = (-6, -4, -2, 2, 4, 6)
K = 6                      # candidates kept per frame
DISP = 2                   # full-frame display scale factor is 1 (640x512); zoom tiles are 4x


# ------------------------------------------------------------------ propose
def highpass(g):
    return g - cv2.GaussianBlur(g, (0, 0), 3)


def saliency(frame, neighbours):
    """Temporal (camera-motion compensated) + spatial small-target saliency."""
    h, w = frame.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    hf = highpass(frame)
    aligned = []
    for n in neighbours:
        (dx, dy), _ = cv2.phaseCorrelate(highpass(n), hf, win)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned.append(cv2.warpAffine(n, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    bg = np.median(np.stack(aligned), 0)
    temporal = cv2.GaussianBlur(np.abs(frame - bg), (0, 0), 1.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    spatial = np.maximum(frame - cv2.morphologyEx(frame, cv2.MORPH_OPEN, k),
                         cv2.morphologyEx(frame, cv2.MORPH_CLOSE, k) - frame)
    spatial = cv2.GaussianBlur(spatial, (0, 0), 1.0)
    return clutter_norm(temporal), clutter_norm(spatial)


def clutter_norm(x, win=41):
    """Response relative to LOCAL clutter (rms over a win x win neighbourhood), with a global floor.
    Without it, textured terrain outranks a drone in smooth sky (the drone was in the top-6 in only
    31-44% of positive frames); with it, a blob in flat sky scores high and the same blob in
    vegetation scores low -- the classic local-contrast small-target measure."""
    local = np.sqrt(cv2.boxFilter(x * x, -1, (win, win)))
    floor = 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-3
    return x / (local + floor)


def fit_box(score, x, y, win=24):
    """Connected region >= 50% of the peak around (x, y) -> (x0, y0, x1, y1), 1 px margin."""
    h, w = score.shape
    X0, Y0, X1, Y1 = max(x - win, 0), max(y - win, 0), min(x + win + 1, w), min(y + win + 1, h)
    sub = score[Y0:Y1, X0:X1]
    peak = score[y, x]
    mask = (sub >= 0.5 * peak).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    comp = lab == lab[y - Y0, x - X0]
    ys, xs = np.nonzero(comp)
    x0, y0, x1, y1 = X0 + xs.min() - 1, Y0 + ys.min() - 1, X0 + xs.max() + 2, Y0 + ys.max() + 2
    if x1 - x0 < 3:
        x0, x1 = x - 1, x + 2
    if y1 - y0 < 3:
        y0, y1 = y - 1, y + 2
    return [int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h))]


def fit_click(g, x, y, max_half=24):
    """Box for a clicked target, polarity-aware (dark drone over hot terrain OR bright drone on sky).
    Snap to the strongest local-contrast pixel within 6 px, estimate the local background from a ring,
    then grow the >= 45%-of-peak component in windows of increasing size. Returns None when the component
    still touches the window edge at the largest size (target bleeds into clutter): the caller asks for a
    drawn box instead of saving a capped one (the old fit silently saved 51x51 boxes there)."""
    h, w = g.shape
    loc = cv2.medianBlur(g.astype(np.uint8), 15).astype(np.float32)
    c = g - loc
    X0, Y0, X1, Y1 = max(x - 6, 0), max(y - 6, 0), min(x + 7, w), min(y + 7, h)
    sub = np.abs(c[Y0:Y1, X0:X1])
    dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
    sx, sy = X0 + dx, Y0 + dy
    pol = 1.0 if c[sy, sx] >= 0 else -1.0
    ring_in, ring_out = 10, 18
    yy, xx = np.mgrid[max(sy - ring_out, 0):min(sy + ring_out + 1, h), max(sx - ring_out, 0):min(sx + ring_out + 1, w)]
    rr = np.hypot(yy - sy, xx - sx)
    bg = float(np.median(g[yy[(rr >= ring_in) & (rr <= ring_out)], xx[(rr >= ring_in) & (rr <= ring_out)]]))
    contrast = cv2.GaussianBlur(pol * (g - bg), (0, 0), 0.7)
    for half in (6, 10, 16, max_half):
        X0, Y0, X1, Y1 = max(sx - half, 0), max(sy - half, 0), min(sx + half + 1, w), min(sy + half + 1, h)
        win = contrast[Y0:Y1, X0:X1]
        py0, px0 = sy - Y0, sx - X0
        near = win[max(py0 - 2, 0):py0 + 3, max(px0 - 2, 0):px0 + 3]
        peak = float(near.max())
        if peak <= 0:
            return None
        mask = (win >= 0.45 * peak).astype(np.uint8)
        n, lab = cv2.connectedComponents(mask, connectivity=8)
        ny, nx = np.unravel_index(np.argmax(near), near.shape)
        comp = lab == lab[max(py0 - 2, 0) + ny, max(px0 - 2, 0) + nx]
        ys, xs = np.nonzero(comp)
        touches = xs.min() == 0 or ys.min() == 0 or xs.max() == win.shape[1] - 1 or ys.max() == win.shape[0] - 1
        if not touches:
            return [int(max(X0 + xs.min() - 1, 0)), int(max(Y0 + ys.min() - 1, 0)),
                    int(min(X0 + xs.max() + 2, w)), int(min(Y0 + ys.max() + 2, h))]
    return None


def candidates(frame, neighbours, border=4):
    zt, zs = saliency(frame, neighbours)
    S = np.maximum(zt, 0) + np.maximum(zs, 0)
    S[:border] = S[-border:] = 0
    S[:, :border] = S[:, -border:] = 0
    peaks = (S == cv2.dilate(S, np.ones((11, 11), np.uint8))) & (S > 1.0)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-S[ys, xs])
    out = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if any(abs(x - c["x"]) < 10 and abs(y - c["y"]) < 10 for c in out):
            continue
        out.append({"x": x, "y": y, "score": float(S[y, x]), "z_temporal": float(zt[y, x]),
                    "z_spatial": float(zs[y, x]), "box": fit_box(S, x, y)})
        if len(out) >= K:
            break
    return out


def cmd_propose(a):
    m = pd.read_csv(a.manifest)
    want = set(int(f) for f in m.frame_id)
    need = set(f + d for f in want for d in (0,) + NEIGH)
    cap = cv2.VideoCapture(a.video)
    buf, props, idx = {}, {}, 0
    last = max(need)
    while idx <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in need:
            buf[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = idx - max(NEIGH)                  # all neighbours of f are now read
        if f in want and f in buf:
            nb = [buf[f + d] for d in NEIGH if f + d in buf]
            props[f] = candidates(buf[f], nb)
        for k in [k for k in buf if k < idx - 2 * max(NEIGH) and k not in want]:
            del buf[k]
        idx += 1
        if idx % 2000 == 0:
            print(f"  frame {idx}  proposals for {len(props)}/{len(want)} GT frames", flush=True)
    for f in sorted(want - set(props)):          # last frames: use whatever neighbours exist
        if f in buf:
            nb = [buf[f + d] for d in NEIGH if f + d in buf]
            if nb:
                props[f] = candidates(buf[f], nb)
    with open(a.out, "w") as f:
        json.dump({"video": a.video, "method": "motion-compensated temporal median + top-hat (model-free)",
                   "neighbours": NEIGH, "proposals": {str(k): v for k, v in props.items()}}, f)
    print(f"wrote {a.out}: {len(props)} frames")


# ------------------------------------------------------------------ frame access
class FrameReader:
    """Frames by SEQUENTIAL index (the convention of run_yolo_inference.py and `propose`).
    cv2 seeking (CAP_PROP_POS_FRAMES) is not frame-exact on every file: on the variable-frame-rate
    2026-07-30 recording it lands tens of frames away, and on 2026-06-23 it is off by one at some frames
    (a probe of a few frames does not catch that). So the needed frames are always decoded sequentially
    once and cached; any other frame (e.g. the +-12 frames of 'p') falls back to seeking (approximate)."""

    def __init__(self, video, needed=()):
        self.video = video
        self.cap = cv2.VideoCapture(video)
        self.cache = {}
        want = set(int(f) for f in needed)
        if want:
            print(f"  decoding {len(want)} frames of {os.path.basename(video)} sequentially ...", flush=True)
            cap, i, last = cv2.VideoCapture(video), 0, max(want)
            while i <= last:
                ok, im = cap.read()
                if not ok:
                    break
                if i in want:
                    self.cache[i] = im
                i += 1

    def read(self, fi):
        if fi in self.cache:
            return self.cache[fi]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, im = self.cap.read()
        return im if ok else None


# ------------------------------------------------------------------ review
class Day:
    """Optional 4K day-camera crop for a thermal point, via video_pairs registration."""

    def __init__(self, registration, day_video):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
        from day2thermal.video_pairs import und_from_dist, day_index, build_rgb_maps
        self.und, self.day_index, self.build_maps = und_from_dist, day_index, build_rgb_maps
        self.reg = json.load(open(registration))
        self.cap = cv2.VideoCapture(day_video)
        self.segs = [g for g in self.reg["segments"] if g["verified"]]
        self.maps = {}

    def _day_frame(self, ti):
        if getattr(self, "_cache", (None, None))[0] == ti:
            return self._cache[1]
        di = self.day_index(ti, self.reg["fps_thermal"], self.reg["fps_day"], self.reg["offset_ms"] / 1000,
                            self.reg.get("drift_ms_per_s", 0.0) / 1000)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, di)
        ok, im = self.cap.read()
        self._cache = (ti, im if ok else None)
        return self._cache[1]

    def _segment(self, ti):
        inside = [g for g in self.segs if g["span"][0] <= ti <= g["span"][1]]
        if inside:
            return inside[0], True
        return min(self.segs, key=lambda g: min(abs(ti - g["span"][0]), abs(ti - g["span"][1]))), False

    def zoom_tile(self, ti, x, y, thermal_half=16, size=192):
        """Full-resolution day crop covering the same area as the IR zoom tile around thermal (x, y)."""
        if not self.segs:
            return None
        g, _ = self._segment(ti)
        H = np.array(g["H_day_to_undistorted_thermal"])
        xu, yu = self.und(np.array([float(x)]), np.array([float(y)]), self.reg["lens"]["lambda"],
                          tuple(self.reg["thermal_size"]))
        p = np.linalg.inv(H) @ np.array([xu[0], yu[0], 1.0])
        px, py = p[0] / p[2], p[1] / p[2]
        half = int(round(thermal_half / np.sqrt(abs(np.linalg.det(H[:2, :2])))))   # thermal px -> day px
        im = self._day_frame(ti)
        W, Hh = self.reg["rgb_size"]
        if im is None or not (0 <= px < W and 0 <= py < Hh):
            return None
        pad = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        c = pad[int(py):int(py) + 2 * half, int(px):int(px) + 2 * half]
        return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)

    def registered(self, ti):
        """The synced day frame warped into the thermal frame's geometry (same pixel coordinates as the IR).
        Uses the verified segment containing ti; otherwise the nearest one (status 'approx')."""
        if not self.segs:
            return None, "no verified registration segment"
        g, inside = self._segment(ti)
        status = f"seg {g['id']}" if inside else f"approx (nearest seg {g['id']})"
        if g["id"] not in self.maps:
            ds = self.reg["day_decode_downscale"]
            mx, my, _ = self.build_maps(np.array(g["H_day_to_undistorted_thermal"]), self.reg["lens"]["lambda"],
                                        tuple(self.reg["thermal_size"]), tuple(self.reg["rgb_size"]), ds)
            self.maps[g["id"]] = (mx, my, ds)
        mx, my, ds = self.maps[g["id"]]
        im = self._day_frame(ti)
        if im is None:
            return None, "day frame unreadable"
        W, H = self.reg["rgb_size"]
        small = cv2.resize(im, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        out = cv2.remap(small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        return out, status

    def crop(self, ti, x, y, half=160):
        segs = [g for g in self.reg["segments"] if g["verified"] and g["span"][0] <= ti <= g["span"][1]]
        if not segs:
            return None, "frame not in a verified registration segment"
        Hi = np.linalg.inv(np.array(segs[0]["H_day_to_undistorted_thermal"]))
        xu, yu = self.und(np.array([float(x)]), np.array([float(y)]), self.reg["lens"]["lambda"],
                          tuple(self.reg["thermal_size"]))
        p = Hi @ np.array([xu[0], yu[0], 1.0])
        px, py = int(p[0] / p[2]), int(p[1] / p[2])
        W, H = self.reg["rgb_size"]
        if not (0 <= px < W and 0 <= py < H):
            return None, "point outside the day camera's field of view"
        im = self._day_frame(ti)
        if im is None:
            return None, "day frame unreadable"
        c = im[max(py - half, 0):py + half, max(px - half, 0):px + half].copy()
        cv2.circle(c, (min(half, px), min(half, py)), 40, (0, 0, 255), 1)
        return cv2.resize(c, (480, 480), interpolation=cv2.INTER_NEAREST), "ok"


def zoom(frame_bgr, cx, cy, half=16, scale=6):
    h, w = frame_bgr.shape[:2]
    x0, y0 = int(np.clip(cx - half, 0, w - 2 * half)), int(np.clip(cy - half, 0, h - 2 * half))
    return cv2.resize(frame_bgr[y0:y0 + 2 * half, x0:x0 + 2 * half], (2 * half * scale, 2 * half * scale),
                      interpolation=cv2.INTER_NEAREST), (x0, y0)


COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (255, 255, 0)]


def cmd_review(a):
    m = pd.read_csv(a.manifest)
    if "label" not in m.columns:
        m["label"] = ""
    m["label"] = m["label"].fillna("")
    m = m[m.label.isin(["TP", "TP_loose", "FP", ""])]          # "" = unlabelled (new sessions)
    props = json.load(open(a.proposals))["proposals"]
    done = {}
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done[int(r["frame_id"])] = r
    todo = [int(f) for f in m.sort_values("frame_id").frame_id]
    def is_capped(r):
        try:
            return (r["verdict"] in ("drone", "bird_or_other") and
                    (float(r["x1"]) - float(r["x0"]) >= 50 or float(r["y1"]) - float(r["y0"]) >= 50))
        except (TypeError, ValueError):
            return False

    # suspects: 'nothing' next to a drone frame in sampled order (click-then-Enter bug signature, or drone
    # briefly out of view -- the reviewer decides)
    order = sorted(done)
    suspect = set()
    for j, f in enumerate(order):
        if done[f]["verdict"] == "nothing":
            nb = [order[j - 1]] if j > 0 else []
            nb += [order[j + 1]] if j + 1 < len(order) else []
            if any(done[n]["verdict"] == "drone" for n in nb):
                suspect.add(f)
    if a.verify:
        cats = set(a.only.split(","))
        def want(f):
            r = done.get(f)
            if r is None:
                return "unlabelled" in cats or "all" in cats
            return ("all" in cats or (is_capped(r) and "capped" in cats) or r["verdict"] in cats
                    or (f in suspect and "suspect" in cats))
        todo = [f for f in todo if want(f)]
        print(f"--verify: {len(todo)} frames ({a.only}); {len(suspect & set(todo))} flagged SUSPECT")
    if a.redo_capped:
        # frames whose drone box hit the old 51-px auto-fit cap: review them again, nothing else
        todo = [f for f in todo if f in done and done[f]["verdict"] in ("drone", "bird_or_other")
                and (int(float(done[f]["x1"])) - int(float(done[f]["x0"])) >= 50
                     or int(float(done[f]["y1"])) - int(float(done[f]["y0"])) >= 50)]
        print(f"--redo-capped: {len(todo)} frames with a capped box to redo")
    lab = dict(zip(m.frame_id.astype(int), m.label))
    frames = FrameReader(a.video, needed=todo)
    day = Day(a.registration, a.day) if a.registration and a.day else None
    fields = ["frame_id", "orig_label", "verdict", "x0", "y0", "x1", "y1", "source", "verified"]

    def save():
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for k in sorted(done):
                wr.writerow({f: done[k].get(f, "") for f in fields})

    def read(fi):
        return frames.read(fi)

    # mouse: press-release without moving = click (auto-fit), press-drag-release = drawn box.
    # Both panels share one pixel grid, so coordinates are taken modulo 640.
    mouse = {"down": None, "cur": None, "event": None, "last": None}
    n_panels = 2 if day is not None else 1          # IR | registered RGB
    tiles_from = {"src": "IR"}
    MOUSE = 256

    def on_mouse(ev, x, y, flags, param):
        if x >= 640 * n_panels or y >= 512:
            return
        p = (x % 640, min(y, 511))
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse["down"], mouse["cur"], mouse["panel"] = p, p, x // 640
        elif ev == cv2.EVENT_MOUSEMOVE and mouse["down"] is not None:
            mouse["cur"] = p
        elif ev == cv2.EVENT_LBUTTONUP and mouse["down"] is not None:
            (x0, y0), (x1, y1) = mouse["down"], p
            if abs(x1 - x0) > 3 or abs(y1 - y0) > 3:
                mouse["event"] = ("drag", [min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1])
            else:
                mouse["event"] = ("click", p)
            mouse["last"] = p
            mouse["down"] = mouse["cur"] = None

    def wait_input(base):
        """Wait for a key or a completed mouse gesture; draws the rubber band while dragging."""
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k != 255:
                return k
            if mouse["event"] is not None:
                return MOUSE
            if mouse["down"] is not None:
                show = base.copy()
                (x0, y0), (x1, y1) = mouse["down"], mouse["cur"]
                for off in range(n_panels):
                    cv2.rectangle(show, (x0 + 640 * off, y0), (x1 + 640 * off, y1), (255, 255, 255), 1)
                cv2.imshow("GT boxes", show)

    cv2.namedWindow("GT boxes")
    cv2.setMouseCallback("GT boxes", on_mouse)
    i = 0 if (a.redo_capped or a.verify) else next((k for k, f in enumerate(todo) if f not in done), len(todo))
    while 0 <= i < len(todo):
        fi = todo[i]
        is_pos = lab[fi] in ("TP", "TP_loose")
        is_new = lab[fi] == ""
        im = read(fi)
        cands = props.get(str(fi), [])
        prevrow = done.get(fi) if a.verify else None
        prevbox = None
        if prevrow is not None and prevrow.get("x0", "") not in ("", None):
            try:
                prevbox = [int(float(prevrow[k])) for k in ("x0", "y0", "x1", "y1")]
            except ValueError:
                prevbox = None
        if a.verify:
            cands = cands[:5]                              # tile 6 shows the previous box area
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        vis = im.copy()
        rgb, rgb_status = day.registered(fi) if day is not None else (None, "")
        rvis = rgb.copy() if rgb is not None else None
        tiles = []
        for n, c in enumerate(cands):
            x0, y0, x1, y1 = c["box"]
            for v in (vis, rvis):
                if v is None:
                    continue
                cv2.rectangle(v, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), COLORS[n], 1)
                cv2.putText(v, str(n + 1), (x1 + 4, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[n], 1)
            z = None
            if tiles_from["src"] == "RGB" and day is not None:
                z = day.zoom_tile(fi, c["x"], c["y"])        # full-res day pixels, same area as the IR tile
            if z is None:
                z, _ = zoom(im, c["x"], c["y"])
            cv2.putText(z, f"{n + 1} {tiles_from['src'] if day is not None else 'IR'}  t{c['z_temporal']:.0f} "
                           f"s{c['z_spatial']:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[n], 1)
            tiles.append(z)
        while len(tiles) < K:
            tiles.append(np.zeros((192, 192, 3), np.uint8))
        if prevbox is not None:
            pcx, pcy = (prevbox[0] + prevbox[2]) // 2, (prevbox[1] + prevbox[3]) // 2
            for v in (vis, rvis):
                if v is not None:
                    cv2.rectangle(v, (prevbox[0] - 1, prevbox[1] - 1), (prevbox[2], prevbox[3]), (255, 255, 255), 2)
            z = day.zoom_tile(fi, pcx, pcy) if (tiles_from["src"] == "RGB" and day is not None) else None
            if z is None:
                z, (zx0, zy0) = zoom(im, pcx, pcy)
                s6 = 192 / 32
                cv2.rectangle(z, (int((prevbox[0] - zx0) * s6), int((prevbox[1] - zy0) * s6)),
                              (int((prevbox[2] - zx0) * s6), int((prevbox[3] - zy0) * s6)), (255, 255, 255), 1)
            cv2.putText(z, "prev box", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles[5] = z
        panel = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])  # 576 x 384
        left = 640 * n_panels
        canvas = np.zeros((max(512, 576), left + 384, 3), np.uint8)
        canvas[:512, :640] = vis
        if n_panels == 2:
            if rvis is None:
                cv2.putText(canvas, f"RGB: {rgb_status}", (650, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                canvas[:512, 640:1280] = rvis
                cv2.putText(canvas, f"RGB registered: {rgb_status}", (646, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255) if rgb_status.startswith("seg") else (0, 128, 255), 1)
        canvas[:576, left:] = panel
        prev = done.get(fi, {}).get("verdict", "")
        if a.verify and prevrow is not None:
            flag = ("  CAPPED box: tighten it (drag)" if is_capped(prevrow) else "") + \
                   ("  SUSPECT: next to a drone frame, look carefully" if fi in suspect else "")
            cv2.putText(canvas, f"PREVIOUS: {prevrow['verdict']}{flag}   Enter=keep  n=nothing  1-5/click/drag=drone  w=bird",
                        (6, 556), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 128, 255) if flag else (255, 255, 255), 1)
        head = (f"[{i + 1}/{len(todo)}] frame {fi}  label {lab[fi]}  "
                + ("VERIFY: see line below" if a.verify else
                   "1-6/click/drag=drone  n=no drone" if is_pos else "1-6/click/drag=drone  Enter=nothing  w=bird")
                + "  p=play v=4K t=tiles IR/RGB s=skip b=back q=quit" + (f"   (was: {prev})" if prev else ""))
        cv2.putText(canvas, head, (6, 534), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imshow("GT boxes", canvas)
        mouse["event"] = None
        key = wait_input(canvas)

        def confirm(box, src):
            """Show the box on both panels; Enter accept, Esc cancel, +/- grow/shrink, drag again to redraw."""
            while True:
                show = canvas.copy()
                if box is None:
                    msg = "auto-fit failed (target bleeds into clutter): DRAG a box around it   Esc=cancel"
                else:
                    for off in range(n_panels):
                        cv2.rectangle(show, (box[0] - 1 + 640 * off, box[1] - 1), (box[2] + 640 * off, box[3]),
                                      (255, 255, 255), 1)
                    msg = f"{box[2] - box[0]}x{box[3] - box[1]} px   Enter=accept  +/-=grow/shrink  drag=redraw  Esc=cancel"
                cv2.putText(show, msg, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("GT boxes", show)
                mouse["event"] = None
                k = wait_input(show)
                if k == MOUSE:
                    kind, val = mouse["event"]
                    box, src = (val, "drag") if kind == "drag" else (fit_click(g, *val), "click")
                elif k in (27, ord("q")):
                    return None, None
                elif k in (13, 10) and box is not None:
                    return box, src
                elif k in (ord("+"), ord("=")) and box is not None:
                    box = [max(box[0] - 1, 0), max(box[1] - 1, 0), min(box[2] + 1, 640), min(box[3] + 1, 512)]
                elif k == ord("-") and box is not None and box[2] - box[0] > 3 and box[3] - box[1] > 3:
                    box = [box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1]

        def pick_box(key):
            """(box, source) from a candidate number key, or a mouse click/drag (waits for one if needed)."""
            if ord("1") <= key <= ord("6") and key - ord("1") < len(cands):
                return cands[key - ord("1")]["box"], f"proposal{key - ord('0')}"
            if key != MOUSE:
                mouse["event"] = None
                k2 = wait_input(canvas)
                if ord("1") <= k2 <= ord("6") and k2 - ord("1") < len(cands):
                    return cands[k2 - ord("1")]["box"], f"proposal{k2 - ord('0')}"
                if k2 != MOUSE:
                    return None, None
            kind, val = mouse["event"]
            if kind == "drag":
                return confirm(val, "drag")
            return confirm(fit_click(g, *val), "click")

        if key == ord("q"):
            break
        if key == ord("b"):
            i = max(i - 1, 0)
            continue
        if key == ord("s"):
            done[fi] = {"frame_id": fi, "orig_label": lab[fi], "verdict": "unsure", "x0": "", "y0": "", "x1": "",
                        "y1": "", "source": ""}
            i += 1
            continue
        if key == ord("t"):
            tiles_from["src"] = "RGB" if tiles_from["src"] == "IR" else "IR"
            continue
        if key == ord("p"):
            cx, cy = mouse["last"] or ((cands[0]["x"], cands[0]["y"]) if cands else (320, 256))
            for _ in range(3):
                for d in range(-12, 13):
                    fr = read(fi + d)
                    if fr is None:
                        continue
                    z, _ = zoom(fr, cx, cy, half=32, scale=6)
                    cv2.putText(z, f"{d:+d}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow("play", z)
                    cv2.waitKey(80)
            cv2.destroyWindow("play")
            continue
        if key == ord("v"):
            if day is None:
                print("  (pass --registration and --day to enable the day-camera view)")
                continue
            cx, cy = mouse["last"] or ((cands[0]["x"], cands[0]["y"]) if cands else (320, 256))
            c, msg = day.crop(fi, cx, cy)
            if c is None:
                print(f"  day view: {msg}")
            else:
                cv2.imshow("day camera", c)
                cv2.waitKey(0)
                cv2.destroyWindow("day camera")
            continue
        verdict, box, src = None, None, None
        if a.verify and prevrow is not None and key in (13, 10):
            if is_capped(prevrow):
                box, src = confirm(prevbox, "prev_adjusted")      # capped: must be looked at and tightened
                verdict = prevrow["verdict"] if box else None
            else:
                done[fi] = {**prevrow, "verified": "v2"}
                if len(done) % 10 == 0:
                    save()
                i += 1
                continue
        elif a.verify and key == ord("n"):
            verdict = "no_drone_visible" if is_pos else "nothing"
        elif key == MOUSE or ord("1") <= key <= ord("6") or key == ord("d"):
            box, src = pick_box(key)
            verdict = "drone" if box else None
        elif key == ord("w"):
            box, src = pick_box(0)
            verdict = "bird_or_other" if box else None
        elif key == ord("n") and is_pos:
            verdict = "no_drone_visible"
        elif key in (13, 10) and not is_pos:
            verdict = "nothing"
        if verdict is None:
            continue
        b = box or ["", "", "", ""]
        done[fi] = {"frame_id": fi, "orig_label": lab[fi], "verdict": verdict,
                    "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "source": src or "",
                    "verified": "v2" if (a.verify or a.redo_capped) else ""}
        if len(done) % 10 == 0:
            save()
        i += 1
    save()
    cv2.destroyAllWindows()
    v = pd.Series([r["verdict"] for r in done.values()]).value_counts().to_dict()
    if a.verify or a.redo_capped:
        nv = sum(1 for f in todo if done.get(f, {}).get("verified") == "v2")
        print(f"saved {a.out}: {nv}/{len(todo)} selected frames verified; all labels now {v}")
    else:
        print(f"saved {a.out}: {len(done)}/{len(todo)} frames reviewed {v}")


# ------------------------------------------------------------------ sample / export (new sessions)
def cmd_sample(a):
    """Every --every-th frame of [--start-s, --end-s); the last --val-frac of the range (contiguous,
    after a --gap-s buffer) is the val split. Frames are unlabelled; review them with `review`."""
    cap = cv2.VideoCapture(a.video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    f0 = int(a.start_s * fps)
    f1 = min(n, int(a.end_s * fps)) if a.end_s else n
    frames = list(range(f0 + a.every // 2, f1, a.every))
    cut = f0 + int((1 - a.val_frac) * (f1 - f0))
    gap = int(a.gap_s * fps)
    rows = []
    for f in frames:
        if not a.all_split and a.val_frac > 0 and cut - gap <= f < cut + gap:
            continue                                    # buffer between splits (adjacent frames ~identical)
        rows.append({"filename": f"frame_{f:06d}.png", "frame_id": f, "label": "",
                     "split": a.all_split or ("val" if (a.val_frac > 0 and f >= cut + gap) else "train")})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    sp = pd.Series([r["split"] for r in rows]).value_counts().to_dict()
    print(f"{a.video}: {n} frames @ {fps:.2f} fps -> {len(rows)} frames every {a.every} ({sp}) -> {a.out}")


def cmd_export_yolo(a):
    """boxes.csv (+ the sample manifest's split) -> YOLO dataset at native resolution.
    drone -> --drone_class, bird_or_other -> --bird_class (or dropped if < 0); 'nothing' frames are
    exported as background images; 'unsure' frames are skipped."""
    b = pd.read_csv(a.boxes)
    m = pd.read_csv(a.manifest)[["frame_id", "split"]]
    b = b.merge(m, on="frame_id", how="left")
    frames = FrameReader(a.video, needed=b.frame_id.unique())
    W, H = int(frames.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(frames.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counts = {}
    for fid, g in b.groupby("frame_id"):
        if (g.verdict == "unsure").all():
            continue
        split = g.split.iloc[0] if isinstance(g.split.iloc[0], str) else "train"
        im = frames.read(int(fid))
        if im is None:
            continue
        stem = f"{a.prefix}_{int(fid):06d}"
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
        cv2.imwrite(os.path.join(a.out, split, "images", stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for r in g.itertuples():
            if r.verdict == "drone":
                c = a.drone_class
            elif r.verdict == "bird_or_other" and a.bird_class >= 0:
                c = a.bird_class
            else:
                continue
            x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
            lines.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        with open(os.path.join(a.out, split, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        key = (split, "with_drone" if any(l.startswith(f"{a.drone_class} ") for l in lines) else "no_drone")
        counts[key] = counts.get(key, 0) + 1
    print(f"exported to {a.out}: {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--video", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("review")
    r.add_argument("--video", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--proposals", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--registration", default=None,
                   help="video_pairs registration.json of this session (enables the RGB panel, 't' and 'v')")
    r.add_argument("--day", default=None, help="the synced day video of this session")
    r.add_argument("--verify", action="store_true",
                   help="re-verify existing labels with the previous box pre-filled (Enter keeps it)")
    r.add_argument("--only", default="capped,nothing,unsure",
                   help="with --verify: comma list of capped,nothing,unsure,drone,suspect,unlabelled,all")
    r.add_argument("--redo-capped", action="store_true",
                   help="revisit only frames whose box hit the old 51-px auto-fit cap (fixed 2026-09-30)")
    sm = sub.add_parser("sample", help="sample frames of a new session for labelling (train/val split)")
    sm.add_argument("--video", required=True)
    sm.add_argument("--out", required=True, help="manifest.csv to write")
    sm.add_argument("--every", type=int, default=25, help="take every N-th frame (25 = 1/s at 25 fps)")
    sm.add_argument("--start-s", type=float, default=0.0)
    sm.add_argument("--end-s", type=float, default=None)
    sm.add_argument("--val-frac", type=float, default=0.2, help="last fraction of the range -> val (contiguous)")
    sm.add_argument("--gap-s", type=float, default=5.0, help="frames within this of the split point are dropped")
    sm.add_argument("--all-split", default=None, help="put every frame in this split (e.g. 'test' for a held-out session)")
    ex = sub.add_parser("export-yolo", help="reviewed boxes -> YOLO dataset")
    ex.add_argument("--video", required=True)
    ex.add_argument("--boxes", required=True)
    ex.add_argument("--manifest", required=True, help="sample manifest (for the split column)")
    ex.add_argument("--out", required=True)
    ex.add_argument("--prefix", required=True, help="filename prefix, e.g. the session date")
    ex.add_argument("--drone_class", type=int, default=1, help="thermal-uav id in thermal-1-noleak (1)")
    ex.add_argument("--bird_class", type=int, default=0, help="bird id (0); -1 drops bird boxes")
    a = ap.parse_args()
    {"propose": cmd_propose, "review": cmd_review, "sample": cmd_sample, "export-yolo": cmd_export_yolo}[a.cmd](a)


if __name__ == "__main__":
    main()

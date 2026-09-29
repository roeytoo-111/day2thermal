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
    click  on the full frame at the drone -> box auto-fitted there (then Enter to accept, Esc to cancel)
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
    lab = dict(zip(m.frame_id.astype(int), m.label))
    cap = cv2.VideoCapture(a.video)
    day = Day(a.registration, a.day) if a.registration and a.day else None
    fields = ["frame_id", "orig_label", "verdict", "x0", "y0", "x1", "y1", "source"]

    def save():
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for k in sorted(done):
                wr.writerow(done[k])

    def read(fi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im = cap.read()
        return im

    click = {"pt": None}
    n_panels = 2 if day is not None else 1          # IR | registered RGB
    tiles_from = {"src": "IR"}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and x < 640 * n_panels and y < 512:
            click["pt"] = (x % 640, y)                  # same pixel grid in both panels

    cv2.namedWindow("GT boxes")
    cv2.setMouseCallback("GT boxes", on_mouse)
    i = next((k for k, f in enumerate(todo) if f not in done), len(todo))
    while 0 <= i < len(todo):
        fi = todo[i]
        is_pos = lab[fi] in ("TP", "TP_loose")
        is_new = lab[fi] == ""
        im = read(fi)
        cands = props.get(str(fi), [])
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
        head = (f"[{i + 1}/{len(todo)}] frame {fi}  label {lab[fi]}  "
                + ("1-6/click=drone  n=no drone" if is_pos else
                   "1-6/click=drone  Enter=nothing  w=bird" if is_new else "Enter=nothing  d=drone  w=bird")
                + "  p=play v=4K t=tiles IR/RGB s=skip b=back q=quit" + (f"   (was: {prev})" if prev else ""))
        cv2.putText(canvas, head, (6, 534), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imshow("GT boxes", canvas)
        click["pt"] = None
        key = cv2.waitKey(0) & 0xFF

        def pick_box(key):
            """Return (box, source) from a candidate number key or a click (waits if needed)."""
            if ord("1") <= key <= ord("6") and key - ord("1") < len(cands):
                return cands[key - ord("1")]["box"], f"proposal{key - ord('0')}"
            while click["pt"] is None:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 == 27:
                    return None, None
                if ord("1") <= k2 <= ord("6") and k2 - ord("1") < len(cands):
                    return cands[k2 - ord("1")]["box"], f"proposal{k2 - ord('0')}"
            x, y = click["pt"]
            # snap to the strongest small-blob response within 6 px of the click
            win = g[max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            th = np.maximum(win - cv2.morphologyEx(win, cv2.MORPH_OPEN, k), cv2.morphologyEx(win, cv2.MORPH_CLOSE, k) - win)
            dy, dx = np.unravel_index(np.argmax(th), th.shape)
            sx, sy = max(x - 6, 0) + dx, max(y - 6, 0) + dy
            full = np.maximum(g - cv2.morphologyEx(g, cv2.MORPH_OPEN, k), cv2.morphologyEx(g, cv2.MORPH_CLOSE, k) - g)
            box = fit_box(cv2.GaussianBlur(full, (0, 0), 1.0), sx, sy)
            show = canvas.copy()
            cv2.rectangle(show, (box[0] - 2, box[1] - 2), (box[2] + 2, box[3] + 2), (255, 255, 255), 1)
            cv2.putText(show, "Enter=accept  Esc=cancel", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("GT boxes", show)
            k3 = cv2.waitKey(0) & 0xFF
            return (box, "click") if k3 in (13, 10) else (None, None)

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
            cx, cy = (cands[0]["x"], cands[0]["y"]) if cands else (320, 256)
            if click["pt"]:
                cx, cy = click["pt"]
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
            cx, cy = (cands[0]["x"], cands[0]["y"]) if cands else (320, 256)
            if click["pt"]:
                cx, cy = click["pt"]
            c, msg = day.crop(fi, cx, cy)
            if c is None:
                print(f"  day view: {msg}")
            else:
                cv2.imshow("day camera", c)
                cv2.waitKey(0)
                cv2.destroyWindow("day camera")
            continue
        verdict, box, src = None, None, None
        if is_new:
            if key in (13, 10):
                verdict = "nothing"
            elif key == ord("w"):
                box, src = pick_box(0xFF)
                verdict = "bird_or_other" if box else None
            elif ord("1") <= key <= ord("6") or key == 0xFF or click["pt"] is not None:
                box, src = pick_box(key)
                verdict = "drone" if box else None
        elif is_pos:
            if key == ord("n"):
                verdict = "no_drone_visible"
            elif ord("1") <= key <= ord("6") or key == 0xFF or click["pt"] is not None:
                box, src = pick_box(key)
                verdict = "drone" if box else None
            else:
                box, src = pick_box(key) if key not in (13, 10) else (None, None)
                verdict = "drone" if box else None
        else:
            if key in (13, 10):
                verdict = "nothing"
            elif key in (ord("d"), ord("w")):
                box, src = pick_box(0xFF)
                verdict = ("drone" if key == ord("d") else "bird_or_other") if box else None
        if verdict is None:
            continue
        b = box or ["", "", "", ""]
        done[fi] = {"frame_id": fi, "orig_label": lab[fi], "verdict": verdict,
                    "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "source": src or ""}
        if len(done) % 10 == 0:
            save()
        i += 1
    save()
    cv2.destroyAllWindows()
    v = pd.Series([r["verdict"] for r in done.values()]).value_counts().to_dict()
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
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counts = {}
    for fid, g in b.groupby("frame_id"):
        if (g.verdict == "unsure").all():
            continue
        split = g.split.iloc[0] if isinstance(g.split.iloc[0], str) else "train"
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, im = cap.read()
        if not ok:
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

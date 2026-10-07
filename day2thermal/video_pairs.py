"""Registered, diversity-subsampled RGB/thermal training pairs from a synced
day + thermal VIDEO pair (rigid rig, day camera FOV inside the thermal FOV).

Two stages:

calibrate
    Automatic cross-modal registration, verified across many frames.
    * Frames are chosen where the rig is nearly still (from the ego-motion
      cache written by detection/src/data/verify_stream_sync.py), so a
      frame-timing error cannot masquerade as a geometric offset.
    * Per frame: coarse search over scale x rotation x translation, by
      normalised cross-correlation of gradient-magnitude maps (edges are the
      only structure the two modalities share), seeded with the pixel-scale
      ratio measured from ego-motion; then ECC refinement (affine by default).
    * The thermal core is wide-angle with strong barrel distortion, so a
      plain homography fits each frame differently depending on where the
      structure is. A division-model lens coefficient (lambda) is searched:
      the right one makes the per-frame solutions agree.
    * The day camera is NOT rigid w.r.t. the thermal over the whole recording:
      the relative pose steps between stretches of time (first seen on
      day.mp4/thermal.mp4: ~1.5 deg roll / ~13 px jumps between tight clusters).
      So solutions are segmented in time; a segment is trusted only with
      >= --min-segment-frames agreeing frames, a lone disagreeing frame
      (near-field parallax, bad fit) is an outlier, and the gaps between
      segments (change happened at an unknown moment) are excluded.
    * End-to-end check with the exact remap extract uses: per-cell residual
      over the overlap and per-frame edge NCC.
    Writes registration.json + overlay previews.

extract
    One sequential decode of both videos. Candidates on a time grid; a
    candidate is kept only if the scene changed enough since the last kept
    pair (so a minute of static sky yields one pair, not 1500) and the
    thermal frame is not a NUC/FFC freeze. RGB is warped into the thermal
    geometry (thermal pixels never resampled) and both are cropped to the
    fully-valid overlap. RGB is sampled through undistort(thermal pixel) ->
    H^-1, i.e. warped INTO the distorted thermal geometry. Split by thermal
    frame index, so a detector evaluation holdout on the same video stays
    untouched:
        train: thermal frame <  --train-max-frame
        val:   thermal frame >= --val-min-frame   (frames in between dropped)
    Output layout is what day2thermal.train expects: out/{train,val}/{rgb,thermal}/.

Example
-------
python -m day2thermal.video_pairs calibrate --day day.mp4 --thermal thermal.mp4 \
    --sync-report reports/stream_sync/sync_report.json --out data/arsuf2_pairs
python -m day2thermal.video_pairs extract --day day.mp4 --thermal thermal.mp4 \
    --registration data/arsuf2_pairs/registration.json --out data/arsuf2_pairs
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np

from .register import valid_crop_from_H
from .utils import ensure_dir, save_json


# --------------------------------------------------------------------- io
def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,nb_frames:format=duration", "-of", "json", path])
    info = json.loads(out)
    s = info["streams"][0]
    n = int(s["nb_frames"])
    return {"n": n, "fps": n / float(info["format"]["duration"]), "w": int(s["width"]), "h": int(s["height"])}


def ffmpeg_frames(path, size=None, gray=False):
    """Yield decoded frames sequentially (optionally area-downscaled) via an ffmpeg pipe."""
    meta = probe(path)
    w, h = size if size else (meta["w"], meta["h"])
    vf = []
    if size:
        vf.append(f"scale={w}:{h}:flags=area")
    vf.append("format=gray" if gray else "format=bgr24")
    ch = 1 if gray else 3
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "0", "-i", path, "-fps_mode", "passthrough",
                             "-vf", ",".join(vf),   # passthrough: files whose header fps is wrong (Boson .ts: 60/120 for 30) would be duplicated
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE, bufsize=w * h * ch * 8)
    try:
        while True:
            buf = proc.stdout.read(w * h * ch)
            if len(buf) < w * h * ch:
                return
            f = np.frombuffer(buf, np.uint8).reshape(h, w, ch) if ch == 3 else \
                np.frombuffer(buf, np.uint8).reshape(h, w)
            yield f
    finally:
        proc.kill()
        proc.wait()


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    return f if ok else None


def day_index(th_idx, fps_th, fps_day, offset_s, drift_s_per_s=0.0):
    """thermal_time = day_time + offset(t), offset(t) = offset_s + drift_s_per_s * t
    (verify_stream_sync.py convention; drift is non-zero when the two recorders' clocks differ)."""
    t = th_idx / fps_th
    return int(round((t - (offset_s + drift_s_per_s * t)) * fps_day))


# --------------------------------------------------------------- geometry
def T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def S(s):
    return np.diag([s, s, 1.0])


def R(s):
    """Coordinate map of cv2.resize by factor s (pixel centres: x' = (x + 0.5) * s - 0.5)."""
    return T(0.5 * s - 0.5, 0.5 * s - 0.5) @ S(s)


def edge_map(gray, sigma):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    hi = np.percentile(m, 99.5)
    return np.clip(m / (hi + 1e-6), 0, 1).astype(np.float32)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def coarse_search(th_e, day_gray_full, s0, scale_span, rot_span, search_px):
    """Best similarity (day full-res -> thermal) by NCC of edge maps. Returns (H, score, peak_margin)."""
    th_h, th_w = th_e.shape
    best = None
    for s in s0 * np.linspace(1 - scale_span, 1 + scale_span, 9):
        small = cv2.resize(day_gray_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        e_small = edge_map(small, 1.0)
        hs, ws = e_small.shape
        for r in np.linspace(-rot_span, rot_span, 7):
            Rot = np.vstack([cv2.getRotationMatrix2D((ws / 2, hs / 2), r, 1.0), [0, 0, 1]])
            rot = cv2.warpAffine(e_small, Rot[:2], (ws, hs))
            mx, my = int(0.05 * ws), int(0.05 * hs)
            tpl = rot[my:hs - my, mx:ws - mx]
            th_, tw_ = tpl.shape
            # search window around the centred placement
            cx0 = (th_w - tw_) // 2 - search_px
            cy0 = (th_h - th_) // 2 - search_px
            x0, y0 = max(cx0, 0), max(cy0, 0)
            x1 = min(th_w, (th_w + tw_) // 2 + search_px)
            y1 = min(th_h, (th_h + th_) // 2 + search_px)
            region = th_e[y0:y1, x0:x1]
            if region.shape[0] < th_ or region.shape[1] < tw_:
                continue
            res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[1]:
                sup = res.copy()
                cv2.circle(sup, loc, 6, -1.0, -1)
                margin = score - float(sup.max())
                H = T(x0 + loc[0], y0 + loc[1]) @ T(-mx, -my) @ Rot @ R(s)
                best = (H, float(score), float(margin))
    return best


def ecc_refine(th_e, day_gray_full, H_coarse, work_scale, motion, iters):
    """Refine day->thermal H with ECC on edge maps. Template = thermal edges inside the footprint."""
    dw = cv2.resize(day_gray_full, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    day_e = edge_map(dw, 1.5)
    Hd = H_coarse @ np.linalg.inv(R(work_scale))            # day-work -> thermal
    crop = valid_crop_from_H(Hd.astype(np.float32), (dw.shape[1], dw.shape[0]),
                             (th_e.shape[1], th_e.shape[0]), margin=8)
    if crop is None:
        return None, None
    cx, cy, cw, chh = crop
    tpl = np.ascontiguousarray(th_e[cy:cy + chh, cx:cx + cw])
    W = np.linalg.inv(Hd) @ T(cx, cy)                        # template(crop) -> day-work
    W = W / W[2, 2]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    if motion == "affine":
        warp = W[:2].astype(np.float32)
        mt = cv2.MOTION_AFFINE
    else:
        warp = W.astype(np.float32)
        mt = cv2.MOTION_HOMOGRAPHY
    try:
        cc, warp = cv2.findTransformECC(tpl, day_e, warp, mt, crit, None, 5)
    except cv2.error:
        return None, None
    W = np.vstack([warp, [0, 0, 1]]) if motion == "affine" else warp.astype(np.float64)
    Hd_ref = np.linalg.inv(W.astype(np.float64) @ T(-cx, -cy))   # day-work -> thermal
    H = Hd_ref @ R(work_scale)                                     # day full -> thermal
    return H / H[2, 2], float(cc)


def map_pts(H, pts):
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H)
    return p.reshape(-1, 2)


def grid_pts(w, h, n=5):
    xs, ys = np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, n))
    return np.stack([xs.ravel(), ys.ravel()], 1)


# ----------------------------------------------------- thermal lens model
# Division model (Fitzgibbon 2001), closed-form in both directions:
#     x_u = c + (x_d - c) / (1 + lam * r_d^2),   r in units of f = thermal_width / 2
# lam < 0 = barrel distortion (the wide-angle thermal core). lam = 0 = pinhole.
def und_from_dist(xd, yd, lam, size):
    f, cx, cy = size[0] / 2, (size[0] - 1) / 2, (size[1] - 1) / 2
    dx, dy = (xd - cx) / f, (yd - cy) / f
    s = 1.0 / (1.0 + lam * (dx * dx + dy * dy))
    return cx + dx * s * f, cy + dy * s * f


def dist_from_und(xu, yu, lam, size):
    f, cx, cy = size[0] / 2, (size[0] - 1) / 2, (size[1] - 1) / 2
    dx, dy = (xu - cx) / f, (yu - cy) / f
    if lam == 0:
        return xu, yu
    ru = np.sqrt(dx * dx + dy * dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = (1 - np.sqrt(1 - 4 * lam * ru * ru)) / (2 * lam * ru)
        s = np.where(ru > 1e-9, rd / ru, 1.0)
    return cx + dx * s * f, cy + dy * s * f


def undistort_image(img, lam):
    """Resample to the undistorted (pinhole) geometry -- used for CALIBRATION only."""
    h, w = img.shape[:2]
    yu, xu = np.mgrid[0:h, 0:w].astype(np.float64)
    mx, my = dist_from_und(xu, yu, lam, (w, h))
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def build_rgb_maps(H, lam, th_size, day_size, ds):
    """remap() maps: for every (distorted) thermal pixel, where to sample the day frame
    decoded at 1/ds scale. Thermal pixels are never resampled."""
    w, h = th_size
    yd, xd = np.mgrid[0:h, 0:w].astype(np.float64)
    xu, yu = und_from_dist(xd, yd, lam, th_size)
    Hi = np.linalg.inv(H)
    den = Hi[2, 0] * xu + Hi[2, 1] * yu + Hi[2, 2]
    # full-res day coords -> coords in the frame decoded at 1/ds (resize pixel-centre convention)
    X = ((Hi[0, 0] * xu + Hi[0, 1] * yu + Hi[0, 2]) / den + 0.5) / ds - 0.5
    Y = ((Hi[1, 0] * xu + Hi[1, 1] * yu + Hi[1, 2]) / den + 0.5) / ds - 0.5
    valid = (X >= 0) & (Y >= 0) & (X <= day_size[0] / ds - 1) & (Y <= day_size[1] / ds - 1)
    return X.astype(np.float32), Y.astype(np.float32), valid


def inner_rect(valid, margin=2):
    """Greedy largest axis-aligned rectangle fully inside the valid mask."""
    ys, xs = np.nonzero(valid)
    x0, x1, y0, y1 = xs.min() + margin, xs.max() + 1 - margin, ys.min() + margin, ys.max() + 1 - margin
    while x1 - x0 > 16 and y1 - y0 > 16:
        sub = valid[y0:y1, x0:x1]
        if sub.all():
            return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        bad = {"t": (~sub[0]).sum(), "b": (~sub[-1]).sum(), "l": (~sub[:, 0]).sum(), "r": (~sub[:, -1]).sum()}
        k = max(bad, key=bad.get)
        if k == "t":
            y0 += 1
        elif k == "b":
            y1 -= 1
        elif k == "l":
            x0 += 1
        else:
            x1 -= 1
    return None


# -------------------------------------------------------------- calibrate
def pick_still_frames(sync_report, n_th, fps_th, n_frames):
    """Evenly spread thermal indices where the rig is nearly still (low ego-motion)."""
    cache = os.path.join(os.path.dirname(sync_report), "motion_cache.npz") if sync_report else None
    cand = np.arange(int(2 * fps_th), n_th - int(2 * fps_th))
    if cache and os.path.exists(cache):
        m = np.load(cache)["thermal"]
        speed = np.hypot(m[:, 0], m[:, 1])
        k = int(fps_th)
        roll = np.array([speed[max(i - k, 0):i + k + 1].max() for i in range(len(speed))])
        cand = cand[roll[cand] < np.percentile(roll[cand], 40)]
    return [int(b[len(b) // 2]) for b in np.array_split(cand, n_frames) if len(b)]


def solve_frames(frames, lam, s0, a, verbose=False, cache=None):
    """Per-frame day->undistorted-thermal H. Returns list of dicts (ok frames only).
    cache: dict persisted by the caller, keyed "lam|thermal_idx" (re-segmenting needs no re-solve)."""
    sols = []
    for ti, th_g, dy_g in frames:
        key = f"{lam:.4f}|{ti}"
        if cache is not None and key in cache:
            c = dict(cache[key])
            if c.get("H") is not None:
                c["H"] = np.array(c["H"])
                sols.append(c)
            continue
        th_e = edge_map(undistort_image(th_g, lam), 1.0)
        if th_e.mean() < 0.01:
            continue
        cs = coarse_search(th_e, dy_g, s0, a.scale_span, a.rot_span, a.search_px)
        H = None
        if cs is not None:
            Hc, score, margin = cs
            H, cc = ecc_refine(th_e, dy_g, Hc, a.work_scale, a.motion, a.ecc_iters)
        if H is None:
            if cache is not None:
                cache[key] = {"thermal_idx": ti, "H": None}
            continue
        sx = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
        rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
        sols.append({"thermal_idx": ti, "H": H, "ecc_cc": cc, "coarse_score": score, "coarse_margin": margin,
                     "scale": sx, "rot_deg": rot, "tx": float(H[0, 2]), "ty": float(H[1, 2])})
        if cache is not None:
            cache[key] = {**sols[-1], "H": H.tolist()}
        if verbose:
            print(f"    thermal {ti:6d}: coarse {score:.3f} ecc {cc:.3f} scale {sx:.4f} rot {rot:+.2f} "
                  f"t=({H[0, 2]:.1f},{H[1, 2]:.1f})", flush=True)
    return sols


def footprint_dev(Ha, Hb, gp):
    """Max disagreement (thermal px) between two day->thermal maps over a grid of day points."""
    return float(np.abs(map_pts(Ha, gp) - map_pts(Hb, gp)).max())


def median_H(group, day_size):
    dw, dh = day_size
    corners = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], np.float64)
    Pm = np.median(np.stack([map_pts(s["H"], corners) for s in group]), 0)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), Pm.astype(np.float32)).astype(np.float64)


def segment_timeline(sols, day_size, min_cc, max_dev):
    """Split time-ordered per-frame solutions into segments of constant geometry.
    A solution that disagrees with the running segment starts a new one only if the NEXT good
    solution agrees with it (a lone disagreeing frame -- near-field parallax, bad fit -- is an outlier)."""
    gp = grid_pts(*day_size)
    good = sorted([s for s in sols if s["ecc_cc"] >= min_cc], key=lambda s: s["thermal_idx"])
    for s in sols:
        s["segment"] = None
        s["role"] = "low_cc" if s["ecc_cc"] < min_cc else None
    segs, cur = [], []
    for i, s in enumerate(good):
        if not cur:
            cur = [s]
            continue
        Hc = median_H(cur, day_size)
        if footprint_dev(s["H"], Hc, gp) <= max_dev:
            cur.append(s)
            continue
        nxt = good[i + 1] if i + 1 < len(good) else None
        if nxt is not None and footprint_dev(nxt["H"], Hc, gp) <= max_dev:
            s["role"] = "outlier"
            continue
        if nxt is not None and footprint_dev(nxt["H"], s["H"], gp) > max_dev:
            s["role"] = "outlier"            # agrees with neither side
            continue
        segs.append(cur)
        cur = [s]
    if cur:
        segs.append(cur)
    out = []
    for k, grp in enumerate(segs):
        Hm = median_H(grp, day_size)
        dev = np.array([footprint_dev(s["H"], Hm, gp) for s in grp])
        for s, d in zip(grp, dev):
            s["segment"], s["role"], s["dev_from_segment_px"] = k, "member", float(d)
        out.append({"id": k, "H": Hm, "members": [s["thermal_idx"] for s in grp], "n": len(grp),
                    "dev_px_median": float(np.median(dev)), "dev_px_max": float(dev.max()),
                    "first": grp[0]["thermal_idx"], "last": grp[-1]["thermal_idx"]})
    return out


def local_agreement(sols, day_size, min_cc, max_dev):
    """Lens-model score: fraction of temporally adjacent good solutions that agree (geometry is
    piecewise constant, so neighbours should agree when the lens model is right)."""
    gp = grid_pts(*day_size)
    good = sorted([s for s in sols if s["ecc_cc"] >= min_cc], key=lambda s: s["thermal_idx"])
    if len(good) < 3:
        return 0.0, float("inf")
    d = np.array([footprint_dev(a["H"], b["H"], gp) for a, b in zip(good, good[1:])])
    return float((d <= max_dev).mean()), float(np.median(d))


def structure_frac(gray, thresh=12.0):
    """Fraction of pixels with a real (un-normalised) gradient > thresh grey levels/px.
    Pure sky is ~0; the sparsest cloud pairs are ~0.04."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float((m > thresh).mean())


def edge_ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def cmd_calibrate(a):
    out = ensure_dir(a.out)
    prev_dir = ensure_dir(os.path.join(out, "calib_preview"))
    md, mt = probe(a.day), probe(a.thermal)
    offset = a.offset_ms / 1000 if a.offset_ms is not None else None
    drift = 0.0
    s0 = a.scale
    if a.sync_report:
        rep = json.load(open(a.sync_report))
        if offset is None:
            lm = rep.get("lag_model")
            if lm and lm["type"] in ("constant", "linear"):
                offset, drift = lm["intercept_ms"] / 1000, lm["slope_ms_per_s"] / 1000
            else:
                offset = rep["global"]["lag_ms"] / 1000
        if s0 is None:
            dsz = [int(v) for v in rep["decode_sizes"]["day"].split("x")]
            tsz = [int(v) for v in rep["decode_sizes"]["thermal"].split("x")]
            sc = np.mean(rep["thermal_per_day_px_scale_at_decode"])
            # motion regression attenuates the slope (noise in the day velocities) -> seed only
            s0 = float(sc * (dsz[0] / md["w"]) / (tsz[0] / mt["w"]))
    if offset is None or s0 is None:
        sys.exit("need --sync-report or both --offset-ms and --scale")
    print(f"offset {offset * 1000:+.0f} ms, drift {drift * 1000:+.3f} ms/s | seed scale {s0:.4f} thermal px per day px")

    idxs = pick_still_frames(a.sync_report, mt["n"], mt["fps"], a.n_frames)
    cap_d, cap_t = cv2.VideoCapture(a.day), cv2.VideoCapture(a.thermal)
    frames = []
    for ti in idxs:
        th = grab(cap_t, ti)
        dy = grab(cap_d, day_index(ti, mt["fps"], md["fps"], offset, drift))
        if th is not None and dy is not None:
            frames.append((ti, to_gray(th), to_gray(dy)))
    th_size = (frames[0][1].shape[1], frames[0][1].shape[0])
    day_size = (md["w"], md["h"])
    spacing = float(np.median(np.diff([f[0] for f in frames])))
    print(f"{len(frames)} calibration frames read (median spacing {spacing / mt['fps']:.1f} s)")

    cache_path = os.path.join(out, "solutions_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    # 1) lens distortion: the lambda at which temporally adjacent solutions agree best
    lams = [float(v) for v in a.lambdas.split(",")] if a.lambdas else [0.0]
    sub = frames[:: max(1, len(frames) // a.n_search_frames)]
    search = []
    for lam in lams:
        sols = solve_frames(sub, lam, s0, a, cache=cache)
        json.dump(cache, open(cache_path, "w"))
        frac, med = local_agreement(sols, day_size, a.min_cc, a.max_dev_px)
        mcc = float(np.mean([s["ecc_cc"] for s in sols])) if sols else 0.0
        search.append({"lambda": lam, "n_solved": len(sols), "adjacent_agree_frac": frac,
                       "adjacent_dev_median_px": med, "mean_ecc_cc": mcc})
        print(f"  lambda {lam:+.3f}: adjacent solutions agree {frac:.0%} (median dev {med:.2f} px), "
              f"mean ecc {mcc:.3f}", flush=True)
    best = max(search, key=lambda r: (round(r["adjacent_agree_frac"], 3), -r["adjacent_dev_median_px"]))
    lam = best["lambda"]
    print(f"-> lambda {lam:+.3f}")

    # 2) all frames at that lambda, segmented in time
    sols = solve_frames(frames, lam, s0, a, verbose=True, cache=cache)
    json.dump(cache, open(cache_path, "w"))
    segs = segment_timeline(sols, day_size, a.min_cc, a.max_dev_px)
    half = int(spacing // 2)
    verified = [g for g in segs if g["n"] >= a.min_segment_frames]
    for g in segs:
        g["verified"] = g["n"] >= a.min_segment_frames
        # conservative validity span: member range +/- half a sampling interval
        g["span"] = [max(0, g["first"] - half), min(mt["n"] - 1, g["last"] + half)]
    if not verified:
        sys.exit("no verified segment (>= --min-segment-frames agreeing frames); see calib_preview/")

    # 3) common crop over verified segments, in the ORIGINAL thermal geometry
    ds = max(1, int(round(1 / np.median([np.sqrt(abs(np.linalg.det(g["H"][:2, :2]))) for g in verified]))))
    valid_all = None
    maps = {}
    for g in verified:
        mx, my, valid = build_rgb_maps(g["H"], lam, th_size, day_size, ds)
        maps[g["id"]] = (mx, my)
        valid_all = valid if valid_all is None else (valid_all & valid)
    crop = inner_rect(valid_all)
    cx, cy, cw, chh = crop

    # 4) end-to-end check with the exact maps extract will use: per-cell residual + alignment NCC
    gx, gy = 4, 3
    cells = [[[] for _ in range(gx)] for _ in range(gy)]
    fmap = {f[0]: f for f in frames}
    tiles, nccs = [], []
    members = [s for s in sols if s["role"] == "member" and segs[s["segment"]]["verified"]]
    for n_s, s in enumerate(members):
        _, th_g, dy_g = fmap[s["thermal_idx"]]
        mx, my = maps[s["segment"]]
        small = cv2.resize(dy_g, (day_size[0] // ds, day_size[1] // ds), interpolation=cv2.INTER_AREA)
        rgb_w = cv2.remap(small, mx, my, cv2.INTER_LINEAR)
        e1 = edge_map(th_g, 1.0)[cy:cy + chh, cx:cx + cw]
        e2 = edge_map(rgb_w, 1.0)[cy:cy + chh, cx:cx + cw]
        s["align_ncc"] = edge_ncc(e1, e2)
        nccs.append(s["align_ncc"])
        for j in range(gy):
            for i in range(gx):
                ys, xs = slice(j * chh // gy, (j + 1) * chh // gy), slice(i * cw // gx, (i + 1) * cw // gx)
                c1, c2 = e1[ys, xs], e2[ys, xs]
                if c1.std() < 0.02 or c2.std() < 0.02:
                    continue
                (sx_, sy_), resp = cv2.phaseCorrelate(c2.astype(np.float64), c1.astype(np.float64))
                if resp > 0.1 and max(abs(sx_), abs(sy_)) < 10:
                    cells[j][i].append((sx_, sy_))
        if n_s % max(1, len(members) // 8) == 0 and len(tiles) < 8:
            vis = cv2.cvtColor(th_g, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.7
            vis[cy:cy + chh, cx:cx + cw, 1] = np.clip(vis[cy:cy + chh, cx:cx + cw, 1] + 255 * e2, 0, 255)
            vis = vis.astype(np.uint8)
            cv2.rectangle(vis, (cx, cy), (cx + cw, cy + chh), (0, 0, 255), 1)
            cv2.putText(vis, f"t{s['thermal_idx']} seg{s['segment']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)
            tiles.append(vis)
            cv2.imwrite(os.path.join(prev_dir, f"overlay_{s['thermal_idx']:06d}.png"), vis)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(prev_dir, "overlay_grid.png"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    cell_summary = [[{"n": len(cl), "median_dx": float(np.median([v[0] for v in cl])) if cl else None,
                      "median_dy": float(np.median([v[1] for v in cl])) if cl else None} for cl in row]
                    for row in cells]

    # report
    print(f"\nlambda {lam:+.3f} | {len(sols)}/{len(frames)} frames solved | segments:")
    for g in segs:
        H = g["H"]
        print(f"  seg {g['id']:2d} {'VERIFIED' if g['verified'] else 'unverified':10s} frames {g['first']:6d}-"
              f"{g['last']:6d} n={g['n']:3d}  scale {np.sqrt(abs(np.linalg.det(H[:2, :2]))):.4f} "
              f"rot {np.degrees(np.arctan2(H[1, 0], H[0, 0])):+.2f}  t=({H[0, 2]:.1f},{H[1, 2]:.1f})  "
              f"dev median {g['dev_px_median']:.2f} max {g['dev_px_max']:.2f} px")
    n_out = sum(s["role"] == "outlier" for s in sols)
    covered = sum(g["span"][1] - g["span"][0] + 1 for g in verified) / mt["n"]
    print(f"outlier frames: {n_out}; verified segments cover {covered:.0%} of the recording")
    print("per-cell residual in the original thermal geometry (dx,dy px; n frames):")
    for row in cell_summary:
        print("   " + "  ".join(f"({cl['median_dx']:+.1f},{cl['median_dy']:+.1f};{cl['n']:2d})" if cl["n"]
                                else "(   --   ;  0)" for cl in row))
    dev_members = np.array([s["dev_from_segment_px"] for s in members])
    cell_mags = [np.hypot(cl["median_dx"], cl["median_dy"]) for row in cell_summary for cl in row if cl["n"] >= 3]
    verdict = "PASS" if (covered >= 0.5 and np.median(dev_members) <= 1.5
                         and cell_mags and max(cell_mags) <= 2.0) else "FAIL"
    align_min = float(np.percentile(nccs, 5)) if nccs else 0.0
    print(f"member deviation from own segment: median {np.median(dev_members):.2f}, max {dev_members.max():.2f} px; "
          f"alignment NCC p05 {align_min:.3f}; pair crop {crop} -> VERDICT: {verdict}")

    save_json({"mode": "video_pairs_segmented",
               "lens": {"model": "division", "lambda": lam, "f_px": th_size[0] / 2,
                        "center": [(th_size[0] - 1) / 2, (th_size[1] - 1) / 2]},
               "segments": [{**{k: v for k, v in g.items() if k != "H"},
                             "H_day_to_undistorted_thermal": g["H"].tolist()} for g in segs],
               "crop": crop, "thermal_size": list(th_size), "rgb_size": list(day_size), "day_decode_downscale": ds,
               "offset_ms": offset * 1000, "drift_ms_per_s": drift * 1000,
               "fps_day": md["fps"], "fps_thermal": mt["fps"],
               "day": a.day, "thermal": a.thermal, "motion": a.motion, "seed_scale": s0,
               "lambda_search": search, "coverage": covered, "align_ncc_p05": align_min,
               "cell_residual": cell_summary, "verdict": verdict,
               "frames": [{k: v for k, v in s.items() if k != "H"} for s in sols]},
              os.path.join(out, "registration.json"))
    print(f"wrote {os.path.join(out, 'registration.json')} and {prev_dir}/")
    return 0 if verdict == "PASS" else 2


# ---------------------------------------------------------------- extract
def descriptor(th_g, day_small_g):
    a = cv2.resize(th_g, (40, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    b = cv2.resize(day_small_g, (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-3)
    b = (b - b.mean()) / (b.std() + 1e-3)
    return np.concatenate([a, b])


def cmd_extract(a):
    reg = json.load(open(a.registration))
    if reg.get("verdict") != "PASS" and not a.force:
        sys.exit("registration verdict is not PASS; inspect calib_preview/ (or pass --force).")
    lam = reg["lens"]["lambda"]
    th_size, day_size, ds = tuple(reg["thermal_size"]), tuple(reg["rgb_size"]), reg["day_decode_downscale"]
    segs = [g for g in reg["segments"] if g["verified"]]
    maps = {g["id"]: build_rgb_maps(np.array(g["H_day_to_undistorted_thermal"]), lam, th_size, day_size, ds)[:2]
            for g in segs}

    def seg_of(ti):
        for g in segs:
            if g["span"][0] <= ti <= g["span"][1]:
                return g["id"]
        return None

    x, y, w, h = reg["crop"]
    min_align = a.min_align if a.min_align is not None else reg["align_ncc_p05"]
    md, mt = probe(a.day), probe(a.thermal)
    offset = reg["offset_ms"] / 1000
    drift = reg.get("drift_ms_per_s", 0.0) / 1000
    out = a.out
    for sp in ("train", "val"):
        ensure_dir(os.path.join(out, sp, "rgb"))
        ensure_dir(os.path.join(out, sp, "thermal"))
    man = open(os.path.join(out, "pairs_manifest.csv"), "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["name", "split", "thermal_idx", "day_idx", "thermal_time_s", "segment", "align_ncc",
                 "scene_dist", "gap_s"])

    step = max(1, int(round(a.step_s * mt["fps"])))
    min_gap = int(round(a.min_gap_s * mt["fps"]))
    max_gap = int(round(a.max_gap_s * mt["fps"]))
    day_it = ffmpeg_frames(a.day, (day_size[0] // ds, day_size[1] // ds))
    day_pos, day_frame = -1, None
    last_desc, last_kept, prev_th = None, -10 ** 9, None
    counts = {"train": 0, "val": 0, "gap_dropped": 0, "unregistered": 0, "freeze": 0, "static": 0,
              "misaligned": 0}
    for ti, th in enumerate(ffmpeg_frames(a.thermal, gray=True)):
        if ti % 2000 == 0:
            print(f"  thermal {ti}/{mt['n']}  kept train {counts['train']} val {counts['val']}", flush=True)
        freeze = prev_th is not None and float(np.mean(np.abs(th.astype(np.int16) - prev_th))) < 0.05
        prev_th = th.astype(np.int16)
        if ti % step:
            continue
        if a.train_max_frame <= ti < a.val_min_frame:
            counts["gap_dropped"] += 1
            continue
        seg = seg_of(ti)
        if seg is None:
            counts["unregistered"] += 1
            continue
        if freeze:
            counts["freeze"] += 1
            continue
        di = day_index(ti, mt["fps"], md["fps"], offset, drift)
        if di < 0:
            continue
        while day_pos < di:
            try:
                day_frame = next(day_it)
            except StopIteration:
                day_frame = None
                break
            day_pos += 1
        if day_frame is None:
            break
        if ti - last_kept < min_gap:
            continue
        d = descriptor(th, to_gray(day_frame))
        dist = float(np.mean(np.abs(d - last_desc))) if last_desc is not None else float("inf")
        if dist < a.scene_thresh and ti - last_kept < max_gap:
            counts["static"] += 1
            continue
        mx, my = maps[seg]
        rgb_w = cv2.remap(day_frame, mx, my, cv2.INTER_LINEAR)[y:y + h, x:x + w]
        th_c = th[y:y + h, x:x + w]
        # alignment QA: only judgeable when both crops have structure (sky-only pairs pass through)
        judgeable = structure_frac(th_c) >= a.min_structure and structure_frac(to_gray(rgb_w)) >= a.min_structure
        ncc = edge_ncc(edge_map(th_c, 1.0), edge_map(to_gray(rgb_w), 1.0)) if judgeable else float("nan")
        if ncc == ncc and ncc < min_align:
            counts["misaligned"] += 1
            continue
        split = "train" if ti < a.train_max_frame else "val"
        name = f"{a.prefix}_{ti:06d}.png"
        cv2.imwrite(os.path.join(out, split, "rgb", name), rgb_w)
        cv2.imwrite(os.path.join(out, split, "thermal", name), th_c)
        mw.writerow([name, split, ti, di, round(ti / mt["fps"], 3), seg,
                     "" if ncc != ncc else round(ncc, 4), round(dist, 4),
                     round((ti - last_kept) / mt["fps"], 2) if last_kept > 0 else ""])
        counts[split] += 1
        last_desc, last_kept = d, ti
    man.close()
    save_json({"registration": a.registration, "crop": reg["crop"], "pair_size": [w, h],
               "step_s": a.step_s, "min_gap_s": a.min_gap_s, "max_gap_s": a.max_gap_s,
               "scene_thresh": a.scene_thresh, "min_align": min_align, "min_structure": a.min_structure,
               "registration_verdict": reg.get("verdict"), "forced": bool(a.force),
               "train_max_frame": a.train_max_frame, "val_min_frame": a.val_min_frame, "counts": counts,
               "leak_note": "Pairs come from the detector's evaluation video. A translator trained on "
                            "train/ (< train_max_frame) must not be used to make detector training data "
                            "that is then evaluated on thermal frames < train_max_frame of this video."},
              os.path.join(out, "pairs_summary.json"))
    print(f"done: {counts} -> {out}  (pairs are {w}x{h})")


def cmd_resplit(a):
    """Move already-extracted pairs between train/ and val/ by thermal frame index (no re-extraction)."""
    path = os.path.join(a.out, "pairs_manifest.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    moved = 0
    for r in rows:
        new = "val" if int(r["thermal_idx"]) >= a.val_min_frame else "train"
        if new != r["split"]:
            for sub in ("rgb", "thermal"):
                os.replace(os.path.join(a.out, r["split"], sub, r["name"]), os.path.join(a.out, new, sub, r["name"]))
            r["split"] = new
            moved += 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ_path = os.path.join(a.out, "pairs_summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    summ.setdefault("resplits", []).append({"val_min_frame": a.val_min_frame, "moved": moved})
    summ["counts"]["train"] = sum(r["split"] == "train" for r in rows)
    summ["counts"]["val"] = sum(r["split"] == "val" for r in rows)
    save_json(summ, summ_path)
    print(f"moved {moved}; now train {summ['counts']['train']} / val {summ['counts']['val']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="estimate + verify the day->thermal homography")
    c.add_argument("--day", required=True)
    c.add_argument("--thermal", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--sync-report", help="verify_stream_sync.py sync_report.json (offset, seed scale, motion cache)")
    c.add_argument("--offset-ms", type=float, default=None, help="thermal_time = day_time + offset")
    c.add_argument("--scale", type=float, default=None, help="seed: thermal px per day px")
    c.add_argument("--n-frames", type=int, default=100, help="calibration frames (~every 10 s)")
    c.add_argument("--scale-span", type=float, default=0.18,
                   help="coarse search: +/- fraction around seed scale (seed from motion is biased low)")
    c.add_argument("--rot-span", type=float, default=5.0, help="coarse search: +/- degrees")
    c.add_argument("--lambdas", default="-0.075,-0.1,-0.125,-0.15,-0.175,-0.2",
                   help="division-model lens distortion values to search (thermal); '0' = pinhole")
    c.add_argument("--n-search-frames", type=int, default=34, help="frames used for the lambda search")
    c.add_argument("--min-segment-frames", type=int, default=3,
                   help="a constant-geometry segment is trusted only with this many agreeing frames")
    c.add_argument("--search-px", type=int, default=90, help="coarse search: +/- px around centred placement")
    c.add_argument("--work-scale", type=float, default=0.25, help="day downscale used for ECC")
    c.add_argument("--motion", choices=["affine", "homography"], default="affine")
    c.add_argument("--ecc-iters", type=int, default=200)
    c.add_argument("--min-cc", type=float, default=0.2)
    c.add_argument("--max-dev-px", type=float, default=3.0,
                   help="frames agree if their maps differ by <= this (thermal px); above per-frame noise "
                        "(~1-2 px), below real pose steps (>= ~5 px)")
    e = sub.add_parser("extract", help="write registered, subsampled pairs")
    e.add_argument("--day", required=True)
    e.add_argument("--thermal", required=True)
    e.add_argument("--registration", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--prefix", default="vid")
    e.add_argument("--step-s", type=float, default=0.2, help="candidate spacing")
    e.add_argument("--min-gap-s", type=float, default=1.0, help="never keep two pairs closer than this")
    e.add_argument("--max-gap-s", type=float, default=5.0, help="keep one pair at least this often, even if static")
    e.add_argument("--scene-thresh", type=float, default=0.15,
                   help="min mean |descriptor change| (z-scored thumbnails) since the last kept pair")
    e.add_argument("--min-align", type=float, default=None,
                   help="drop pairs whose edge NCC is below this (default: calibration p05); sky-only pairs exempt")
    e.add_argument("--min-structure", type=float, default=0.02,
                   help="alignment is judged only if both crops have >= this fraction of strong-edge px")
    e.add_argument("--train-max-frame", type=int, default=19900)
    e.add_argument("--val-min-frame", type=int, default=20100)
    e.add_argument("--force", action="store_true")
    r = sub.add_parser("resplit", help="move extracted pairs between train/val by thermal frame index")
    r.add_argument("--out", required=True, help="pairs dir (with pairs_manifest.csv)")
    r.add_argument("--val-min-frame", type=int, required=True)
    a = ap.parse_args()
    sys.exit({"calibrate": cmd_calibrate, "extract": cmd_extract, "resplit": cmd_resplit}[a.cmd](a))


if __name__ == "__main__":
    main()

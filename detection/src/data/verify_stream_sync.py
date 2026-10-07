"""
Verify (or refute) the time offset between a day RGB video and a thermal video
of the same rig, without relying on pixel similarity.

Pixel-similarity scores (MAD, SSIM) cannot compare RGB with thermal -- the
two sensors record unrelated intensities. What both streams DO share, if the
cameras are rigidly mounted together, is ego-motion: every pan/tilt of the
rig shifts both images at the same instant. So:

  1. decode both videos small (ffmpeg -> gray raw frames),
  2. estimate the global frame-to-frame shift (phase correlation) per frame,
  3. resample both velocity signals (vx, vy in px/s) onto one 25 Hz grid,
  4. cross-correlate thermal vs day over a lag range.

Pass criteria (all must hold):
  * global peak correlation is high and stands clear of the sidelobes
    (peak-to-sidelobe margin, sidelobe = best corr > --exclude_s away),
  * independent windows spread through the recording each find the SAME lag
    (within ~1 frame) -- a single lucky match cannot do that,
  * the lag does not drift across windows (drift => fps / clock mismatch).

Also reports the thermal/day pixel-scale ratio from the motion regression,
i.e. how many day pixels one thermal pixel spans -- needed for registration.

The lag may drift linearly when the two recorders' clocks differ slightly; then
lag_model in the report is linear and consumers must use lag(t).

Positive lag = thermal event happens LATER in thermal time than in day time,
i.e. thermal_time = day_time + lag   (same sign as extract_frames --offset-ms
with offset = thermal clock - RGB clock for the same event).

Usage:
    python3 src/data/verify_stream_sync.py --day data/videos/day.mp4 \
        --thermal data/videos/thermal.mp4 --out reports/stream_sync
"""

import os
import json
import argparse
import subprocess

import cv2
import numpy as np

GRID_HZ = 25.0


def parse_args():
    p = argparse.ArgumentParser(description="Verify RGB/thermal stream sync via ego-motion cross-correlation.")
    p.add_argument("--day", required=True)
    p.add_argument("--thermal", required=True)
    p.add_argument("--out", required=True, help="Output dir: sync_report.json, motion cache, plot.")
    p.add_argument("--day_size", default="384x216", help="Decode size for the day video (WxH).")
    p.add_argument("--thermal_size", default="320x256", help="Decode size for the thermal video (WxH).")
    p.add_argument("--max_lag_s", type=float, default=30.0)
    p.add_argument("--n_windows", type=int, default=10)
    p.add_argument("--exclude_s", type=float, default=1.0,
                   help="Sidelobe = best correlation at lags further than this from the peak.")
    return p.parse_args()


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=nb_frames:format=duration", "-of", "json", path])
    info = json.loads(out)
    n = int(info["streams"][0]["nb_frames"])
    return n, n / float(info["format"]["duration"])


def frame_motion(path, size):
    """Global (dx, dy, response) between consecutive frames, via ffmpeg gray decode + phase correlation."""
    w, h = map(int, size.split("x"))
    cmd = ["ffmpeg", "-v", "error", "-threads", "0", "-i", path, "-fps_mode", "passthrough",   # no frame dup/drop to a header fps
           "-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 64)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    prev, rows = None, []
    while True:
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)
        # high-pass so slow AGC / vignetting changes don't dominate the shift estimate
        cur = cur - cv2.GaussianBlur(cur, (0, 0), 8)
        if prev is None:
            rows.append((0.0, 0.0, 0.0))
        else:
            (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
            rows.append((dx, dy, resp))
        prev = cur
        if len(rows) % 5000 == 0:
            print(f"    {os.path.basename(path)}: {len(rows)} frames", flush=True)
    proc.wait()
    return np.array(rows, np.float64)


def to_grid(motion, fps, t_end):
    """Velocity (px/s) per frame -> 25 Hz grid by linear interpolation; robust-clipped."""
    t = np.arange(len(motion)) / fps
    v = motion[:, :2] * fps
    v[motion[:, 2] < 0.05] = 0.0          # unreliable estimate (featureless / NUC freeze) -> no motion
    grid = np.arange(0, t_end, 1 / GRID_HZ)
    out = np.stack([np.interp(grid, t, v[:, k]) for k in range(2)], 1)
    med = np.median(out, 0)
    mad = np.median(np.abs(out - med), 0) * 1.4826 + 1e-6
    return grid, np.clip(out, med - 6 * mad, med + 6 * mad)


def xcorr(day, th, lags):
    """Pearson correlation of (vx, vy) jointly, thermal shifted by each lag (in grid samples)."""
    res = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        if L >= 0:
            a, b = day[:len(day) - L], th[L:]
        else:
            a, b = day[-L:], th[:len(th) + L]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        a = (a - a.mean(0)) / (a.std(0) + 1e-9)
        b = (b - b.mean(0)) / (b.std(0) + 1e-9)
        res[i] = float((a * b).mean())
    return res


def peak_stats(corr, lags, exclude):
    i = int(np.nanargmax(corr))
    far = np.abs(lags - lags[i]) > exclude
    side = float(np.nanmax(corr[far])) if far.any() else float("nan")
    # parabolic refinement for sub-sample lag
    frac = 0.0
    if 0 < i < len(corr) - 1:
        y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return lags[i] + frac, float(corr[i]), side


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_day, fps_day = probe(a.day)
    n_th, fps_th = probe(a.thermal)
    print(f"day: {n_day} frames @ {fps_day:.4f} fps | thermal: {n_th} frames @ {fps_th:.4f} fps")

    cache = os.path.join(a.out, "motion_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        m_day, m_th = z["day"], z["thermal"]
    else:
        print("Estimating ego-motion (decode + phase correlation)...")
        m_day = frame_motion(a.day, a.day_size)
        m_th = frame_motion(a.thermal, a.thermal_size)
        np.savez_compressed(cache, day=m_day, thermal=m_th)

    t_end = min(len(m_day) / fps_day, len(m_th) / fps_th)
    _, g_day = to_grid(m_day, fps_day, t_end)
    _, g_th = to_grid(m_th, fps_th, t_end)

    max_lag = int(a.max_lag_s * GRID_HZ)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = xcorr(g_day, g_th, lags)
    lag, peak, side = peak_stats(corr, lags, a.exclude_s * GRID_HZ)
    print(f"\nGLOBAL: lag {lag / GRID_HZ * 1000:+.0f} ms  peak r={peak:.3f}  sidelobe r={side:.3f}  "
          f"margin={peak - side:.3f}")

    # pixel-scale ratio (thermal px per day px, at decode sizes) from motion regression at the best lag
    L = int(round(lag))
    d, t = (g_day[:len(g_day) - L], g_th[L:]) if L >= 0 else (g_day[-L:], g_th[:len(g_th) + L])
    n = min(len(d), len(t))
    d, t = d[:n], t[:n]
    moving = np.linalg.norm(d, axis=1) > np.percentile(np.linalg.norm(d, axis=1), 75)
    scale = [float(np.dot(d[moving, k], t[moving, k]) / np.dot(d[moving, k], d[moving, k])) for k in range(2)]

    # independent windows
    win_len = len(g_day) // a.n_windows
    wlags = np.arange(-max_lag, max_lag + 1)
    windows = []
    for w in range(a.n_windows):
        s, e = w * win_len, (w + 1) * win_len
        c = xcorr(g_day[s:e], g_th[s:e], wlags)
        wl, wp, ws = peak_stats(c, wlags, a.exclude_s * GRID_HZ)
        windows.append({"t_start_s": s / GRID_HZ, "t_end_s": e / GRID_HZ, "lag_ms": wl / GRID_HZ * 1000,
                        "peak_r": wp, "sidelobe_r": ws, "margin": wp - ws})
        print(f"  window {w}: {s / GRID_HZ:6.0f}-{e / GRID_HZ:6.0f}s  lag {wl / GRID_HZ * 1000:+7.0f} ms  "
              f"r={wp:.3f}  sidelobe={ws:.3f}  margin={wp - ws:.3f}")

    good = [x for x in windows if x["margin"] > 0.1]
    wl = np.array([x["lag_ms"] for x in good])
    # tolerance = one frame of the SLOWER stream: pairing cannot be finer than that stream's frame period
    frame_ms = 1000 / min(fps_th, fps_day)
    tc = np.array([(x["t_start_s"] + x["t_end_s"]) / 2 for x in good])
    # Lag model: constant, or linear if the two clocks run at slightly different rates
    # (seen on 2026-06-23 / 2026-07-15: ~0.5-0.8 ms of lag per second). Either way every window must
    # sit within one thermal frame of the model -- a single lucky match cannot do that.
    if len(good) >= 3:
        slope, intercept = np.polyfit(tc, wl, 1)
        resid_lin = np.abs(wl - (slope * tc + intercept))
        resid_const = np.abs(wl - np.median(wl))
    else:
        slope, intercept, resid_lin, resid_const = 0.0, float(np.median(wl)) if len(wl) else 0.0, np.array([np.inf]), np.array([np.inf])
    enough = len(good) >= max(3, a.n_windows // 2)
    if enough and resid_const.max() <= frame_ms:
        model = {"type": "constant", "intercept_ms": float(np.median(wl)), "slope_ms_per_s": 0.0}
        consistent = True
    elif enough and resid_lin.max() <= frame_ms:
        model = {"type": "linear", "intercept_ms": float(intercept), "slope_ms_per_s": float(slope)}
        consistent = True
    else:
        model = {"type": "none", "intercept_ms": float(intercept), "slope_ms_per_s": float(slope)}
        consistent = False
    drift = float(slope * 1000)
    verdict = "PASS" if (peak - side > 0.1 and consistent) else "FAIL"
    print(f"lag model: {model['type']}  lag(t) = {model['intercept_ms']:+.0f} ms {model['slope_ms_per_s']:+.3f} ms/s * t;  "
          f"max window residual {min(resid_const.max(), resid_lin.max()):.1f} ms (tolerance = one frame of the slower stream = {frame_ms:.0f} ms)")
    print(f"\nwindows with a clear peak: {len(good)}/{a.n_windows}; lags {np.round(wl).astype(int).tolist()} ms; "
          f"drift {drift:+.1f} ms per 1000 s")
    print(f"thermal/day pixel scale at decode sizes: x {scale[0]:.3f}, y {scale[1]:.3f}")
    print(f"VERDICT: {verdict}")

    report = {"day": a.day, "thermal": a.thermal, "fps_day": fps_day, "fps_thermal": fps_th,
              "n_day": n_day, "n_thermal": n_th, "decode_sizes": {"day": a.day_size, "thermal": a.thermal_size},
              "global": {"lag_ms": lag / GRID_HZ * 1000, "peak_r": peak, "sidelobe_r": side},
              "windows": windows, "drift_ms_per_1000s": drift, "lag_model": model,
              "thermal_per_day_px_scale_at_decode": scale, "verdict": verdict,
              "convention": "thermal_time = day_time + lag  (== extract_frames --offset-ms)"}
    with open(os.path.join(a.out, "sync_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(lags / GRID_HZ, corr)
        ax[0].set_xlabel("lag (s), thermal_time = day_time + lag")
        ax[0].set_ylabel("corr")
        ax[0].set_title(f"global ego-motion cross-correlation: peak {lag / GRID_HZ * 1000:+.0f} ms")
        ax[1].plot([(x["t_start_s"] + x["t_end_s"]) / 2 for x in windows], [x["lag_ms"] for x in windows], "o-")
        ax[1].set_xlabel("recording time (s)")
        ax[1].set_ylabel("window lag (ms)")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "sync_xcorr.png"), dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main()

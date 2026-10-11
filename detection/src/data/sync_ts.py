r"""
Time offset between a day and a thermal .ts recording of the same rig, in TIMESTAMP terms: thermal_t = day_t + lag
(t = pts_time on each file's own timeline, the pipeline_eval frame identity). Same method as verify_stream_sync.py
(global frame-to-frame shift by phase correlation, both velocity signals on one grid, cross-correlation), but each
frame is placed at its own pts instead of index / fps: these recordings start mid-stream and the day stream's frame
gaps vary (31-35 ms), so a frame count drifts.

A lag is accepted only if --n-windows independent windows agree with the global peak (multi-window rule).

    python3 src/data/sync_ts.py --day ../data/arsuf_1005/day_output.ts --thermal ../data/arsuf_1005/th_output.ts \
        --out reports/stream_sync_arsuf1005
"""
import os
import re
import json
import argparse
import subprocess

import cv2
import numpy as np

from verify_stream_sync import xcorr, peak_stats

GRID_HZ = 50.0
INFO = re.compile(r"pts_time:([0-9.]+)")


def motion_ts(path, w, h):
    """[(t, dx, dy, response)] per decoded frame, t = pts_time."""
    p = subprocess.Popen(["ffmpeg", "-v", "info", "-copyts", "-threads", "0", "-i", path, "-vsync", "0",
                          "-vf", f"scale={w}:{h}:flags=area,format=gray,showinfo", "-f", "rawvideo", "pipe:1"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 64)
    import threading, queue
    q = queue.Queue()

    def err():
        for line in iter(p.stderr.readline, b""):
            s = line.decode("utf-8", "replace")
            if "Parsed_showinfo" in s:
                m = INFO.search(s)
                if m:
                    q.put(float(m.group(1)))
        q.put(None)
    threading.Thread(target=err, daemon=True).start()
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    prev, rows = None, []
    while True:
        t = q.get()
        if t is None:
            break
        buf = p.stdout.read(w * h)
        if len(buf) < w * h:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)
        cur = cur - cv2.GaussianBlur(cur, (0, 0), 8)
        if prev is not None:
            (dx, dy), r = cv2.phaseCorrelate(prev, cur, win)
            rows.append((t, dx, dy, r))
        prev = cur
    p.wait()
    return np.array(rows)


def to_grid(m, t0, t1):
    """velocity px/s at each frame's own time -> common grid [t0, t1)."""
    dt = np.diff(m[:, 0], prepend=m[0, 0] - np.median(np.diff(m[:, 0])))
    v = m[:, 1:3] / np.maximum(dt, 1e-3)[:, None]
    v[m[:, 3] < 0.05] = 0.0
    g = np.arange(t0, t1, 1 / GRID_HZ)
    out = np.stack([np.interp(g, m[:, 0], v[:, k]) for k in range(2)], 1)
    med = np.median(out, 0); mad = np.median(np.abs(out - med), 0) * 1.4826 + 1e-6
    return g, np.clip(out, med - 6 * mad, med + 6 * mad)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True); ap.add_argument("--thermal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=float, default=20.0); ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.04, help="windows must agree within this many seconds")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "motion.npz")
    if os.path.exists(cache):
        z = np.load(cache); md, mt = z["day"], z["th"]
    else:
        md = motion_ts(a.day, 384, 216); mt = motion_ts(a.thermal, 320, 256)
        np.savez(cache, day=md, th=mt)
    t0, t1 = max(md[0, 0], mt[0, 0]), min(md[-1, 0], mt[-1, 0])
    # one grid on the DAY clock; the thermal signal is evaluated at day_t + lag via its own grid shifted by lag
    gd, vd = to_grid(md, t0, t1)
    gt, vt = to_grid(mt, t0, t1)
    lags = np.arange(-int(a.max_lag * GRID_HZ), int(a.max_lag * GRID_HZ) + 1)
    corr = xcorr(vd, vt, lags)
    L, pk, side = peak_stats(corr, lags, GRID_HZ * 1.0)
    lag = L / GRID_HZ
    # scale: thermal px per day px (from the regression of |v_th| on |v_day| at the found lag)
    sh = int(round(L))
    a_, b_ = (vd[:len(vd) - sh], vt[sh:]) if sh >= 0 else (vd[-sh:], vt[:len(vt) + sh])
    n = min(len(a_), len(b_)); sp_d = np.hypot(*a_[:n].T); sp_t = np.hypot(*b_[:n].T)
    ok = (sp_d > np.percentile(sp_d, 60))
    scale = float(np.median(sp_t[ok] / np.maximum(sp_d[ok], 1e-6)))
    wins = []
    edges = np.linspace(0, len(vd), a.n_windows + 1).astype(int)
    for i in range(a.n_windows):
        s, e = edges[i], edges[i + 1]
        lo = max(0, s - int(a.max_lag * GRID_HZ)); hi = min(len(vt), e + int(a.max_lag * GRID_HZ))
        c = []
        for Lw in lags:
            ta, tb = s + Lw, e + Lw
            if ta < 0 or tb > len(vt):
                c.append(np.nan); continue
            x, y = vd[s:e], vt[ta:tb]
            x = (x - x.mean(0)) / (x.std(0) + 1e-9); y = (y - y.mean(0)) / (y.std(0) + 1e-9)
            c.append(float((x * y).mean()))
        c = np.array(c)
        Lw, pw, sw = peak_stats(c, lags, GRID_HZ * 1.0)
        wins.append({"t_mid": float(gd[(s + e) // 2]), "lag": Lw / GRID_HZ, "peak": pw, "side": sw})
    agree = [abs(w["lag"] - lag) <= a.tol for w in wins]
    rep = {"lag_s": lag, "definition": "thermal_t = day_t + lag_s (pts_time on each file)", "peak": pk,
           "sidelobe": side, "thermal_px_per_day_px": scale, "windows": wins,
           "windows_agree": int(sum(agree)), "n_windows": a.n_windows,
           "verdict": "PASS" if sum(agree) >= a.n_windows - 1 and pk - side > 0.1 else "FAIL",
           "day_frames": len(md) + 1, "thermal_frames": len(mt) + 1}
    json.dump(rep, open(os.path.join(a.out, "sync_report.json"), "w"), indent=1)
    print(f"lag {lag:+.3f} s (thermal_t = day_t + lag), peak {pk:.3f} vs sidelobe {side:.3f}, "
          f"scale {scale:.3f} thermal px / day px (at 384-px day decode), windows agreeing {sum(agree)}/{a.n_windows}: "
          f"{rep['verdict']}")
    for w in wins:
        print(f"  t~{w['t_mid']:6.0f}s lag {w['lag']:+.3f} peak {w['peak']:.2f} side {w['side']:.2f}")


if __name__ == "__main__":
    main()

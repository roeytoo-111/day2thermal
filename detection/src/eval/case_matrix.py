r"""
Per-case scorecard for the day + thermal pair: which kinds of drone each camera finds, and whether the two cover each other.

Every labelled THERMAL frame with a drone (pipeline_eval labels.csv) is tagged from the pixels of BOTH cameras at that moment
(day frame = thermal_t - lag, drone box mapped with the registration):
  shade        relative to the ring around the box (0.75 x box size): white = 95th percentile above the ring median by
               >= --shade-min grey levels and more than the dark side; black = 5th percentile below by >= --shade-min
               and more than the bright side; grey = neither (low contrast). Measured separately in thermal and in day.
  background   from the ring's grey-level std in the day frame (day sees the scene best): smooth (< 6: clear or overcast
               sky), cloud (6-20: cloud texture, edges, haze over terrain), ground (> 20: terrain, buildings, vegetation)
  size         thermal box long side: < 10 / 10-25 / >= 25 px
Then, at ONE operating point per camera (default: the pipeline's thresholds, thermal 0.40, day 0.40), per tag:
  thermal recall, day recall, blend recall (either camera), and found by both / thermal only / day only / neither.
Empty frames give each camera's and the blend's false-fire rate at the same thresholds.

    python3 src/eval/case_matrix.py --gt ../data/boson_work/cut0858/pipeline_gt/boson_0858 \
        --affine ../data/boson_work/cut0858/day2th_affine.npy --lag -0.1167 --day-tol 0.06 \
        --thermal-video ../data/boson_work/cut0858/2026-10-06T08_58_15_thermal_from_281.ts \
        --day-video ../data/boson_work/cut0858/2026-10-06T08_58_15_from_281.ts --cache ../data/gtfix/frame_cache \
        --thermal M0u_s1 ../data/boson_work/cut0858/dets/th_bosonM0u_s1.json --day fw4 ../data/rgb_dets/day_fw4.json
"""
import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_arsuf_gt import load_dets, hit       # noqa: E402
from blend_cameras import map_box               # noqa: E402
from review_day_gt import Frames                # noqa: E402


def shade_of(g, b, smin):
    H, W = g.shape
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None, None
    p = max(3, int(0.75 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    reg = g[Y0:Y1, X0:X1].astype(np.float32); m = np.ones(reg.shape, bool); m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    if m.sum() < 8:
        return None, None
    ring = reg[m]; med = np.median(ring); inner = g[y0:y1, x0:x1].astype(np.float32)
    br, dk = np.percentile(inner, 95) - med, med - np.percentile(inner, 5)
    shade = "white" if br >= smin and br > dk else "black" if dk >= smin and dk > br else "grey"
    return shade, float(ring.std())


def inv_affine(A):
    M = np.vstack([A, [0, 0, 1]]); return np.linalg.inv(M)[:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--affine", required=True); ap.add_argument("--lag", type=float, required=True)
    ap.add_argument("--thermal-video", required=True); ap.add_argument("--day-video", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--thermal", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--day", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--th-thr", type=float, default=0.40); ap.add_argument("--day-thr", type=float, default=0.40)
    ap.add_argument("--day-tol", type=float, default=0.06); ap.add_argument("--tol", type=float, default=0.021)
    ap.add_argument("--shade-min", type=float, default=20.0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lab = pd.read_csv(os.path.join(a.gt, "labels.csv"), dtype={"class": str}).rename(columns={"class": "cls"})
    lab = lab[lab.camera == "thermal"].copy(); lab["cls"] = lab.cls.fillna(""); lab["difficult"] = lab.difficult.fillna(0)
    A = np.load(a.affine); Ai = inv_affine(A)
    TH = {n: load_dets(p) for n, p in a.thermal}; DY = {n: load_dets(p) for n, p in a.day}
    dts0 = next(iter(DY.values()))[0]; tts0 = next(iter(TH.values()))[0]
    frames = []                                    # (t_thermal, t_day, kind, gt dict or None)
    for t, g in lab.groupby("t"):
        if np.min(np.abs(tts0 - t)) > a.tol:
            continue
        k = int(np.argmin(np.abs(dts0 - (t - a.lag))))
        if abs(dts0[k] - (t - a.lag)) > a.day_tol:
            continue
        obj = g[g.cls == "uav"]; gs = obj[obj.difficult == 0].to_dict("records")
        if len(obj) and not gs:
            continue
        frames.append((float(t), float(dts0[k]), "pos" if gs else "neg", gs[0] if gs else None))
    # pixel tags (decode only the drone frames)
    # thermal: sequential decode by frame index (the thermal JSONs hold every frame, so index j <-> its t); the exact-pts
    # extractor missed ~2/3 of these thermal frames
    want = {int(np.argmin(np.abs(tts0 - f[0]))): f[0] for f in frames if f[2] == "pos"}
    thf, cap, i = {}, cv2.VideoCapture(a.thermal_video), 0
    while want and i <= max(want):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            thf[want[i]] = im
        i += 1
    fdy = Frames(a.day_video, a.cache, [f[1] for f in frames if f[2] == "pos"], keep=80)    # a whole keyframe segment of 4K frames
    rows = []
    for t, td, kind, g in frames:
        r = {"t": t, "kind": kind}
        if kind == "pos":
            bt = [g["left"], g["top"], g["left"] + g["width"], g["top"] + g["height"]]
            r["size"] = max(g["width"], g["height"])
            im = thf.get(t)
            if im is not None:
                r["th_shade"], _ = shade_of(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), bt, a.shade_min)
            dm = fdy.get(td)
            if dm is not None:
                bd = map_box(bt, Ai)
                r["day_shade"], ring_std = shade_of(cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY), bd, a.shade_min)
                r["background"] = None if ring_std is None else ("smooth" if ring_std < 6 else "cloud" if ring_std <= 20 else "ground")
        for n, (ts, dets) in TH.items():
            j = int(np.argmin(np.abs(ts - t)))
            d = [x for x in dets[j] if x[4] >= a.th_thr]
            r[f"th:{n}"] = any(hit(x, g) for x in d) if kind == "pos" else bool(d)
        for n, (ts, dets) in DY.items():
            j = int(np.argmin(np.abs(ts - td)))
            d = []
            for x in dets[j]:
                if x[4] < a.day_thr:
                    continue
                b = map_box(x[:4], A); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if 0 <= cx < 640 and 0 <= cy < 512:
                    d.append(b + [x[4]])
            r[f"day:{n}"] = any(hit(x, g) for x in d) if kind == "pos" else bool(d)
        rows.append(r)
    D = pd.DataFrame(rows)
    if a.out:
        D.to_csv(a.out.replace(".md", ".csv"), index=False)
    P, N = D[D.kind == "pos"].copy(), D[D.kind == "neg"]
    P["size_bin"] = pd.cut(P["size"], [0, 10, 25, 1e9], labels=["<10", "10-25", ">=25"], right=False).astype(str)
    out = [f"# Case matrix: {os.path.basename(a.gt.rstrip('/'))}  (thermal >= {a.th_thr}, day >= {a.day_thr}; {len(P)} drone / {len(N)} empty frames)", ""]
    for tn in TH:
        for dn in DY:
            th, dy = P[f"th:{tn}"], P[f"day:{dn}"]
            out += [f"## thermal {tn} + day {dn}", "",
                    f"False fire on empty frames: thermal {100 * N[f'th:{tn}'].mean():.1f}%, day {100 * N[f'day:{dn}'].mean():.1f}%, "
                    f"blend {100 * (N[f'th:{tn}'] | N[f'day:{dn}']).mean():.1f}%", "",
                    "| case | frames | thermal | day | blend | both / th only / day only / neither |", "|---|---|---|---|---|---|"]
            for col in ("all", "th_shade", "day_shade", "background", "size_bin"):
                groups = [("all", P.index)] if col == "all" else [(f"{col} = {v}", P.index[P[col] == v]) for v in P[col].dropna().unique()]
                for name, idx in groups:
                    if len(idx) == 0:
                        continue
                    t_, d_ = th[idx], dy[idx]
                    out.append(f"| {name} | {len(idx)} | {100 * t_.mean():.0f}% | {100 * d_.mean():.0f}% | {100 * (t_ | d_).mean():.0f}% | "
                               f"{100 * (t_ & d_).mean():.0f} / {100 * (t_ & ~d_).mean():.0f} / {100 * (~t_ & d_).mean():.0f} / {100 * (~t_ & ~d_).mean():.0f} |")
            out.append("")
    txt = "\n".join(out); print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()

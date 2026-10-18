r"""
How well do the day and thermal detectors COMPLEMENT each other? On a video with both cameras and a timestamp-keyed GT
(pipeline_eval labels.csv), every labelled THERMAL frame gets its thermal detections plus the day detections of the
matching moment (thermal_t = day_t + lag), day boxes mapped into thermal pixels with the registration (day2th_affine.npy,
2x3, day px -> thermal px) and dropped when they fall outside the thermal frame. A hit is a box on the labelled drone
(score_arsuf_gt.hit). The cameras are blended with two thresholds (a thermal, b day): a frame counts as found if EITHER
camera hits the drone at its threshold, and as a false fire if EITHER has any uav box above its threshold on a frame
without a drone. Reported per frame set:
  thermal alone / day alone  recall at false fire <= 5, 10, 25 % and mean recall over FF 1-25 % (as ff_sweep.py)
  blend                      best recall over the (a, b) grid at the same false-fire levels, and mean recall FF 1-25 %
  overlap at the blend's FF<=10 % point   found by both / thermal only / day only / neither
The day frame nearest to the thermal frame is used when it is within --day-tol seconds (day detections are sparse: every
3rd/6th frame); labelled thermal frames without one are skipped for ALL columns so the comparison stays like for like.

    python3 src/eval/blend_cameras.py --gt ../data/boson_work/cut0858/pipeline_gt/boson_0858 \
        --affine ../data/boson_work/cut0858/day2th_affine.npy --lag -0.1167 --day-tol 0.06 \
        --thermal bosonM0u_s1 ../data/boson_work/cut0858/dets/th_bosonM0u_s1.json --day fw4 ../data/rgb_dets/day_fw4.json
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_arsuf_gt import load_dets, hit   # noqa: E402

THR = np.round(np.arange(0.05, 0.96, 0.05), 2)


def map_box(b, A):
    xs, ys = np.array([b[0], b[2]]), np.array([b[1], b[3]])
    X = A[0, 0] * xs[:, None] + A[0, 1] * ys[None, :] + A[0, 2]
    Y = A[1, 0] * xs[:, None] + A[1, 1] * ys[None, :] + A[1, 2]
    return [X.min(), Y.min(), X.max(), Y.max()]


def frame_scores(lab, th, day, A, lag, tol, day_tol, size=(640, 512)):
    """rows: kind, th_hit, th_neg, day_hit, day_neg, size. *_hit = best conf of a box on the drone, *_neg = best conf of any box."""
    tts, tdets = th; dts, ddets = day; rows = []
    for t, g in lab.groupby("t"):
        j = int(np.argmin(np.abs(tts - t)))
        if abs(tts[j] - t) > tol:
            continue
        k = int(np.argmin(np.abs(dts - (t - lag))))
        if abs(dts[k] - (t - lag)) > day_tol:
            continue
        obj = g[g.cls == "uav"]
        gs = obj[obj.difficult == 0].to_dict("records")
        if len(obj) and not gs:
            continue                                   # only 'difficult' drones: not scored (as score_arsuf_gt)
        mapped = []
        for d in ddets[k]:
            b = map_box(d[:4], A); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if 0 <= cx < size[0] and 0 <= cy < size[1]:
                mapped.append([*b, d[4]])
        td = tdets[j]
        if gs:
            th_hit = max([d[4] for d in td for gg in gs if hit(d, gg)], default=0.0)
            dy_hit = max([d[4] for d in mapped for gg in gs if hit(d, gg)], default=0.0)
            rows.append(("pos", th_hit, 0.0, dy_hit, 0.0, max(max(x["width"], x["height"]) for x in gs)))
        else:
            rows.append(("neg", 0.0, max([d[4] for d in td], default=0.0), 0.0, max([d[4] for d in mapped], default=0.0), np.nan))
    return pd.DataFrame(rows, columns=["kind", "th_hit", "th_neg", "day_hit", "day_neg", "size"])


def best_at_ff(F, R, ff):
    ok = F <= ff + 1e-9
    return R[ok].max() if ok.any() else 0.0


def single(df, cam):
    pos, neg = df[df.kind == "pos"][f"{cam}_hit"].values, df[df.kind == "neg"][f"{cam}_neg"].values
    R = np.array([(pos >= t).mean() for t in THR]); F = np.array([(neg >= t).mean() for t in THR])
    return F, R


def blend(df):
    P, N = df[df.kind == "pos"], df[df.kind == "neg"]
    Fs, Rs, pts = [], [], []
    for a in list(THR) + [1.01]:
        for b in list(THR) + [1.01]:
            Rs.append(((P.th_hit >= a) | (P.day_hit >= b)).mean()); Fs.append(((N.th_neg >= a) | (N.day_neg >= b)).mean()); pts.append((a, b))
    return np.array(Fs), np.array(Rs), pts


def summarize(df):
    out = {}
    for name, (F, R) in (("thermal", single(df, "th")), ("day", single(df, "day"))):
        out[name] = [best_at_ff(F, R, f) * 100 for f in (0.05, 0.10, 0.25)] + [np.mean([best_at_ff(F, R, f / 100) for f in range(1, 26)]) * 100]
    F, R, pts = blend(df)
    out["blend"] = [best_at_ff(F, R, f) * 100 for f in (0.05, 0.10, 0.25)] + [np.mean([best_at_ff(F, R, f / 100) for f in range(1, 26)]) * 100]
    ok = F <= 0.10 + 1e-9
    i = int(np.argmax(np.where(ok, R, -1)))
    a, b = pts[i]; P = df[df.kind == "pos"]
    th, dy = P.th_hit >= a, P.day_hit >= b
    ov = {"both": (th & dy).mean() * 100, "thermal only": (th & ~dy).mean() * 100, "day only": (~th & dy).mean() * 100, "neither": (~th & ~dy).mean() * 100}
    return out, (a, b), ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--affine", required=True); ap.add_argument("--lag", type=float, required=True)
    ap.add_argument("--day-tol", type=float, default=0.06); ap.add_argument("--tol", type=float, default=0.021)
    ap.add_argument("--thermal", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--day", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--small", type=float, default=25.0, help="px (thermal) below which a drone counts as small")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lab = pd.read_csv(os.path.join(a.gt, "labels.csv"), dtype={"class": str}).rename(columns={"class": "cls"})
    lab = lab[lab.camera == "thermal"].copy(); lab["cls"] = lab.cls.fillna(""); lab["difficult"] = lab.difficult.fillna(0)
    A = np.load(a.affine)
    L = []
    for dn, dp in a.day:
        day = load_dets(dp)
        for tn, tp in a.thermal:
            df = frame_scores(lab, load_dets(tp), day, A, a.lag, a.tol, a.day_tol)
            for sub, d in (("all", df), (f"drone < {a.small:g} px", df[(df.kind == "neg") | (df["size"] < a.small)]),
                           (f"drone >= {a.small:g} px", df[(df.kind == "neg") | (df["size"] >= a.small)])):
                o, (ta, tb), ov = summarize(d)
                for cam in ("thermal", "day", "blend"):
                    L.append({"thermal": tn, "day": dn, "frames": sub, "camera": cam, "pos": int((d.kind == "pos").sum()), "neg": int((d.kind == "neg").sum()),
                              "R@FF5": o[cam][0], "R@FF10": o[cam][1], "R@FF25": o[cam][2], "mean FF1-25": o[cam][3],
                              "blend point (a thermal, b day)": f"{ta:.2f}, {tb:.2f}" if cam == "blend" else "",
                              "found by both / th only / day only / neither (%)": " / ".join(f"{ov[k]:.0f}" for k in ("both", "thermal only", "day only", "neither")) if cam == "blend" else ""})
    T = pd.DataFrame(L)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(T.round(1).to_string(index=False))
    if a.out:
        T.round(1).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

r"""
Failure-mode sheets for day-camera detections against a timestamp-keyed GT (pipeline_eval labels.csv):
  FP  empty GT frames where a model has a uav box >= --fp-conf          (false alarm: what is it?)
  FN  GT-drone frames with no hit >= --fn-conf                          (miss: what does the drone look like?)
  FPpos  uav boxes >= --fp-conf on drone frames that are NOT on the drone (second object / clutter)
Each event is a crop of the decoded frame; events are clustered over time (events within --merge-s of each other are one
episode) so the sheet shows episodes, not 30 near-identical frames. Models are compared by episode overlap: an episode
seen by every model is a data problem; one seen by a single model is training noise.

    python3 src/eval/failure_modes_day.py --gt ../data/boson_work/cut0858/pipeline_gt/boson_0858 \
        --video ../data/boson_work/cut0858/2026-10-06T08_58_15_from_281.ts --every 3 --out results/failmodes_0858 \
        --model fw4 ../data/rgb_dets/day_fw4.json --model v5s_s0 ../data/rgb_dets/day_v5s_s0.json
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_arsuf_gt import load_dets, hit   # noqa: E402


def episodes(times, gap):
    out, cur = [], []
    for t in sorted(times):
        if cur and t - cur[-1] > gap:
            out.append(cur); cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--every", type=int, default=3, help="the detection JSON keeps one decoded frame in N")
    ap.add_argument("--fp-conf", type=float, default=0.25); ap.add_argument("--fn-conf", type=float, default=0.15)
    ap.add_argument("--merge-s", type=float, default=1.0); ap.add_argument("--tol", type=float, default=0.021)
    ap.add_argument("--max-per-kind", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lab = pd.read_csv(os.path.join(a.gt, "labels.csv"), dtype={"class": str}).rename(columns={"class": "cls"})
    lab = lab[lab.camera == "day"].copy(); lab["cls"] = lab.cls.fillna(""); lab["difficult"] = lab.difficult.fillna(0)
    M = {n: load_dets(p) for n, p in a.model}
    ev = []        # (kind, model, t, box, conf, gt)
    for n, (ts, dets) in M.items():
        for t, g in lab.groupby("t"):
            j = int(np.argmin(np.abs(ts - t)))
            if abs(ts[j] - t) > a.tol:
                continue
            obj = g[(g.cls == "uav") & (g.difficult == 0)].to_dict("records")
            d = dets[j]
            if not obj:
                for x in d:
                    if x[4] >= a.fp_conf:
                        ev.append(("FP", n, t, x[:4], x[4], None))
            else:
                best = max([x[4] for x in d for gg in obj if hit(x, gg)], default=0.0)
                if best < a.fn_conf:
                    ev.append(("FN", n, t, None, best, obj[0]))
                for x in d:
                    if x[4] >= a.fp_conf and not any(hit(x, gg) for gg in obj):
                        ev.append(("FPpos", n, t, x[:4], x[4], obj[0]))
    E = pd.DataFrame(ev, columns=["kind", "model", "t", "box", "conf", "gt"])
    rows = []
    for kind in ("FP", "FN", "FPpos"):
        K = E[E.kind == kind]
        eps = episodes(K.t.unique(), a.merge_s)
        for k, e in enumerate(eps):
            sub = K[(K.t >= e[0]) & (K.t <= e[-1])]
            rows.append({"kind": kind, "episode": k, "t0": e[0], "t1": e[-1], "frames": len(set(sub.t)),
                         "models": ",".join(sorted(set(sub.model))), "n_models": sub.model.nunique(), "max_conf": sub.conf.max()})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "episodes.csv"), index=False)
    print(R.groupby("kind").agg(episodes=("episode", "count"), all_models=("n_models", lambda s: int((s == len(M)).sum())),
                                one_model=("n_models", lambda s: int((s == 1).sum()))).to_string())
    # crops: decode once, sequentially (cv2 reads these cut .ts files; index = every * json index)
    want = {}
    for r in R.itertuples():
        sub = E[(E.kind == r.kind) & (E.t >= r.t0) & (E.t <= r.t1)]
        t = float(sub.loc[sub.conf.idxmax(), "t"]) if len(sub) else r.t0
        ts0 = next(iter(M.values()))[0]
        want.setdefault(int(np.argmin(np.abs(ts0 - t))) * a.every, []).append((r, t, sub))
    cap = cv2.VideoCapture(a.video); i, last = 0, max(want) if want else -1
    shots = {}
    while i <= last:
        if i not in want:
            if not cap.grab():
                break
            i += 1; continue
        ok, im = cap.read()
        if not ok:
            break
        for r, t, sub in want[i]:
            H, W = im.shape[:2]
            ref = sub.iloc[(sub.conf).argmax()]
            if ref.box is not None:
                cx, cy = (ref.box[0] + ref.box[2]) / 2, (ref.box[1] + ref.box[3]) / 2; sz = max(4 * max(ref.box[2] - ref.box[0], ref.box[3] - ref.box[1]), 120)
            else:
                g = ref["gt"]; cx, cy = g["left"] + g["width"] / 2, g["top"] + g["height"] / 2; sz = max(4 * max(g["width"], g["height"]), 120)
            x0, y0 = int(np.clip(cx - sz / 2, 0, max(W - sz, 0))), int(np.clip(cy - sz / 2, 0, max(H - sz, 0)))
            c = im[y0:y0 + int(sz), x0:x0 + int(sz)].copy(); z = 200 / max(c.shape[:2])
            c = cv2.resize(c, (int(c.shape[1] * z), int(c.shape[0] * z)), interpolation=cv2.INTER_NEAREST if z > 2 else cv2.INTER_AREA)
            for _, row in sub[sub.t == t].iterrows():
                if row.box is not None:
                    b = row.box; cv2.rectangle(c, (int((b[0] - x0) * z), int((b[1] - y0) * z)), (int((b[2] - x0) * z), int((b[3] - y0) * z)), (255, 0, 255), 1)
            if ref["gt"] is not None:
                g = ref["gt"]; cv2.rectangle(c, (int((g["left"] - x0) * z), int((g["top"] - y0) * z)),
                                          (int((g["left"] + g["width"] - x0) * z), int((g["top"] + g["height"] - y0) * z)), (0, 255, 0), 1)
            T = np.zeros((222, 200, 3), np.uint8); T[:c.shape[0], :c.shape[1]] = c
            cv2.putText(T, f"{r.kind} t{t:.0f} {r.models.replace(',', '/')[:22]}", (2, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
            shots[(r.kind, r.episode)] = T
        i += 1
    for kind in ("FP", "FN", "FPpos"):
        keys = [k for k in shots if k[0] == kind]
        R_ = R[R.kind == kind].sort_values(["n_models", "frames"], ascending=False)
        pick = [(kind, int(e)) for e in R_.episode if (kind, int(e)) in shots][:a.max_per_kind]
        tiles = [shots[k] for k in pick]
        if not tiles:
            continue
        while len(tiles) % 8:
            tiles.append(np.zeros((222, 200, 3), np.uint8))
        cv2.imwrite(os.path.join(a.out, f"{kind}.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))


if __name__ == "__main__":
    main()

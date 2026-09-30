"""Survey recording sessions (thermal only) before extracting pairs: what content does each hold, and are
sessions redundant with each other?

Every --step-s seconds inside each session's window:
  smooth_frac  fraction of the frame that is smooth (local high-pass std < 4 grey levels) -- in LWIR that
               is sky (and sea); vegetation/terrain is strongly textured after the camera's detail
               enhancement. Bins: terrain (< 0.2), mixed, sky (> 0.5).
  descriptor   48x38 z-scored thumbnail, for similarity.
Redundancy: for every frame, the distance to its nearest neighbour in its OWN session (excluding +-30 s)
and in each OTHER session. A session whose frames are as close to another session as to themselves adds
little new content.

    python -m day2thermal.survey_sessions --session 0623s1:<thermal.mp4>:660:1140 --session ... --out <dir>
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from .utils import ensure_dir


def features(g):
    g = g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 2)
    local = np.sqrt(cv2.blur(hp * hp, (15, 15)))
    t = cv2.resize(g, (48, 38), interpolation=cv2.INTER_AREA).ravel()
    return float((local < 4).mean()), (t - t.mean()) / (t.std() + 1e-3)


def sample(path, t0, t1, step):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0, f1 = int(t0 * fps), min(n, int(t1 * fps)) if t1 > 0 else n
    want = set(range(f0, f1, max(1, int(step * fps))))
    out, i = [], 0
    while i < f1:
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, im = cap.retrieve()
            if ok:
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                sf, d = features(g)
                out.append((i / fps, sf, d, g))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append", required=True, help="name:path:start_s:end_s (end 0 = to end)")
    ap.add_argument("--step-s", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ensure_dir(a.out)
    data = {}
    for spec in a.session:
        name, path, t0, t1 = spec.rsplit(":", 3)
        data[name] = sample(path, float(t0), float(t1), a.step_s)
        print(f"  {name}: {len(data[name])} frames sampled", flush=True)
    rows = []
    for name, fr in data.items():
        for t, sf, _, _ in fr:
            rows.append((name, t, sf))
    d = pd.DataFrame(rows, columns=["session", "t", "smooth"])
    d["content"] = pd.cut(d.smooth, [-0.01, 0.2, 0.5, 1.01], labels=["terrain", "mixed", "sky"])
    tab = pd.crosstab(d.session, d.content, normalize="index").round(2)
    tab["frames"] = d.groupby("session").size()
    print("\nCONTENT (share of sampled frames; thermal smoothness):\n", tab)
    # redundancy
    names = list(data)
    desc = {n: np.stack([x[2] for x in data[n]]) for n in names}
    times = {n: np.array([x[0] for x in data[n]]) for n in names}
    red = pd.DataFrame(index=names, columns=names, dtype=float)
    for a_ in names:
        for b_ in names:
            D = np.abs(desc[a_][:, None, :] - desc[b_][None, :, :]).mean(-1)
            if a_ == b_:
                D[np.abs(times[a_][:, None] - times[a_][None, :]) < 30] = np.inf
            red.loc[a_, b_] = float(np.median(D.min(1)))
    print("\nNEAREST-NEIGHBOUR DISTANCE (median; row = session's frames, column = where the neighbour is searched;"
          "\ndiagonal = own session >30 s away; similar off-diagonal value = redundant content):\n", red.round(3))
    tab.to_csv(os.path.join(out, "content.csv"))
    red.to_csv(os.path.join(out, "nn_distance.csv"))
    # visual timeline: 8 frames per session
    rowsimg = []
    for n in names:
        fr = data[n]
        idx = np.linspace(0, len(fr) - 1, 8).astype(int)
        tiles = []
        for i in idx:
            im = cv2.cvtColor(cv2.resize(fr[i][3], (200, 160)), cv2.COLOR_GRAY2BGR)
            cv2.putText(im, f"{n} {fr[i][0] / 60:.1f}min", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(im)
        rowsimg.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(out, "timeline.jpg"), np.vstack(rowsimg))
    print(f"\nwrote {out}/ (content.csv, nn_distance.csv, timeline.jpg)")


if __name__ == "__main__":
    main()

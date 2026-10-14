r"""
One decision per object track from mine_rgb_tracks.py mine (<W>/tracks/<session>.json). Top: up to 8 zoomed crops spread
over the track (magenta box = the detection, green = its confidence). Bottom: the whole frame at the track's best
detection (magenta circle).

  u   drone (uav)          p   airplane          b   bird
  f   false alarm (cloud, terrain, glare, ...): its tiles become negatives
  x   unsure / mixed (track jumps between objects): never used
  space  skip (asked again)      z  back      q  save and quit (resumable)
No Enter key.

    python3 src/eval/review_rgb_tracks.py --W ../data/flight_20261014
    python3 src/eval/review_rgb_tracks.py --W ../data/flight_20261014 --min-n 3     # long tracks first pass
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np

KEYS = {ord("u"): "uav", ord("p"): "airplane", ord("b"): "bird", ord("f"): "f", ord("x"): "x"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--W", required=True); ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--session", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.W, "tracks", "*.json")))
    if a.session:
        files = [f for f in files if os.path.basename(f)[:-5] == a.session]
    if not files:
        raise SystemExit(f"no tracks in {a.W}/tracks yet: run detection (rgb_label_videos.py) and "
                         "mine_rgb_tracks.py mine first")
    docs = {f: json.load(open(f)) for f in files}
    todo = [(f, i) for f in files for i, t in enumerate(docs[f]["tracks"]) if t["decision"] == "" and t["n"] >= a.min_n]
    n_all = sum(len(d["tracks"]) for d in docs.values())
    save = lambda: [json.dump(d, open(f, "w")) for f, d in docs.items()]
    cv2.namedWindow("tracks")
    hist, j = [], 0
    while 0 <= j < len(todo):
        f, i = todo[j]; d = docs[f]; t = d["tracks"][i]; s = d["session"]
        cdir = os.path.join(a.W, "tracks", "crops")
        crops = [cv2.imread(os.path.join(cdir, f"{s}_{t['id']}_{n}.jpg")) for n in range(len(t["crop_idx"]))]
        crops = [c for c in crops if c is not None]
        strip = np.zeros((170, 8 * 165, 3), np.uint8)
        for n, c in enumerate(crops[:8]):
            strip[5:5 + c.shape[0], n * 165:n * 165 + c.shape[1]] = c
        ctx = cv2.imread(os.path.join(cdir, f"{s}_{t['id']}_ctx.jpg"))
        ctx = ctx if ctx is not None else np.zeros((540, 960, 3), np.uint8)
        W = max(strip.shape[1], ctx.shape[1])
        canvas = np.zeros((52 + strip.shape[0] + ctx.shape[0], W, 3), np.uint8)
        canvas[52:52 + strip.shape[0], :strip.shape[1]] = strip
        canvas[52 + strip.shape[0]:, :ctx.shape[1]] = ctx
        done = sum(1 for dd in docs.values() for tt in dd["tracks"] if tt["decision"])
        dur = (t["k"][-1] - t["k"][0]) / 30
        votes = " ".join(f"{k}:{v}" for k, v in sorted(t["cls_votes"].items(), key=lambda x: -x[1]))
        cv2.putText(canvas, f"[{done}/{n_all}] {s} track {t['id']}: {t['n']} dets over {dur:.1f}s, max conf {t['max_conf']:.2f}, model says {votes}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "u=drone  p=airplane  b=bird  f=false alarm  x=unsure/mixed  space=skip  z=back  q=quit",
                    (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("tracks", canvas)
        c = cv2.waitKey(0) & 0xFF
        if c in KEYS:
            t["decision"] = KEYS[c]; hist.append(j); j += 1
        elif c == ord(" "):
            hist.append(j); j += 1
        elif c == ord("z") and hist:
            j = hist.pop(); docs[todo[j][0]]["tracks"][todo[j][1]]["decision"] = ""
        elif c == ord("q"):
            save(); print("saved"); return
        if len(hist) % 25 == 0:
            save()
    save(); print("all tracks decided; run mine_rgb_tracks.py export")


if __name__ == "__main__":
    main()

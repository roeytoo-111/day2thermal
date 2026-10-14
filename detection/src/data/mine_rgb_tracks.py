r"""
RGB-only flight videos (no thermal witness) -> object tracks for one human decision each -> training tiles.

  mine    link the day model's tiled detections (rgb_label_videos.py JSON, conf >= --link-conf, any class) into
          tracks; keep tracks whose best confidence reaches --alarm-conf (what the deployed model would report),
          i.e. every true and false alarm. Low-confidence detections linked to such a track stay in it: on a drone
          track they are the frames where the model nearly lost the drone (hard positives).
          Writes <W>/tracks/<session>.json and review crops (<W>/tracks/crops/<session>_<tid>_{0..7}.jpg strip +
          _ctx.jpg frame thumbnail) for review_rgb_tracks.py.
  export  cut 640 tiles from the decided tracks into a YOLO set <W>/rgb_tracks_addon/train (names uav, airplane, bird):
            uav / airplane / bird tracks: one tile every --step processed frames (box at a random place in the tile)
            false-alarm tracks: one tile every --fp-step processed frames around the false box -> negative
          Every decided object at that frame inside the tile is labelled with its decided class (false alarms are
          left unlabelled = background). A tile is skipped if it holds a box from an undecided / unsure track, or
          the interpolated box of a decided object track that the model missed at that frame.

    python3 src/data/mine_rgb_tracks.py mine   --W ../data/flight_20261014 --dets-dir ../data/flight_20261014/rgbdets
    python3 src/eval/review_rgb_tracks.py --W ../data/flight_20261014
    python3 src/data/mine_rgb_tracks.py export --W ../data/flight_20261014
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np

TILE, MARGIN = 640, 32
NAMES = ["uav", "airplane", "bird"]


def link(frames, conf, radius, max_gap):
    """Class-agnostic constant-velocity linking (as prefill_from_tracks.link). frames: list of (k, [[x0,y0,x1,y1,c,cls]])."""
    tracks, active = [], []
    for k, dets in frames:
        dets = [d for d in dets if d[4] >= conf]
        used = set()
        for tr in sorted(active, key=lambda t: -len(t["k"])):
            lk, lb = tr["k"][-1], np.array(tr["b"][-1])
            v = (lb - np.array(tr["b"][-2])) / max(lk - tr["k"][-2], 1) if len(tr["k"]) >= 2 else np.zeros(4)
            pred = lb + v * (k - lk)
            pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
            size = max(lb[2] - lb[0], lb[3] - lb[1])
            r = max(radius, 2 * size) * (1 + 0.25 * (k - lk - 1))
            best, bd = None, r
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dd = np.hypot((d[0] + d[2]) / 2 - pc[0], (d[1] + d[3]) / 2 - pc[1])
                if dd < bd:
                    best, bd = j, dd
            if best is not None:
                used.add(best); d = dets[best]
                tr["k"].append(k); tr["b"].append([round(x, 1) for x in d[:4]]); tr["c"].append(round(d[4], 3)); tr["cls"].append(d[5])
        for j, d in enumerate(dets):
            if j not in used:
                t = {"k": [k], "b": [[round(x, 1) for x in d[:4]]], "c": [round(d[4], 3)], "cls": [d[5]]}
                tracks.append(t); active.append(t)
        active = [t for t in active if k - t["k"][-1] <= max_gap]
    return tracks


def load_dets(path):
    D = json.load(open(path))
    return [(e["src_frame"], [[*d["bbox"], d["conf"], d["cls"]] for d in e["detections"]]) for e in D]


def interp(tr, k):
    ks = tr["k"]
    if k < ks[0] or k > ks[-1]:
        return None
    B = np.array(tr["b"])
    return [float(np.interp(k, ks, B[:, i])) for i in range(4)]


def crop(im, b, out=160, minside=48):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(3 * max(b[2] - b[0], b[3] - b[1]), minside)
    H, W = im.shape[:2]
    x0, y0 = int(np.clip(cx - s / 2, 0, max(W - s, 0))), int(np.clip(cy - s / 2, 0, max(H - s, 0)))
    c = im[y0:y0 + int(s), x0:x0 + int(s)].copy()
    z = out / max(c.shape[:2])
    c = cv2.resize(c, (int(c.shape[1] * z), int(c.shape[0] * z)), interpolation=cv2.INTER_NEAREST if z > 2 else cv2.INTER_AREA)
    cv2.rectangle(c, (int((b[0] - x0) * z) - 3, int((b[1] - y0) * z) - 3), (int((b[2] - x0) * z) + 3, int((b[3] - y0) * z) + 3), (255, 0, 255), 1)
    return c


def cmd_mine(a):
    os.makedirs(os.path.join(a.W, "tracks", "crops"), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.dets_dir, "*.json"))):
        s = os.path.basename(f)[:-5]
        out = os.path.join(a.W, "tracks", s + ".json")
        if os.path.exists(out) and not a.force:
            print("exists", out); continue
        frames = load_dets(f)
        step = int(np.median(np.diff([k for k, _ in frames]))) if len(frames) > 1 else 1
        tr = link(frames, a.link_conf, a.radius, a.max_gap * step)
        tr = [t for t in tr if max(t["c"]) >= a.alarm_conf]
        tr.sort(key=lambda t: t["k"][0])
        for i, t in enumerate(tr):
            t.update({"id": i, "max_conf": max(t["c"]), "n": len(t["k"]), "decision": "",
                      "cls_votes": {c: t["cls"].count(c) for c in set(t["cls"])}})
        # crops: 8 frames spread over the track (+ the best frame) and a whole-frame thumbnail of the best frame
        want = {}
        for t in tr:
            j = sorted(set(np.linspace(0, len(t["k"]) - 1, min(8, len(t["k"]))).round().astype(int).tolist()))
            t["crop_idx"] = j
            for n_, jj in enumerate(j):
                want.setdefault(t["k"][jj], []).append((t["id"], n_, t["b"][jj], t["c"][jj]))
            jb = int(np.argmax(t["c"]))
            want.setdefault(t["k"][jb], []).append((t["id"], "ctx", t["b"][jb], t["c"][jb]))
        cap = cv2.VideoCapture(os.path.join(a.W, "mp4", f"{s}_day.mp4"))
        i, last = 0, max(want) if want else -1
        while i <= last:
            if i not in want:
                if not cap.grab():
                    break
                i += 1; continue
            ok, im = cap.read()
            if not ok:
                break
            for tid, n_, b, c in want[i]:
                if n_ == "ctx":
                    th = cv2.resize(im, (960, int(960 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
                    z = 960 / im.shape[1]
                    cx, cy = int((b[0] + b[2]) / 2 * z), int((b[1] + b[3]) / 2 * z)
                    cv2.circle(th, (cx, cy), 18, (255, 0, 255), 2)
                    cv2.imwrite(os.path.join(a.W, "tracks", "crops", f"{s}_{tid}_ctx.jpg"), th, [cv2.IMWRITE_JPEG_QUALITY, 90])
                else:
                    c_ = crop(im, b)
                    cv2.putText(c_, f"{c:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(a.W, "tracks", "crops", f"{s}_{tid}_{n_}.jpg"), c_, [cv2.IMWRITE_JPEG_QUALITY, 92])
            i += 1
        json.dump({"session": s, "step": step, "args": vars(a), "tracks": tr}, open(out, "w"))
        lens = [t["n"] for t in tr]
        print(f"{s}: {len(tr)} tracks with max conf >= {a.alarm_conf} (median {np.median(lens) if lens else 0:.0f} detections, "
              f"{sum(n >= 10 for n in lens)} with >= 10)", flush=True)


def place(box, W, H, rng):
    x0, y0, x1, y1 = box
    lo_x, hi_x = max(0, int(x1) + MARGIN - TILE), min(W - TILE, int(x0) - MARGIN)
    lo_y, hi_y = max(0, int(y1) + MARGIN - TILE), min(H - TILE, int(y0) - MARGIN)
    if x1 - x0 > TILE - 2 * MARGIN or y1 - y0 > TILE - 2 * MARGIN or lo_x > hi_x or lo_y > hi_y:
        return None
    return int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1))


def inside(b, tx, ty):
    return b[2] > tx and b[0] < tx + TILE and b[3] > ty and b[1] < ty + TILE


def cmd_export(a):
    rng = np.random.default_rng(a.seed)
    out = os.path.join(a.W, "rgb_tracks_addon")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(out, "train", sub), exist_ok=True)
    tot = {"uav": 0, "airplane": 0, "bird": 0, "neg": 0, "skipped": 0}
    for f in sorted(glob.glob(os.path.join(a.W, "tracks", "*.json"))):
        T = json.load(open(f)); s, step = T["session"], T["step"]; tr = T["tracks"]
        obj = [t for t in tr if t["decision"] in NAMES]
        fps = [t for t in tr if t["decision"] == "f"]
        open_ = [t for t in tr if t["decision"] not in NAMES + ["f"]]
        at = {}                     # frame -> list of (track, box at that frame or None if interpolated)
        for t in tr:
            for k, b in zip(t["k"], t["b"]):
                at.setdefault(k, []).append((t, b))
        want = {}
        for t in obj:
            for k, b in list(zip(t["k"], t["b"]))[::a.step]:
                want.setdefault(k, []).append((t, b))
        for t in fps:
            for k, b in list(zip(t["k"], t["b"]))[::a.fp_step]:
                want.setdefault(k, []).append((t, b))
        if not want:
            continue
        cap = cv2.VideoCapture(os.path.join(a.W, "mp4", f"{s}_day.mp4"))
        i, last = 0, max(want)
        while i <= last:
            if i not in want:
                if not cap.grab():
                    break
                i += 1; continue
            ok, im = cap.read()
            if not ok:
                break
            H, W = im.shape[:2]
            for n_, (t, b) in enumerate(want[i]):
                pos = place(b, W, H, rng)
                if pos is None:
                    tot["skipped"] += 1; continue
                tx, ty = pos
                bad = any(inside(bb, tx, ty) and tt["decision"] not in NAMES + ["f"] and max(tt["c"]) >= a.alarm_conf
                          for tt, bb in at.get(i, []))
                present = {id(tt) for tt, _ in at.get(i, [])}
                bad |= any(id(o) not in present and (ib := interp(o, i)) is not None and inside(ib, tx, ty) for o in obj)
                if bad:
                    tot["skipped"] += 1; continue
                lines = []
                for tt, bb in at.get(i, []):
                    if tt["decision"] in NAMES and inside(bb, tx, ty):
                        x0, y0 = max(bb[0], tx), max(bb[1], ty); x1, y1 = min(bb[2], tx + TILE), min(bb[3], ty + TILE)
                        lines.append(f"{NAMES.index(tt['decision'])} {((x0 + x1) / 2 - tx) / TILE:.6f} {((y0 + y1) / 2 - ty) / TILE:.6f} "
                                     f"{(x1 - x0) / TILE:.6f} {(y1 - y0) / TILE:.6f}\n")
                stem = f"{'trk' if t['decision'] in NAMES else 'trkfp'}{s}_{i:06d}_{n_}"
                cv2.imwrite(os.path.join(out, "train", "images", stem + ".jpg"), im[ty:ty + TILE, tx:tx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 95])
                open(os.path.join(out, "train", "labels", stem + ".txt"), "w").write("".join(lines))
                tot[t["decision"] if t["decision"] in NAMES else "neg"] += 1
            i += 1
        print(s, tot, flush=True)
    open(os.path.join(out, "data.yaml"), "w").write(f"path: {os.path.abspath(out)}\ntrain: train/images\nval: train/images\n\nnc: 3\nnames: {NAMES}\n")
    print("tiles ->", out, tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mine", "export"]); ap.add_argument("--W", required=True)
    ap.add_argument("--dets-dir", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--link-conf", type=float, default=0.05); ap.add_argument("--alarm-conf", type=float, default=0.15)
    ap.add_argument("--radius", type=float, default=40, help="4K px"); ap.add_argument("--max-gap", type=int, default=5,
                                                                                         help="processed frames")
    ap.add_argument("--step", type=int, default=5, help="object tiles: every N-th detection of a track")
    ap.add_argument("--fp-step", type=int, default=3, help="false-alarm tiles: every N-th detection of a track")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "mine":
        a.dets_dir = a.dets_dir or os.path.join(a.W, "rgbdets")
        cmd_mine(a)
    else:
        cmd_export(a)


if __name__ == "__main__":
    main()

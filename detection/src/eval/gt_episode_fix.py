r"""
Repair a day-camera GT whose "empty" frames contain real objects. On 08_58_15 most confident "false alarms" of every model were
real aircraft (a drone low over the sea, wings on the runway edge, ...): the GT rule "only the airborne drone is a drone"
turned them into false fires. This tool shows every episode where some model fires >= --fp-conf on an empty frame, or on a
drone frame away from the labelled drone, and lets you decide it once; it never edits the original GT.

  prepare   events -> object clusters (boxes of all models merged) -> episodes in time; first / middle / last crops
            and a whole-frame view per episode in <W>/crops
  review    one episode at a time (keys below); decisions saved to <W>/episodes.json, resumable
  apply     writes a corrected GT folder <--out> (labels.csv + gt.json + decisions copy): u -> uav rows (difficult 0),
            x -> uav rows with difficult = 1 (the frame drops out of the scoring); p / f / unreviewed -> unchanged

Keys (review):  u  it IS our drone (add to GT)     p  manned plane / other aircraft     b  bird
                f  nothing real (clutter, building, ground)               x  unsure (excluded from scoring)
                p / b / f keep the box a false alarm for the drone task (the model called it uav); the decision file keeps the
                type, so the false fires can be split into birds / manned aircraft / clutter.
                drag  draw the box for a real drone the models missed, then u (applies to this episode's middle frame only)
                space skip   z back   q save and quit.   No Enter key.

    python3 src/eval/gt_episode_fix.py prepare --W ../data/gtfix/0858 --gt ../data/boson_work/cut0858/pipeline_gt/boson_0858 \
        --video ../data/boson_work/cut0858/2026-10-06T08_58_15_from_281.ts --cache ../data/gtfix/frame_cache \
        --model fw4 ../data/rgb_dets/day_fw4.json --model v5s_s0 ../data/rgb_dets/day_v5s_s0.json ...
    python3 src/eval/gt_episode_fix.py review --W ../data/gtfix/0858
    python3 src/eval/gt_episode_fix.py apply --W ../data/gtfix/0858 --out ../data/boson_work/cut0858/pipeline_gt/boson_0858_v2
"""
import os
import sys
import json
import shutil
import argparse

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_arsuf_gt import load_dets, hit   # noqa: E402
from review_day_gt import Frames            # noqa: E402


def load_day_gt(gt):
    lab = pd.read_csv(os.path.join(gt, "labels.csv"), dtype={"class": str}).rename(columns={"class": "cls"})
    lab = lab[lab.camera == "day"].copy(); lab["cls"] = lab.cls.fillna(""); lab["difficult"] = lab.difficult.fillna(0)
    return lab


def clusters(dets_by_model, rad_min=20.0):
    """Merge the boxes of all models on one frame into object candidates: [x0,y0,x1,y1,maxconf,{models}]."""
    items = sorted([(d[4], m, d[:4]) for m, ds in dets_by_model.items() for d in ds], key=lambda x: -x[0])
    out = []
    for c, m, b in items:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(b[2] - b[0], b[3] - b[1])
        for o in out:
            ob = o["box"]; ox, oy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
            if np.hypot(cx - ox, cy - oy) <= max(rad_min, 0.75 * max(s, ob[2] - ob[0], ob[3] - ob[1])):
                o["w"].append((c, b)); o["models"].add(m); break
        else:
            out.append({"box": list(b), "w": [(c, b)], "models": {m}, "conf": c})
    for o in out:
        w = np.array([c for c, _ in o["w"]]); B = np.array([b for _, b in o["w"]])
        o["box"] = [float(v) for v in (B * w[:, None]).sum(0) / w.sum()]
    return out


def cmd_prepare(a):
    lab = load_day_gt(a.gt)
    M = {n: load_dets(p) for n, p in a.model}
    ts0 = next(iter(M.values()))[0]
    ev = []                                                     # (t, kind, box, conf, models)
    for t, g in lab.groupby("t"):
        dd = {}
        for n, (ts, dets) in M.items():
            j = int(np.argmin(np.abs(ts - t)))
            if abs(ts[j] - t) <= a.tol:
                dd[n] = [d for d in dets[j] if d[4] >= a.fp_conf]
        obj = g[(g.cls == "uav") & (g.difficult == 0)].to_dict("records")
        for c in clusters(dd):
            if obj and any(hit(c["box"], gg) for gg in obj):
                continue
            ev.append({"t": float(t), "kind": "FPpos" if obj else "FP", "box": c["box"], "conf": c["conf"], "models": sorted(c["models"])})
    ev.sort(key=lambda e: e["t"])
    eps = []                                                    # link events in time and space
    for e in ev:
        cx, cy = (e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2; s = max(e["box"][2] - e["box"][0], e["box"][3] - e["box"][1])
        for ep in eps:
            last = ep["events"][-1]
            lx, ly = (last["box"][0] + last["box"][2]) / 2, (last["box"][1] + last["box"][3]) / 2; ls = max(last["box"][2] - last["box"][0], last["box"][3] - last["box"][1])
            if 0 <= e["t"] - last["t"] <= a.merge_s and np.hypot(cx - lx, cy - ly) <= max(80.0, 2 * max(s, ls)):
                ep["events"].append(e); break
        else:
            eps.append({"events": [e]})
    for i, ep in enumerate(eps):
        E = ep["events"]
        ep.update({"id": i, "t0": E[0]["t"], "t1": E[-1]["t"], "n_frames": len(E), "max_conf": max(x["conf"] for x in E),
                   "models": sorted({m for x in E for m in x["models"]}), "kind": E[0]["kind"], "decision": "", "manual": None})
    os.makedirs(os.path.join(a.W, "crops"), exist_ok=True)
    picks = {}
    for ep in eps:
        E = ep["events"]; ep["shots"] = [E[0]["t"], E[len(E) // 2]["t"], E[-1]["t"]]
        for k, t in enumerate(ep["shots"]):
            picks.setdefault(t, []).append((ep["id"], k))
    fr = Frames(a.video, a.cache, sorted(picks), keep=12)
    W_, H_ = fr.W, fr.H
    for t in sorted(picks):
        im = fr.get(t)
        if im is None:
            print("no frame at", t); continue
        for epid, k in picks[t]:
            ep = eps[epid]; ev_ = min(ep["events"], key=lambda e: abs(e["t"] - t)); b = ev_["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; sz = max(5 * max(b[2] - b[0], b[3] - b[1]), 160)
            x0, y0 = int(np.clip(cx - sz / 2, 0, max(W_ - sz, 0))), int(np.clip(cy - sz / 2, 0, max(H_ - sz, 0)))
            c = im[y0:y0 + int(sz), x0:x0 + int(sz)].copy(); z = 320 / max(c.shape[:2])
            c = cv2.resize(c, (int(c.shape[1] * z), int(c.shape[0] * z)), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_CUBIC)
            cv2.rectangle(c, (int((b[0] - x0) * z) - 2, int((b[1] - y0) * z) - 2), (int((b[2] - x0) * z) + 2, int((b[3] - y0) * z) + 2), (255, 0, 255), 1)
            cv2.imwrite(os.path.join(a.W, "crops", f"ep{epid}_{k}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            ep.setdefault("crop_geo", {})[str(k)] = [x0, y0, z]
            if k == 1:
                th = cv2.resize(im, (1280, int(1280 * H_ / W_)), interpolation=cv2.INTER_AREA)
                cv2.circle(th, (int(cx * 1280 / W_), int(cy * 1280 / W_)), 22, (255, 0, 255), 2)
                for _, g in lab[(abs(lab.t - t) < 1e-6) & lab.left.notna()].iterrows():
                    cv2.rectangle(th, (int(g.left * 1280 / W_) - 3, int(g.top * 1280 / W_) - 3), (int((g.left + g.width) * 1280 / W_) + 3, int((g.top + g.height) * 1280 / W_) + 3), (0, 255, 0), 1)
                cv2.imwrite(os.path.join(a.W, "crops", f"ep{epid}_full.jpg"), th, [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump({"gt": a.gt, "video": a.video, "frame_size": [W_, H_], "args": {k: v for k, v in vars(a).items() if k != "model"},
               "models": a.model, "episodes": eps}, open(os.path.join(a.W, "episodes.json"), "w"))
    print(f"{len(ev)} object events -> {len(eps)} episodes ({sum(e['kind'] == 'FP' for e in eps)} on empty frames, "
          f"{sum(e['kind'] == 'FPpos' for e in eps)} on drone frames) -> {a.W}")


def cmd_review(a):
    P = os.path.join(a.W, "episodes.json"); D = json.load(open(P)); eps = D["episodes"]
    FW, FH = D["frame_size"]
    st = {"down": None, "cur": None, "drawn": None}

    def mouse(ev, x, y, flags, p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["down"] = (x, y); st["cur"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["down"]:
            st["cur"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["down"]:
            x0, y0 = st["down"]
            if abs(x - x0) > 2 and abs(y - y0) > 2 and st["geo"]:
                gx, gy, gw, gh, z, ox, oy = st["geo"]
                if gx <= x0 < gx + gw and gy <= y0 < gy + gh and gx <= x < gx + gw and gy <= y < gy + gh:
                    st["drawn"] = [ox + (min(x0, x) - gx) / z, oy + (min(y0, y) - gy) / z, ox + (max(x0, x) - gx) / z, oy + (max(y0, y) - gy) / z]
            st["down"] = None; st["cur"] = None
    st["geo"] = None
    cv2.namedWindow("episodes"); cv2.setMouseCallback("episodes", mouse)
    todo = [i for i, e in enumerate(eps) if e["decision"] == ""]
    hist, j = [], 0
    while 0 <= j < len(todo):
        e = eps[todo[j]]; st["drawn"] = None
        crops = [cv2.imread(os.path.join(a.W, "crops", f"ep{e['id']}_{k}.jpg")) for k in range(3)]
        full = cv2.imread(os.path.join(a.W, "crops", f"ep{e['id']}_full.jpg"))
        if full is None:
            full = np.zeros((720, 1280, 3), np.uint8)
        while True:
            row = np.zeros((330, 3 * 325, 3), np.uint8)
            for k, c in enumerate(crops):
                if c is not None:
                    row[:c.shape[0], k * 325:k * 325 + c.shape[1]] = c
            x0, y0, z = e["crop_geo"]["1"] if "crop_geo" in e and "1" in e["crop_geo"] else (0, 0, 1)
            canvas = np.zeros((52 + 330 + full.shape[0], max(row.shape[1], full.shape[1]), 3), np.uint8)
            canvas[52:52 + 330, :row.shape[1]] = row; canvas[52 + 330:, :full.shape[1]] = full
            st["geo"] = (325, 52, 320, 320, z, x0, y0)            # the middle crop is the drawing surface
            if st["drawn"]:
                b = st["drawn"]; cv2.rectangle(canvas, (int(325 + (b[0] - x0) * z), int(52 + (b[1] - y0) * z)), (int(325 + (b[2] - x0) * z), int(52 + (b[3] - y0) * z)), (0, 255, 0), 1)
            if st["down"] and st["cur"]:
                cv2.rectangle(canvas, st["down"], st["cur"], (0, 255, 0), 1)
            done = sum(1 for x in eps if x["decision"])
            cv2.putText(canvas, f"[{done}/{len(eps)}] ep {e['id']} {e['kind']} t={e['t0']:.1f}-{e['t1']:.1f}s {e['n_frames']} frames, max conf {e['max_conf']:.2f}, models: {','.join(e['models'])[:60]}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            msg = "DRAWN: press u to add this box as our drone (c clears)" if st["drawn"] else "u=our DRONE  p=manned plane  b=bird  f=nothing real  x=unsure  drag in middle crop=draw  space=skip  z=back  q=quit"
            cv2.putText(canvas, msg, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if st["drawn"] else (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("episodes", canvas)
            c = cv2.waitKey(30) & 0xFF
            if c == 255:
                continue
            if st["drawn"]:
                if c == ord("u"):
                    e["manual"] = st["drawn"]; e["decision"] = "u"; hist.append(j); j += 1; break
                if c == ord("c"):
                    st["drawn"] = None
                continue
            if c in (ord("u"), ord("p"), ord("b"), ord("f"), ord("x")):
                e["decision"] = chr(c); hist.append(j); j += 1; break
            if c == ord(" "):
                hist.append(j); j += 1; break
            if c == ord("z") and hist:
                j = hist.pop(); eps[todo[j]]["decision"] = ""; eps[todo[j]]["manual"] = None; break
            if c == ord("q"):
                json.dump(D, open(P, "w")); print("saved"); return
        if len(hist) % 10 == 0:
            json.dump(D, open(P, "w"))
    json.dump(D, open(P, "w")); print("all episodes decided; run apply")


def cmd_apply(a):
    D = json.load(open(os.path.join(a.W, "episodes.json")))
    lab = pd.read_csv(os.path.join(D["gt"], "labels.csv"), dtype={"class": str})
    rows = []
    for ep in D["episodes"]:
        if ep["decision"] not in ("u", "x"):
            continue
        diff = 1 if ep["decision"] == "x" else 0
        if ep.get("manual"):
            b = ep["manual"]; mid = ep["shots"][1]
            rows.append({"camera": "day", "t": mid, "class": "uav", "left": b[0], "top": b[1], "width": b[2] - b[0], "height": b[3] - b[1], "difficult": diff, "track_id": 9000 + ep["id"]})
        else:
            for e in ep["events"]:
                b = e["box"]
                rows.append({"camera": "day", "t": e["t"], "class": "uav", "left": b[0], "top": b[1], "width": b[2] - b[0], "height": b[3] - b[1], "difficult": diff, "track_id": 9000 + ep["id"]})
    new = pd.concat([lab, pd.DataFrame(rows)], ignore_index=True).sort_values(["camera", "t", "track_id"], kind="stable")
    os.makedirs(a.out, exist_ok=True)
    new.to_csv(os.path.join(a.out, "labels.csv"), index=False, float_format="%.6f")
    g = json.load(open(os.path.join(D["gt"], "gt.json")))
    g["name"] = g.get("name", "") + " [v2: real objects found on 'empty' frames added]"
    g["notes"] = g.get("notes", "") + f" | v2 corrections by Daniel's episode review (gt_episode_fix.py): {sum(e['decision'] == 'u' for e in D['episodes'])} episodes added as uav, {sum(e['decision'] == 'x' for e in D['episodes'])} marked difficult."
    json.dump(g, open(os.path.join(a.out, "gt.json"), "w"), indent=1)
    json.dump({"source_gt": D["gt"], "decisions": [{k: v for k, v in e.items() if k != "events"} for e in D["episodes"]]}, open(os.path.join(a.out, "episode_decisions.json"), "w"), indent=1)
    from collections import Counter
    print(f"{len(rows)} rows added from {Counter(e['decision'] for e in D['episodes'])} -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "review", "apply"]); ap.add_argument("--W", required=True)
    ap.add_argument("--gt"); ap.add_argument("--video"); ap.add_argument("--cache")
    ap.add_argument("--model", nargs=2, action="append", metavar=("NAME", "JSON"))
    ap.add_argument("--fp-conf", type=float, default=0.25); ap.add_argument("--merge-s", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.021); ap.add_argument("--out")
    a = ap.parse_args()
    os.makedirs(a.W, exist_ok=True)
    {"prepare": cmd_prepare, "review": cmd_review, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()

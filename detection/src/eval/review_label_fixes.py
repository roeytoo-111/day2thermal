r"""
Decide the label-fix proposals from src/data/propose_rgb_label_fixes.py (<fixes>/queue.csv), one box at a time.
Left: the whole image with its current labels (uav red, airplane yellow, bird cyan; boxes you added this session
green). Right: zoom on the proposal (magenta). Header: what the model thinks and what each key would do.

  u / p / b   the box IS a uav / airplane / bird  (missing -> add it; existing label -> set its class)
  x           there is NO object here            (missing -> reject; existing label -> delete it)
  k           keep as it is (no change)
  A           accept this and ALL remaining "missing" boxes of this image with the model's class (flocks)
  o           the whole image is out of scope (drone on a stand / in hand / not a flight scene / broken) -> dropped
  drag        redraw the box on either panel, then u / p / b
  space       skip (asked again next run)      z  back      q  save and quit (resumable)
No Enter key anywhere.

    python3 src/eval/review_label_fixes.py --fixes data/rgb_fixes/v3                  # review
    python3 src/eval/review_label_fixes.py --fixes data/rgb_fixes/v3 --kind missing   # one kind only
    python3 src/eval/review_label_fixes.py --fixes data/rgb_fixes/v3 --finalize       # write corrected labels
--finalize writes <fixes>/labels/<stem>.txt for every changed image (always uav,airplane,bird, recorded in
labels/_names.txt, whatever the dataset's order; unchanged images are not written) and <fixes>/excluded.txt; resplit_by_source.py takes them as --labels-override / --exclude.
Several queues can be finalized into one folder with --labels-out.
"""
import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import rgb_ds as R   # noqa: E402

COL = {"uav": (0, 0, 255), "airplane": (0, 220, 255), "bird": (255, 220, 0)}
KEYS = {ord("u"): "uav", ord("p"): "airplane", ord("b"): "bird", ord("x"): "x", ord("k"): "k", ord("o"): "o"}
LEFT_H, ZOOM = 760, 520
CANON = ["uav", "airplane", "bird"]   # corrected labels are always written in this order (+ _names.txt), so fixes made
                                      # on v3 files (bird, airplane, uav) apply unchanged to the same files in v4


def finalize(a, Q, root, names):
    out = a.labels_out or os.path.join(a.fixes, "labels")
    os.makedirs(out, exist_ok=True)
    excl_path = os.path.join(os.path.dirname(out.rstrip("/")) if a.labels_out else a.fixes, "excluded.txt")
    excluded = set(Q[Q.decision == "o"].image.map(lambda p: os.path.splitext(os.path.basename(p))[0]))
    n = {"added": 0, "relabelled": 0, "deleted": 0, "images": 0}
    for img, d in Q[Q.decision.isin(["uav", "airplane", "bird", "x"])].groupby("image"):
        stem = os.path.splitext(os.path.basename(img))[0]
        if stem in excluded:
            continue
        L = [[CANON.index(names[c]), *b] for c, *b in R.read_labels(R.label_path(img)) if 0 <= c < len(names) and names[c] in CANON]
        W, H = d.W.iloc[0], d.H.iloc[0]
        drop, add = set(), []
        for r in d.itertuples():
            box = [r.nx0, r.ny0, r.nx1, r.ny1] if not np.isnan(r.nx0) else [r.x0, r.y0, r.x1, r.y1]
            yb = [(box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H, (box[2] - box[0]) / W, (box[3] - box[1]) / H]
            if r.label_idx < 0:
                if r.decision != "x":
                    add.append([CANON.index(r.decision), *yb]); n["added"] += 1
            elif r.decision == "x":
                drop.add(int(r.label_idx)); n["deleted"] += 1
            else:
                L[int(r.label_idx)] = [CANON.index(r.decision), *(yb if not np.isnan(r.nx0) else L[int(r.label_idx)][1:])]
                n["relabelled"] += 1
        L = [x for j, x in enumerate(L) if j not in drop] + add
        open(os.path.join(out, stem + ".txt"), "w").write("".join(f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for c, *b in L))
        n["images"] += 1
    open(os.path.join(out, "_names.txt"), "w").write(",".join(CANON) + "\n")
    prev = set(x.strip() for x in open(excl_path)) if os.path.exists(excl_path) else set()
    open(excl_path, "w").write("".join(s + "\n" for s in sorted(prev | excluded)))
    print(f"labels -> {out}: {n}; excluded images: {len(excluded)} (file {excl_path})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixes", required=True); ap.add_argument("--kind", default=None, help="missing|wrong_class|unsupported")
    ap.add_argument("--split", default=None); ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--labels-out", default=None)
    a = ap.parse_args()
    qp = os.path.join(a.fixes, "queue.csv")
    Q = pd.read_csv(qp, keep_default_na=False, na_values={"nx0": [""], "ny0": [""], "nx1": [""], "ny1": [""]})
    for c in ("nx0", "ny0", "nx1", "ny1"):
        Q[c] = pd.to_numeric(Q[c], errors="coerce")
    Q["decision"] = Q["decision"].astype(str)
    root, names = [x.strip() for x in open(os.path.join(a.fixes, "dataset.txt"))][:2]
    names = names.split(",")
    if a.finalize:
        return finalize(a, Q, root, names)
    sel = (Q.decision == "")
    if a.kind:
        sel &= Q.kind == a.kind
    if a.split:
        sel &= Q.split == a.split
    todo = list(Q.index[sel]); hist = []
    st = {"down": None, "cur": None, "box": None, "geo": None}

    def to_img(x, y):
        g = st["geo"]
        if x < g["lw"]:
            return x / g["s"], y / g["s"]
        return g["cx0"] + (x - g["lw"]) / g["z"], g["cy0"] + y / g["z"]

    def mouse(ev, x, y, flags, p):
        if st["geo"] is None:
            return
        y -= 52   # header band
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["down"] = (x, y); st["cur"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["down"]:
            st["cur"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["down"]:
            (ax, ay), (bx, by) = to_img(*st["down"]), to_img(x, y)
            if abs(bx - ax) > 1 and abs(by - ay) > 1 and (st["down"][0] < st["geo"]["lw"]) == (x < st["geo"]["lw"]):
                st["box"] = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
            st["down"] = None; st["cur"] = None

    win = "label fixes"
    cv2.namedWindow(win); cv2.setMouseCallback(win, mouse)
    cache = {}
    i = 0
    while 0 <= i < len(todo):
        k = todo[i]; r = Q.loc[k]
        if r.image not in cache:
            cache.clear(); cache[r.image] = cv2.imread(r.image)
        im = cache[r.image]; H, W = im.shape[:2]
        L = R.read_labels(R.label_path(r.image))
        st["box"] = None
        s = LEFT_H / H if H * 1.0 / W > 0.7 else 900 / W
        lw = int(W * s)
        b = [r.x0, r.y0, r.x1, r.y1]
        side = max(4 * max(b[2] - b[0], b[3] - b[1]), 64)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        cx0, cy0 = int(np.clip(cx - side / 2, 0, max(0, W - side))), int(np.clip(cy - side / 2, 0, max(0, H - side)))
        crop = im[cy0:cy0 + int(side), cx0:cx0 + int(side)]
        z = ZOOM / max(crop.shape[:2])
        st["geo"] = {"s": s, "lw": lw, "z": z, "cx0": cx0, "cy0": cy0}
        verb = {"missing": "model sees an unlabelled", "wrong_class": f"labelled {r.cls}, model says", "unsupported": f"labelled {r.cls}, model sees nothing ->"}[r.kind]
        what = r.proposed if r.kind != "unsupported" else "doubtful label"
        xdo = "x=reject" if r.kind == "missing" else "x=delete label"
        added = Q[(Q.image == r.image) & (Q.label_idx < 0) & Q.decision.isin(["uav", "airplane", "bird"])]
        while True:
            left = cv2.resize(im, (lw, int(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            for c, ccx, ccy, w, h in L:
                nm = names[c] if 0 <= c < len(names) else "?"
                cv2.rectangle(left, (int((ccx - w / 2) * W * s), int((ccy - h / 2) * H * s)), (int((ccx + w / 2) * W * s), int((ccy + h / 2) * H * s)), COL.get(nm, (200, 200, 200)), 1)
            for q in added.itertuples():
                bb = [q.nx0, q.ny0, q.nx1, q.ny1] if not np.isnan(q.nx0) else [q.x0, q.y0, q.x1, q.y1]
                cv2.rectangle(left, (int(bb[0] * s), int(bb[1] * s)), (int(bb[2] * s), int(bb[3] * s)), (0, 255, 0), 1)
            bb = st["box"] or b
            cv2.rectangle(left, (int(bb[0] * s) - 3, int(bb[1] * s) - 3), (int(bb[2] * s) + 3, int(bb[3] * s) + 3), (255, 0, 255), 2)
            right = cv2.resize(crop, (int(crop.shape[1] * z), int(crop.shape[0] * z)), interpolation=cv2.INTER_NEAREST if z > 2 else cv2.INTER_LINEAR)
            for c, ccx, ccy, w, h in L:
                nm = names[c] if 0 <= c < len(names) else "?"
                p0 = (int(((ccx - w / 2) * W - cx0) * z), int(((ccy - h / 2) * H - cy0) * z)); p1 = (int(((ccx + w / 2) * W - cx0) * z), int(((ccy + h / 2) * H - cy0) * z))
                cv2.rectangle(right, p0, p1, COL.get(nm, (200, 200, 200)), 1)
            cv2.rectangle(right, (int((bb[0] - cx0) * z) - 4, int((bb[1] - cy0) * z) - 4), (int((bb[2] - cx0) * z) + 4, int((bb[3] - cy0) * z) + 4), (255, 0, 255), 2)
            hh = max(left.shape[0], ZOOM) + 52
            canvas = np.zeros((hh, lw + ZOOM, 3), np.uint8)
            canvas[52:52 + left.shape[0], :lw] = left; canvas[52:52 + right.shape[0], lw:lw + right.shape[1]] = right
            if st["down"] and st["cur"]:
                cv2.rectangle(canvas, (st["down"][0], st["down"][1] + 52), (st["cur"][0], st["cur"][1] + 52), (0, 255, 0), 1)
            done = int((Q.decision != "").sum())
            cv2.putText(canvas, f"[{done}/{len(Q)}] {r.split} {os.path.basename(r.image)[:60]}   {r.kind}: {verb} {what} ({r.conf})",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"u=uav p=airplane b=bird  {xdo}  k=keep  A=accept all missing here  o=image out of scope  drag=redraw  space=skip z=back q=quit",
                        (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            c = cv2.waitKey(30) & 0xFF
            if c in KEYS:
                changed = [k]
                Q.at[k, "decision"] = KEYS[c]
                if st["box"] and KEYS[c] in ("uav", "airplane", "bird"):
                    Q.loc[k, ["nx0", "ny0", "nx1", "ny1"]] = [round(float(v), 1) for v in st["box"]]
                if KEYS[c] == "o":   # whole image: close all its open proposals
                    for j in todo[i + 1:]:
                        if Q.at[j, "image"] == r.image and Q.at[j, "decision"] == "":
                            Q.at[j, "decision"] = "o"; changed.append(j)
                hist.append((i, changed)); i += 1; break
            if c == ord("A"):
                changed = []
                for j in [k] + todo[i + 1:]:
                    if Q.at[j, "image"] == r.image and Q.at[j, "kind"] == "missing" and Q.at[j, "decision"] == "":
                        Q.at[j, "decision"] = Q.at[j, "proposed"]; changed.append(j)
                hist.append((i, changed)); i += 1; break
            if c == ord(" "):
                hist.append((i, [])); i += 1; break
            if c == ord("z") and hist:
                i, changed = hist.pop()
                for j in changed:
                    Q.at[j, "decision"] = ""; Q.loc[j, ["nx0", "ny0", "nx1", "ny1"]] = np.nan
                break
            if c == ord("q"):
                Q.to_csv(qp, index=False); print("saved", qp); return
        while i < len(todo) and Q.at[todo[i], "decision"] != "":
            i += 1          # skip rows closed by A / o
        if len(hist) % 20 == 0:
            Q.to_csv(qp, index=False)
    Q.to_csv(qp, index=False); print("queue done; run again with --finalize")


if __name__ == "__main__":
    main()

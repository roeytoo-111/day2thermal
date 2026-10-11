r"""
Reviewed boxes -> a pipeline_eval ground-truth set (interceptor repo, tools/pipeline_eval/GROUND_TRUTH.md):
<out>/gt.json + <out>/labels.csv (camera,t,class,left,top,width,height,difficult,track_id), boxes in video pixels.

Inputs
  thermal: gt_boxes.py boxes.csv (frame_id = OpenCV sequential index of the .ts, verified identical to the ffmpeg
           frame at the manifest's t) + the manifest (frame_id -> t).
  day:     review_day_gt.py boxes.csv (t, verdict, cls, box, difficult, source).
Only HUMAN decisions are exported (pre-filled rows that nobody reviewed are skipped) unless --include-prefill
(for a preview of the format only; never hand that to the eval as ground truth).
Verdicts: drone -> class uav; bird_or_other -> bird (thermal tool) or the day tool's class (bird / airplane);
unsure -> the box with difficult = 1 if there is a box, else the frame is left unlabelled; nothing -> one row with
an empty class (labelled empty frame). Track ids: uav boxes linked over time per camera (gap <= --track-gap s,
centre jump <= max(--track-px, 3 x box size)).

    python3 src/eval/export_pipeline_gt.py --out ../data/arsuf_1005/pipeline_gt/arsuf_1005 \
        --thermal-boxes ../data/arsuf_1005/gt_thermal/boxes.csv --thermal-manifest ../data/arsuf_1005/gt_thermal/manifest.csv \
        --day-boxes ../data/arsuf_1005/gt_day/boxes.csv
"""
import os
import csv
import json
import argparse

import numpy as np
import pandas as pd

HUMAN_THERMAL_SOURCES_EXCLUDED = ("prefill_det", "prefill_interp")


def rows_thermal(boxes, manifest, include_prefill):
    t_of = {int(r.frame_id): float(r.t) for r in pd.read_csv(manifest).itertuples()}
    out = []
    for r in csv.DictReader(open(boxes)):
        reviewed = r.get("verified") == "v2" or r["source"] not in HUMAN_THERMAL_SOURCES_EXCLUDED
        if not reviewed and not include_prefill:           # gt_boxes --verify keeps the source and sets verified=v2
            continue
        t = t_of[int(r["frame_id"])]
        out.append(("thermal", t, r["verdict"], "bird" if r["verdict"] == "bird_or_other" else "uav",
                    r["x0"], r["y0"], r["x1"], r["y1"]))
    return out


def rows_day(boxes, include_prefill):
    out = []
    for r in csv.DictReader(open(boxes)):
        if r["source"] != "human" and not include_prefill:
            continue
        out.append(("day", float(r["t"]), r["verdict"], r["cls"] or "uav", r["x0"], r["y0"], r["x1"], r["y1"]))
    return out


def to_labels(rows):
    lab = []
    for cam, t, verdict, cls, x0, y0, x1, y1 in rows:
        has_box = x0 not in ("", None) and str(x0) != "nan"
        if verdict == "nothing":
            lab.append([cam, t, "", "", "", "", "", "", ""])
        elif verdict in ("drone", "bird_or_other", "unsure") and has_box:
            x0, y0, x1, y1 = (float(v) for v in (x0, y0, x1, y1))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            lab.append([cam, t, cls, round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1),
                        1 if verdict == "unsure" else 0, ""])
        # unsure without a box / unknown verdict: frame stays unlabelled
    return lab


def add_tracks(lab, gap, px):
    nxt = 1
    for cam in {r[0] for r in lab}:
        idx = sorted([i for i, r in enumerate(lab) if r[0] == cam and r[2] == "uav"], key=lambda i: lab[i][1])
        last = None
        for i in idx:
            t, l, tp, w, h = lab[i][1], lab[i][3], lab[i][4], lab[i][5], lab[i][6]
            c = (l + w / 2, tp + h / 2)
            if last and t - last[0] <= gap and np.hypot(c[0] - last[1][0], c[1] - last[1][1]) <= max(px, 3 * max(w, h)):
                tid = last[2]
            else:
                tid, nxt = nxt, nxt + 1
            lab[i][8] = tid
            last = (t, c, tid)
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thermal-boxes"); ap.add_argument("--thermal-manifest"); ap.add_argument("--day-boxes")
    ap.add_argument("--day-video", default="day_output.ts"); ap.add_argument("--thermal-video", default="th_output.ts")
    ap.add_argument("--name", default="arsuf 2026-10-05 bench")
    ap.add_argument("--include-prefill", action="store_true")
    ap.add_argument("--track-gap", type=float, default=2.0); ap.add_argument("--track-px", type=float, default=40)
    ap.add_argument("--videos-dir", default=None, help="snap every t to the exact ffprobe pts of the video here "
                    "(ffmpeg showinfo prints 6 significant digits: 1 ms rounding above 100 s)")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/pipeline_eval"))
    a = ap.parse_args()
    rows, videos = [], {}
    if a.thermal_boxes:
        rows += rows_thermal(a.thermal_boxes, a.thermal_manifest, a.include_prefill); videos["thermal"] = a.thermal_video
    if a.day_boxes:
        rows += rows_day(a.day_boxes, a.include_prefill); videos["day"] = a.day_video
    if a.videos_dir:
        import sys
        sys.path.insert(0, "/shared/interceptor/tools/pipeline_eval")
        from frames import VideoIndex
        idx = {cam: VideoIndex(os.path.join(a.videos_dir, v), a.cache_dir) for cam, v in videos.items()}
        snapped = []
        for r in rows:
            t = idx[r[0]].nearest(r[1], 0.005)
            if t is None:
                raise SystemExit(f"{r[0]} t={r[1]} is not a frame of {videos[r[0]]} (within 5 ms)")
            snapped.append((r[0], t) + tuple(r[2:]))
        rows = snapped
    lab = add_tracks(to_labels(rows), a.track_gap, a.track_px)
    lab.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(a.out, exist_ok=True)
    json.dump({"name": a.name + (" (PREVIEW: includes unreviewed pre-fill)" if a.include_prefill else ""),
               "videos": videos, "targets": ["uav"],
               "notes": "Pre-filled from M0 (thermal) / production RGB YOLO (day) detection tracks plus model-free and "
                        "cross-camera candidates (sync thermal_t = day_t - 0.094 s, affine from co-detections), then "
                        "reviewed by a human; only reviewed frames are included. day2thermal "
                        "detection/src/eval/export_pipeline_gt.py"},
              open(os.path.join(a.out, "gt.json"), "w"), indent=1)
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["camera", "t", "class", "left", "top", "width", "height", "difficult", "track_id"])
        for r in lab:
            w.writerow([r[0], f"{r[1]:.6f}"] + r[2:])
    df = pd.DataFrame(lab, columns=["camera", "t", "cls", "l", "tp", "w", "h", "d", "tid"])
    for cam, g in df.groupby("camera"):
        o = g[g.cls != ""]
        print(f"{cam}: {g.t.nunique()} labelled frames, {(o.cls == 'uav').sum()} uav boxes "
              f"({(o.d == 1).sum()} difficult), {(g.cls == '').sum()} empty frames, other classes "
              f"{sorted(set(o.cls) - {'uav'})}, {o.tid.replace('', np.nan).nunique()} tracks")


if __name__ == "__main__":
    main()

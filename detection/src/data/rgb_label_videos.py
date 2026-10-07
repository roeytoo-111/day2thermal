r"""
Tiled RGB detection on 4K day videos with the production RGB YOLO (SAHI-style: 640 px tiles, overlap 0.2 --
the same tiling the Jetson pipeline uses, stage 1 only; the stage-2 CNN is a TensorRT engine and not used here).

Output per video: <out>/<name>.json in the run_pipeline_tiled / rgb_to_thermal_labels format:
    [{"frame_id": k+1, "src_frame": <0-based decode index>, "detections": [{"bbox": [x0,y0,x1,y1] (day px),
      "conf": c, "cls": name}]}, ...]   for every --every-th decoded frame (k = 0,1,2.. over processed frames).
So `rgb_to_thermal_labels.py --rgb-frame-stride N` works (frame_id-1)*N == src_frame.

Frames are decoded SEQUENTIALLY (no seeking), `grab()` skips the ones not processed. Logged at a low conf floor;
filter downstream. Class names come from the checkpoint ({0:uav, 1:airplane, 2:bird} for the 2026-09-21 model).

    python3 src/data/rgb_label_videos.py --model yolo26s_20260921_145030_rgbcurrent.pt --every 3 \
        --video 0858:/path/day.mp4 --video ... --out data/boson_work/rgbdets
"""
import os
import json
import time
import argparse

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--video", action="append", required=True, help="name:path (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--max-frames", type=int, default=None, help="stop after this many source frames (testing)")
    p.add_argument("--batch-frames", type=int, default=2, help="frames per predict call (x ~32 tiles each at 4K)")
    return p.parse_args()


def tile_origins(W, H, tile, overlap):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


def nms(dets, iou_thr):
    """Greedy per-class NMS on [x0,y0,x1,y1,conf,cls] rows (tile duplicates of the same object)."""
    out = []
    for c in {d[5] for d in dets}:
        ds = sorted([d for d in dets if d[5] == c], key=lambda d: -d[4])
        keep = []
        for d in ds:
            ok = True
            for k in keep:
                ix = max(0.0, min(d[2], k[2]) - max(d[0], k[0]))
                iy = max(0.0, min(d[3], k[3]) - max(d[1], k[1]))
                inter = ix * iy
                u = (d[2] - d[0]) * (d[3] - d[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
                if u > 0 and inter / u > iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(d)
        out += keep
    return out


def main():
    a = parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    names = model.names
    os.makedirs(a.out, exist_ok=True)
    for spec in a.video:
        name, path = spec.split(":", 1)
        if os.path.exists(os.path.join(a.out, name + ".json")):
            print(f"{name}: exists, skipped")
            continue
        cap = cv2.VideoCapture(path)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        origins = tile_origins(W, H, a.tile, a.overlap)
        results, buf, t0, i = [], [], time.time(), 0

        def flush():
            tiles, meta = [], []
            for src, fr in buf:
                for (x, y) in origins:
                    tiles.append(fr[y:y + a.tile, x:x + a.tile])
                    meta.append((src, x, y))
            res = model.predict(tiles, conf=a.conf, imgsz=a.tile, device=a.device, verbose=False, half=True)
            per = {src: [] for src, _ in buf}
            for (src, x, y), r in zip(meta, res):
                for xyxy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    per[src].append([xyxy[0] + x, xyxy[1] + y, xyxy[2] + x, xyxy[3] + y, c, int(k)])
            for src, _ in buf:
                dets = nms(per[src], a.nms_iou)
                results.append({"frame_id": len(results) + 1, "src_frame": src,
                                "detections": [{"bbox": [round(d[0], 1), round(d[1], 1), round(d[2], 1), round(d[3], 1)],
                                                "conf": round(d[4], 4), "cls": names[d[5]]} for d in dets]})
            buf.clear()

        while True:
            if i % a.every == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                buf.append((i, fr))
                if len(buf) >= a.batch_frames:
                    flush()
            else:
                if not cap.grab():
                    break
            i += 1
            if a.max_frames and i >= a.max_frames:
                break
            if i % 600 == 0:
                print(f"  {name}: {i} frames ({i / (time.time() - t0):.1f} fps decode+detect)", flush=True)
        if buf:
            flush()
        json.dump(results, open(os.path.join(a.out, name + ".json"), "w"))
        n_det = sum(1 for r in results if any(d["conf"] >= 0.25 for d in r["detections"]))
        print(f"{name}: {i} frames read, {len(results)} processed (every {a.every}), "
              f"{n_det} with a det >= 0.25, {time.time() - t0:.0f}s, tiles/frame {len(origins)}", flush=True)


if __name__ == "__main__":
    main()

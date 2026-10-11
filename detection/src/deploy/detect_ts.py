r"""
Run a YOLO model over a recorded .ts video, keyed by each frame's TIMESTAMP (pts_time on the file's own timeline),
the frame identity used by the interceptor repo's pipeline_eval ground truth (tools/pipeline_eval/GROUND_TRUTH.md).
Frame counts are not used: these recordings start mid-stream and decoders skip different numbers of broken frames.

Decoding is one sequential ffmpeg pass with -copyts and showinfo, as export_frames.py does, so a `t` written here is
the same `t` that export_frames.py puts in its file names. --tile > 0 runs SAHI-style tiles (4K day video) and merges
tile duplicates with per-class NMS (same helpers as rgb_label_videos.py); --tile 0 runs the full frame (thermal).

Output JSON: {"video", "model", "names", "width", "height", "every", "frames": [{"t": 59.264111, "dets":
[[x0, y0, x1, y1, conf, cls], ...]}, ...]} in video pixels.

    python3 src/deploy/detect_ts.py --video ../data/arsuf_1005/th_output.ts --model M0.pt --out x.json --tile 0
    python3 src/deploy/detect_ts.py --video ../data/arsuf_1005/day_output.ts --model rgb.pt --out y.json --every 3
"""
import os
import re
import sys
import json
import time
import argparse
import threading
import subprocess
import queue

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from rgb_label_videos import tile_origins, nms   # noqa: E402

INFO = re.compile(r"n:\s*(\d+).*?pts_time:([0-9.]+).*?s:(\d+)x(\d+)")


def frames(video, every):
    """Yield (t, HxWx3 BGR uint8) for every `every`-th decoded frame."""
    vf = f"select=not(mod(n\\,{every})),showinfo" if every > 1 else "showinfo"
    p = subprocess.Popen(["ffmpeg", "-v", "info", "-copyts", "-i", video, "-vf", vf, "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)
    q = queue.Queue()

    def read_err():
        for line in iter(p.stderr.readline, b""):
            m = INFO.search(line.decode("utf-8", "replace"))
            if m and "Parsed_showinfo" in line.decode("utf-8", "replace"):
                q.put((float(m.group(2)), int(m.group(3)), int(m.group(4))))
        q.put(None)
    threading.Thread(target=read_err, daemon=True).start()
    while True:
        info = q.get()
        if info is None:
            break
        t, w, h = info
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield t, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="keep one decoded frame in N")
    ap.add_argument("--tile", type=int, default=640, help="tile size; 0 = whole frame")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=None, help="default: tile size, or 512 for whole frames")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    imgsz = a.imgsz or (a.tile if a.tile else 512)
    out, W, H, t0, n = [], None, None, time.time(), 0
    for t, im in frames(a.video, a.every):
        H, W = im.shape[:2]
        if a.tile:
            org = tile_origins(W, H, a.tile, a.overlap)
            crops = [im[y:y + a.tile, x:x + a.tile] for x, y in org]
            dets = []
            for k in range(0, len(crops), a.batch):
                for (x, y), r in zip(org[k:k + a.batch],
                                     model.predict(crops[k:k + a.batch], imgsz=imgsz, conf=a.conf, verbose=False)):
                    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                        dets.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y, s, int(c)])
            dets = nms(dets, a.nms_iou)
        else:
            r = model.predict(im, imgsz=imgsz, conf=a.conf, verbose=False)[0]
            dets = [[*b, s, int(c)] for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        out.append({"t": round(t, 6), "dets": [[round(v, 1) for v in d[:4]] + [round(d[4], 4), d[5]] for d in dets]})
        n += 1
        if n % 500 == 0:
            print(f"{n} frames, t={t:.2f}, {n / (time.time() - t0):.1f} fps", flush=True)
        if a.max_frames and n >= a.max_frames:
            break
    json.dump({"video": os.path.basename(a.video), "model": os.path.basename(a.model), "names": model.names,
               "width": W, "height": H, "every": a.every, "conf": a.conf, "frames": out}, open(a.out, "w"))
    print(f"{n} frames ({W}x{H}) in {time.time() - t0:.0f}s -> {a.out}; "
          f"{sum(1 for f in out if f['dets'])} with >=1 detection")


if __name__ == "__main__":
    main()

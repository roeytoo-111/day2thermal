r"""
New paste backgrounds from RGB-ONLY footage: crops in the translator's input format, to be translated to
thermal (diffusion infer.py) and used by paste_on_backgrounds.py.

Why: the paste A/B/C/D runs showed (notebook 2026-10-01/02) that pastes help, that reusing the same 773
backgrounds more often does not, and that translated backgrounds are as good as real ones on the sky
(the operational case: the interceptor looks UP at the target). So the lever is NEW scenes, and the
company's RGB videos have plenty that were never recorded in thermal.

Geometry: the ANN-Detection videos come from the same 4K day camera as our registered pairs. Each frame is
warped with ONE verified segment of a registration (day -> undistorted thermal homography + thermal lens),
exactly like `video_pairs extract`, so the crop looks like the translator's training RGB (same scale,
lens, crop). The real day->thermal pose differs a little per flight; for a background that does not matter.

Per sampled frame (every --step-s, after a near-duplicate check):
  * sky_frac of the RGB crop (diagnose.sky_fraction); frames below --min-sky are skipped (pastes are
    sky-only, and the sky is what we want more of),
  * every RGB-detector box of that frame (ANN run_pipeline_tiled JSON, ANY verdict: a drone the cascade
    rejected is still a drone) mapped into the crop -> extra_boxes.json; paste_on_backgrounds.py select
    --extra_boxes inpaints them in the translated image (sky: clean; on texture: the background is dropped).

Output: <out>/pairs/train/rgb/<prefix>_<frame>.png, extra_boxes.json, rgb_backgrounds.csv, rgb_sheet.jpg.
Then (GPU) translate into <out>/pairs/train/thermal/ and run paste_on_backgrounds.py select/build.

    python3 src/data/rgb_backgrounds.py --out data/rgbbg \
        --video a0617a:<ANN>/data/videos/2026-06-17T09_50_39_clip.mp4:<ANN>/.../detection_tiled_2026-06-17T09_50_39_clip_yolo11_finetuned.json ...
"""
import os
import sys
import json
import argparse
import subprocess

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day2thermal.video_pairs import build_rgb_maps, probe          # noqa: E402
from day2thermal.diffusion.diagnose import sky_fraction            # noqa: E402
from rgb_to_thermal_labels import day_box_to_thermal               # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--video", action="append", required=True, help="prefix:video.mp4:rgb_dets.json (repeatable)")
    p.add_argument("--registration", default=os.path.join(ROOT, "data", "vid_pairs_0715", "registration.json"),
                   help="a registration of the SAME 4K day camera (default: 07-15)")
    p.add_argument("--segment", type=int, default=None, help="segment id (default: the longest verified)")
    p.add_argument("--step-s", type=float, default=1.0)
    p.add_argument("--min-sky", type=float, default=0.15)
    p.add_argument("--dedupe", type=float, default=4.0,
                   help="skip a frame whose 64x36 thumbnail differs from the last kept by less (mean abs, 0-255)")
    return p.parse_args()


def sampled_frames(path, size, step):
    """(0-based decode index, downscaled BGR frame) for every step-th frame; ffmpeg skips the rest."""
    w, h = size
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "0", "-i", path, "-vf",
                             f"select=not(mod(n\\,{step})),scale={w}:{h}:flags=area,format=bgr24",
                             "-fps_mode", "passthrough", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    k = 0
    try:
        while True:
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                return
            yield k * step, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        proc.kill()
        proc.wait()


def main():
    a = parse_args()
    reg = json.load(open(a.registration))
    segs = [g for g in reg["segments"] if g["verified"]]
    seg = (max(segs, key=lambda g: g["span"][1] - g["span"][0]) if a.segment is None
           else next(g for g in reg["segments"] if g["id"] == a.segment))
    H = np.array(seg["H_day_to_undistorted_thermal"])
    lam, th_size = reg["lens"]["lambda"], tuple(reg["thermal_size"])
    day_size, ds = tuple(reg["rgb_size"]), reg["day_decode_downscale"]
    x, y, w, h = reg["crop"]
    mx, my, valid = build_rgb_maps(H, lam, th_size, day_size, ds)
    if not valid[y:y + h, x:x + w].all():
        raise SystemExit("the crop is not fully covered by this segment's day frame; pick another --segment")
    rgb_dir = os.path.join(a.out, "pairs", "train", "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    extra, rows, sheet = {}, [], []
    for spec in a.video:
        prefix, path, dets_path = spec.split(":", 2)
        meta = probe(path)
        if (meta["w"], meta["h"]) != day_size:
            raise SystemExit(f"{path}: {meta['w']}x{meta['h']}, registration is for {day_size}; not the same camera")
        dets = {e["frame_id"] - 1: e["detections"] for e in json.load(open(dets_path))}   # JSON is 1-based
        step = max(1, int(round(a.step_s * meta["fps"])))
        last, n_seen, n_sky, n_dup = None, 0, 0, 0
        for fi, fr in sampled_frames(path, (day_size[0] // ds, day_size[1] // ds), step):
            n_seen += 1
            crop = cv2.remap(fr, mx, my, cv2.INTER_LINEAR)[y:y + h, x:x + w]
            sf = sky_fraction(crop)
            if sf < a.min_sky:
                n_sky += 1
                continue
            thumb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
            if last is not None and float(np.abs(thumb.astype(np.float32) - last).mean()) < a.dedupe:
                n_dup += 1
                continue
            last = thumb.astype(np.float32)
            name = f"{prefix}_{fi:06d}.png"
            cv2.imwrite(os.path.join(rgb_dir, name), crop)
            boxes = []
            for d in dets.get(fi, []):
                tb = day_box_to_thermal(d["bbox"], H, lam, th_size)
                bx = [tb[0] - x, tb[1] - y, tb[2] - x, tb[3] - y]
                if bx[2] > 0 and bx[3] > 0 and bx[0] < w and bx[1] < h:
                    boxes.append([round(max(bx[0], 0), 1), round(max(bx[1], 0), 1),
                                  round(min(bx[2], w), 1), round(min(bx[3], h), 1), 1.0])
            extra[name] = boxes
            rows.append({"name": name, "video": os.path.basename(path), "frame_id": fi + 1,
                         "sky_frac": round(sf, 3), "n_rgb_boxes": len(boxes)})
            if len(sheet) < 24 and len(rows) % 15 == 1:
                sheet.append(crop.copy())
        kept = sum(r["video"] == os.path.basename(path) for r in rows)
        print(f"{prefix}: {n_seen} sampled (every {step} fr) -> {kept} kept "
              f"({n_sky} < {a.min_sky} sky, {n_dup} near-duplicates)", flush=True)
    json.dump(extra, open(os.path.join(a.out, "extra_boxes.json"), "w"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rgb_backgrounds.csv"), index=False)
    json.dump({**vars(a), "segment": seg["id"], "n": len(rows)}, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(os.path.join(a.out, "rgb_sheet.jpg"),
                    np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]))
    print(f"-> {len(rows)} RGB crops in {rgb_dir} ({sum(len(v) > 0 for v in extra.values())} with RGB boxes to inpaint)")


if __name__ == "__main__":
    main()

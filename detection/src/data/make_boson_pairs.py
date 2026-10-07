r"""
Registered RGB/thermal pairs for the generator from one Boson session: video_pairs `extract`, then drop frames with a
green overlay mark or outside the usable window, then cut each pair into two 2:1 crops (top / bottom, overlapping) so
the 512x256 generator input is never squashed (the thermal frame is 640x512, aspect 1.25).

Output <out>/<session>/train/{rgb,thermal}/<prefix>_<NNNNNNN>.png with NNNNNNN = thermal_frame*2 + (0 top | 1 bottom)
(prep.py orders by the trailing integer). All pairs are written to train/ (prep.py makes val/test per day).

    python3 src/data/make_boson_pairs.py --session 10_14_01 --raw data/boson_work/pairs_raw --out data/boson_work/pairs_crops \
        --mp4-dir data/boson_work/mp4 --reg data/boson_work/reg --overlay data/boson_work/overlay_flags.json
"""
import os, sys, json, glob, argparse, subprocess
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WINDOWS = {"08_58_15": (286.0, None), "09_09_29": (0.0, 30.0), "10_15_59": (0.0, 90.0)}     # thermal seconds (Daniel)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mp4-dir", required=True)
ap.add_argument("--reg", required=True)
ap.add_argument("--overlay", required=True)
ap.add_argument("--step-s", type=float, default=0.1)
ap.add_argument("--min-gap-s", type=float, default=0.5)
ap.add_argument("--max-gap-s", type=float, default=2.0)
ap.add_argument("--scene-thresh", type=float, default=0.10)
a = ap.parse_args()
s = a.session
raw = os.path.join(a.raw, s)
if not os.path.isdir(os.path.join(raw, "train", "rgb")):
    subprocess.run([sys.executable, "-m", "day2thermal.video_pairs", "extract",
                    "--day", os.path.join(a.mp4_dir, f"{s}_day.mp4"), "--thermal", os.path.join(a.mp4_dir, f"{s}_thermal.mp4"),
                    "--registration", os.path.join(a.reg, s, "registration.json"), "--out", raw, "--prefix", f"b{s}",
                    "--step-s", str(a.step_s), "--min-gap-s", str(a.min_gap_s), "--max-gap-s", str(a.max_gap_s),
                    "--scene-thresh", str(a.scene_thresh), "--train-max-frame", str(10 ** 9), "--val-min-frame", str(10 ** 9 + 1),
                    "--force"], check=True, cwd=ROOT)
reg = json.load(open(os.path.join(a.reg, s, "registration.json")))
fps = reg["fps_thermal"]
flag = set(json.load(open(a.overlay)).get(s, {}).get("flag", []))
lo, hi = WINDOWS.get(s, (None, None))
n_in = n_flag = n_win = n_out = 0
for sub in ("rgb", "thermal"):
    os.makedirs(os.path.join(a.out, s, "train", sub), exist_ok=True)
for p in sorted(glob.glob(os.path.join(raw, "*", "rgb", "*.png"))):
    name = os.path.basename(p)
    ti = int(name.rsplit("_", 1)[1][:-4])
    n_in += 1
    if ti in flag:
        n_flag += 1
        continue
    if (lo is not None and ti / fps < lo) or (hi is not None and ti / fps > hi):
        n_win += 1
        continue
    rgb, th = cv2.imread(p), cv2.imread(p.replace(os.sep + "rgb" + os.sep, os.sep + "thermal" + os.sep))
    h, w = th.shape[:2]
    ch = w // 2                                    # 2:1 crop height
    for k, y0 in enumerate((0, h - ch)):
        out = f"b{s}_{ti * 2 + k:07d}.png"
        cv2.imwrite(os.path.join(a.out, s, "train", "rgb", out), rgb[y0:y0 + ch])
        cv2.imwrite(os.path.join(a.out, s, "train", "thermal", out), th[y0:y0 + ch])
        n_out += 1
print(f"{s}: {n_in} raw pairs -> {n_flag} dropped (overlay), {n_win} outside window, {n_out} crops written")

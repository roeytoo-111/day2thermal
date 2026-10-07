r"""
Registration for a session that cannot calibrate itself (featureless sky, very short clip), built from the rigid rig
geometry measured on well-textured sessions of the same day. 2026-10-06 Boson rig: 26 segments over 6 sessions agree
on scale 0.3005 +- 0.001 thermal px per day px, rotation 0.15 +- 0.1 deg, translation (-253, -41) +- 2 px.

Output has the same format as `video_pairs calibrate` (one segment spanning the whole thermal video, marked verified,
verdict SHARED_H) so `extract`, rgb_to_thermal_labels and build_boson_labels work unchanged. The clock offset
comes from the session's own sync report when it passed, else --offset-ms (median of the passing sessions).
Each extracted pair is still alignment-checked by `extract` (edge NCC vs --min-align, judged only when both images
have structure), and the QA overlays in calib_preview are for human review.

    python3 src/data/make_shared_registration.py --session 09_06_16 --reg data/boson_work/reg --sync data/boson_work/sync \
        --mp4-dir data/boson_work/mp4 --template 10_15_59 --good 10_15_59,10_14_01,10_08_31,10_11_46,10_13_12,10_04_01
"""
import os, json, argparse, subprocess
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True)
ap.add_argument("--reg", required=True)
ap.add_argument("--sync", required=True)
ap.add_argument("--mp4-dir", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--good", required=True, help="comma-separated sessions whose verified segments define the shared H")
ap.add_argument("--offset-ms", type=float, default=-134.0)
ap.add_argument("--drift-ms-per-s", type=float, default=0.0)
ap.add_argument("--min-align", type=float, default=0.30)
a = ap.parse_args()

Hs = []
for g in a.good.split(","):
    for seg in json.load(open(os.path.join(a.reg, g, "registration.json")))["segments"]:
        if seg["verified"]:
            H = np.array(seg["H_day_to_undistorted_thermal"]); Hs.append(H / H[2, 2])
H = np.median(np.stack(Hs), axis=0)
r = json.load(open(os.path.join(a.reg, a.template, "registration.json")))

def probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=nb_frames,duration,r_frame_rate", "-of", "json", path], capture_output=True, text=True).stdout)
    s = j["streams"][0]
    return int(s["nb_frames"]), float(s["duration"])

day = os.path.join(a.mp4_dir, f"{a.session}_day.mp4"); th = os.path.join(a.mp4_dir, f"{a.session}_thermal.mp4")
nd, dd = probe(day); nt, dt = probe(th)
off, drift, src = a.offset_ms, a.drift_ms_per_s, "explicit/median offset"
rep = os.path.join(a.sync, a.session, "sync_report.json")
logf = os.path.join(a.sync, a.session + ".log")
if os.path.exists(rep) and os.path.exists(logf) and "VERDICT: PASS" in open(logf).read():
    lm = json.load(open(rep)).get("lag_model")
    if lm and lm["type"] in ("constant", "linear"):
        off, drift, src = lm["intercept_ms"], lm["slope_ms_per_s"], f"sync report ({lm['type']})"
r.update({"day": day, "thermal": th, "fps_day": nd / dd, "fps_thermal": nt / dt, "offset_ms": off, "drift_ms_per_s": drift,
          "verdict": "SHARED_H", "align_ncc_p05": a.min_align,
          "segments": [{"id": 0, "members": [], "n": 0, "dev_px_median": None, "dev_px_max": None, "first": 0, "last": nt - 1,
                        "verified": True, "span": [0, nt - 1], "H_day_to_undistorted_thermal": H.tolist()}],
          "shared_h_from": a.good, "offset_source": src})
r.pop("frames", None)
out = os.path.join(a.reg, a.session); os.makedirs(out, exist_ok=True)
json.dump(r, open(os.path.join(out, "registration.json"), "w"))
sc = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
print(f"{a.session}: shared H (scale {sc:.4f}, from {len(Hs)} segments), fps thermal {nt / dt:.2f} day {nd / dd:.2f}, "
      f"offset {off:+.0f} ms drift {drift:+.2f} ms/s [{src}]")

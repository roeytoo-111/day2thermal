"""Extract temporally synchronized frame pairs from a day (RGB) recording and
a thermal recording of the same scene.

Both ``--rgb`` and ``--thermal`` accept either a video file or a directory of
frames (for directories, pass the frame rate via ``--rgb-fps/--thermal-fps``).
Timestamps are computed as ``frame_index / fps`` (constant-fps assumption); the
constant clock difference between the two recordings is absorbed by
``--offset-ms``.

Calibrating --offset-ms: create a simultaneous event visible in both spectra
(switch a hot soldering iron / hand warmer into view, or a lighter flame),
find its frame index in each recording, and set
``offset_ms = t_thermal_event_ms - t_rgb_event_ms``.

Example
-------
python -m day2thermal.extract_frames \
    --rgb flight01_day.mp4 --thermal flight01_thermal.mp4 \
    --out data/flight01/raw --fps 5 --offset-ms 120 --drop-static-thermal
"""
import argparse
import os

import cv2
import numpy as np

from .utils import ensure_dir, list_images


class FrameSource:
    """Sequential reader with timestamps for a video file or frame directory."""

    def __init__(self, path, fps_hint=None):
        self.is_dir = os.path.isdir(path)
        self.path = path
        self.idx = 0
        if self.is_dir:
            if not fps_hint:
                raise ValueError(f"{path} is a directory; please pass its fps")
            self.files = list_images(path)
            self.fps = float(fps_hint)
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise IOError(f"cannot open video {path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if fps and fps > 0 else float(fps_hint or 30.0)

    def read(self):
        """Return (t_seconds, frame) or (None, None) at end of stream."""
        if self.is_dir:
            if self.idx >= len(self.files):
                return None, None
            f = os.path.join(self.path, self.files[self.idx])
            frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None, None
        t = self.idx / self.fps
        self.idx += 1
        return t, frame


class NearestFetcher:
    """Streams a FrameSource and returns the frame nearest to a query time.
    Query times must be non-decreasing."""

    def __init__(self, src, time_offset=0.0):
        self.src = src
        self.off = time_offset
        self.prev = None
        self.cur = self._next()

    def _next(self):
        t, f = self.src.read()
        if t is None:
            return None
        return (t + self.off, f)

    @property
    def done(self):
        return self.cur is None

    def fetch(self, t):
        while self.cur is not None and self.cur[0] < t:
            self.prev, self.cur = self.cur, self._next()
        cands = [c for c in (self.prev, self.cur) if c is not None]
        if not cands:
            return None, None
        best = min(cands, key=lambda c: abs(c[0] - t))
        return best[0], best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb", required=True, help="RGB video file or frame dir")
    ap.add_argument("--thermal", required=True, help="thermal video file or frame dir")
    ap.add_argument("--out", required=True, help="output dir (creates rgb/ thermal/)")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="pair sampling rate (pairs per second)")
    ap.add_argument("--offset-ms", type=float, default=0.0,
                    help="thermal clock minus RGB clock for the same event")
    ap.add_argument("--rgb-fps", type=float, default=None)
    ap.add_argument("--thermal-fps", type=float, default=None)
    ap.add_argument("--start", type=float, default=0.0, help="start time [s] on the RGB clock")
    ap.add_argument("--duration", type=float, default=None, help="seconds to process")
    ap.add_argument("--tol-ms", type=float, default=None,
                    help="max |t_frame - t_sample| accepted "
                         "(default: half the thermal frame period)")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--drop-static-thermal", action="store_true",
                    help="skip pairs whose thermal frame is (nearly) identical "
                         "to the previous one - typical of NUC/FFC shutter events")
    args = ap.parse_args()

    rgb_src = FrameSource(args.rgb, args.rgb_fps)
    th_src = FrameSource(args.thermal, args.thermal_fps)
    fr = NearestFetcher(rgb_src, 0.0)
    ft = NearestFetcher(th_src, -args.offset_ms / 1000.0)

    tol = (args.tol_ms / 1000.0) if args.tol_ms else 0.5 / th_src.fps
    out_rgb = ensure_dir(os.path.join(args.out, "rgb"))
    out_th = ensure_dir(os.path.join(args.out, "thermal"))

    t = args.start
    n = 0
    prev_th = None
    dropped_tol = dropped_static = 0
    while True:
        if args.duration is not None and t > args.start + args.duration:
            break
        tr, rgb_img = fr.fetch(t)
        tt, th_img = ft.fetch(t)
        if rgb_img is None or th_img is None:
            break
        if (fr.done and t - tr > tol) or (ft.done and t - tt > tol):
            break
        if abs(tr - t) <= tol and abs(tt - t) <= tol:
            th_gray = th_img if th_img.ndim == 2 else \
                cv2.cvtColor(th_img, cv2.COLOR_BGR2GRAY)
            static = False
            if args.drop_static_thermal and prev_th is not None:
                diff = float(np.mean(np.abs(th_gray.astype(np.float32) - prev_th)))
                static = diff < 0.05
            if static:
                dropped_static += 1
            else:
                name = f"{n:06d}.png"
                cv2.imwrite(os.path.join(out_rgb, name), rgb_img)
                cv2.imwrite(os.path.join(out_th, name), th_gray)
                n += 1
                if args.max_pairs and n >= args.max_pairs:
                    break
            prev_th = th_gray.astype(np.float32)
        else:
            dropped_tol += 1
        t += 1.0 / args.fps

    print(f"wrote {n} pairs to {args.out} "
          f"(dropped: {dropped_tol} out-of-tolerance, {dropped_static} static/NUC)")


if __name__ == "__main__":
    main()

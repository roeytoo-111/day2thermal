"""Interactive correspondence picker for --mode manual registration.

Click matching points on the RGB frame (left) and the thermal frame (right),
alternating. Coordinates are always stored in FULL native resolution for each
camera (3840x2160 for RGB, 640x512 for thermal) regardless of how the panels
are scaled on screen -- picking on a downscaled view and saving the *view*
coordinates is the classic way this file ends up silently wrong.

    python -m day2thermal.pick_points --raw data/raw --frame 2000 --out points.json

Keys
----
left click   place the next point (alternates RGB -> thermal -> RGB ...)
u            undo the last point
c            cycle thermal colormap (grey / jet / inferno)
e            toggle edge-boosted view (helps match structure across modalities)
p            preview: fit H from the current points and blend the warp
s            fit, report reprojection error, save JSON
q / ESC      quit without saving

Pick points spread over the whole overlapping area -- corners and mid-edges,
not a tight cluster. A homography from 4 clustered points extrapolates wildly.
Prefer sharp, unambiguous, *distant* features (road junctions, field corners,
building edges). Avoid clouds: they are at a different depth than the ground
and move between the two exposures.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

CMAPS = [None, cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO]
CMAP_NAMES = ["grey", "jet", "inferno"]
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
          (255, 0, 255), (255, 255, 0), (255, 255, 255), (0, 128, 255)]


def norm8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_boost(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


class Picker:
    def __init__(self, rgb, th, panel_h, zoom_src, zoom_out):
        self.rgb_full = rgb
        self.th_full = th
        self.panel_h = panel_h
        self.zoom_src = zoom_src
        self.zoom_out = zoom_out
        self.cmap = 1
        self.edges = False
        self.pts_rgb = []
        self.pts_th = []
        self.expect_rgb = True
        self.cursor = (0, 0)
        self._build_panels()

    # ---- panels -------------------------------------------------------
    def _build_panels(self):
        rh, rw = self.rgb_full.shape[:2]
        th_h, th_w = self.th_full.shape[:2]
        self.s_rgb = self.panel_h / rh
        self.s_th = self.panel_h / th_h
        rgb_v = self.rgb_full
        th_v = cv2.cvtColor(norm8(self.th_full), cv2.COLOR_GRAY2BGR)
        if self.cmap:
            th_v = cv2.applyColorMap(norm8(self.th_full), CMAPS[self.cmap])
        if self.edges:
            rgb_v = edge_boost(rgb_v)
            th_v = edge_boost(th_v)
        self.p_rgb = cv2.resize(rgb_v, (int(rw * self.s_rgb), self.panel_h),
                                interpolation=cv2.INTER_AREA)
        self.p_th = cv2.resize(th_v, (int(th_w * self.s_th), self.panel_h),
                               interpolation=cv2.INTER_NEAREST)
        self.split = self.p_rgb.shape[1]
        self.base = np.hstack([self.p_rgb, self.p_th])

    # ---- coordinate mapping -------------------------------------------
    def canvas_to_full(self, x, y):
        """-> ('rgb'|'thermal', x_full, y_full) in native camera resolution."""
        if x < self.split:
            return "rgb", x / self.s_rgb, y / self.s_rgb
        return "thermal", (x - self.split) / self.s_th, y / self.s_th

    def full_to_canvas(self, which, x, y):
        if which == "rgb":
            return int(x * self.s_rgb), int(y * self.s_rgb)
        return int(x * self.s_th) + self.split, int(y * self.s_th)

    # ---- interaction ---------------------------------------------------
    def on_mouse(self, event, x, y, flags, _):
        self.cursor = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        which, fx, fy = self.canvas_to_full(x, y)
        if self.expect_rgb and which != "rgb":
            print("  next point goes on the RGB panel (left)")
            return
        if not self.expect_rgb and which != "thermal":
            print("  next point goes on the THERMAL panel (right)")
            return
        (self.pts_rgb if self.expect_rgb else self.pts_th).append([fx, fy])
        print(f"  {which:8s} #{len(self.pts_rgb if self.expect_rgb else self.pts_th)}"
              f"  ({fx:.1f}, {fy:.1f})")
        self.expect_rgb = not self.expect_rgb

    def undo(self):
        if not self.expect_rgb and self.pts_rgb:
            self.pts_rgb.pop()
            self.expect_rgb = True
        elif self.pts_th:
            self.pts_th.pop()
            self.expect_rgb = False
        print(f"  undo -> {len(self.pts_rgb)} rgb / {len(self.pts_th)} thermal")

    # ---- drawing --------------------------------------------------------
    def render(self):
        c = self.base.copy()
        cv2.line(c, (self.split, 0), (self.split, c.shape[0]), (40, 40, 40), 2)
        for i, p in enumerate(self.pts_rgb):
            self._marker(c, self.full_to_canvas("rgb", *p), i)
        for i, p in enumerate(self.pts_th):
            self._marker(c, self.full_to_canvas("thermal", *p), i)
        n = min(len(self.pts_rgb), len(self.pts_th))
        side = "RGB (left)" if self.expect_rgb else "THERMAL (right)"
        bar = (f"pairs:{n}  next:{side}  cmap:{CMAP_NAMES[self.cmap]}"
               f"  edges:{'on' if self.edges else 'off'}   "
               f"[u]ndo [c]map [e]dges [p]review [s]ave [q]uit")
        cv2.rectangle(c, (0, 0), (c.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(c, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return c

    def _marker(self, c, pt, i):
        col = COLORS[i % len(COLORS)]
        cv2.drawMarker(c, pt, col, cv2.MARKER_CROSS, 16, 2)
        cv2.circle(c, pt, 11, col, 1)
        cv2.putText(c, str(i + 1), (pt[0] + 13, pt[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    def zoom(self):
        which, fx, fy = self.canvas_to_full(*self.cursor)
        src = self.rgb_full if which == "rgb" else \
            cv2.applyColorMap(norm8(self.th_full), CMAPS[self.cmap]) if self.cmap else \
            cv2.cvtColor(norm8(self.th_full), cv2.COLOR_GRAY2BGR)
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        h, w = src.shape[:2]
        r = self.zoom_src // 2
        x0, y0 = int(np.clip(fx - r, 0, w - 2 * r)), int(np.clip(fy - r, 0, h - 2 * r))
        patch = src[y0:y0 + 2 * r, x0:x0 + 2 * r]
        if patch.size == 0:
            return np.zeros((self.zoom_out, self.zoom_out, 3), np.uint8)
        z = cv2.resize(patch, (self.zoom_out, self.zoom_out),
                       interpolation=cv2.INTER_NEAREST)
        m = self.zoom_out // 2
        cv2.line(z, (m, 0), (m, self.zoom_out), (0, 255, 255), 1)
        cv2.line(z, (0, m), (self.zoom_out, m), (0, 255, 255), 1)
        cv2.putText(z, f"{which} ({fx:.0f},{fy:.0f})", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return z

    # ---- fitting --------------------------------------------------------
    def fit(self):
        n = min(len(self.pts_rgb), len(self.pts_th))
        if n < 4:
            print(f"need >= 4 pairs, have {n}")
            return None
        src = np.array(self.pts_rgb[:n], np.float32)
        dst = np.array(self.pts_th[:n], np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            print("homography fit FAILED - points are probably collinear "
                  "or badly mismatched")
            return None
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        inl = mask.ravel().astype(bool)
        print(f"  pairs={n} inliers={inl.sum()}  reproj err: "
              f"mean={err.mean():.2f}px max={err.max():.2f}px "
              f"(thermal pixels)")
        for i, e in enumerate(err):
            if not inl[i] or e > 5:
                print(f"    ! pair {i + 1}: err {e:.1f}px"
                      f"{'  OUTLIER' if not inl[i] else ''}")
        return H

    def preview(self, H):
        th_h, th_w = self.th_full.shape[:2]
        w = cv2.warpPerspective(self.rgb_full, H.astype(np.float32), (th_w, th_h),
                                flags=cv2.INTER_AREA)
        thj = cv2.applyColorMap(norm8(self.th_full), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(w, 0.5, thj, 0.5, 0)
        e = cv2.Canny(cv2.GaussianBlur(norm8(self.th_full), (5, 5), 0), 40, 120)
        ed = w.copy()
        ed[e > 0] = (0, 255, 0)
        return np.hstack([w, thj, blend, ed])


def save_points(p, name, rgb, th, out):
    n = min(len(p.pts_rgb), len(p.pts_th))
    data = {"_frame": name,
            "_rgb_size": [rgb.shape[1], rgb.shape[0]],
            "_thermal_size": [th.shape[1], th.shape[0]],
            "rgb": [[float(a), float(b)] for a, b in p.pts_rgb[:n]],
            "thermal": [[float(a), float(b)] for a, b in p.pts_th[:n]]}
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {n} pairs -> {out}")


def run_matplotlib(p, name, out):
    """GUI front-end for headless OpenCV builds. Uses `extent` so mouse
    coordinates arrive already in native full-resolution camera pixels."""
    import matplotlib.pyplot as plt

    rh, rw = p.rgb_full.shape[:2]
    th_h, th_w = p.th_full.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.5))
    fig.canvas.manager.set_window_title(f"pick_points - {name}")

    def rgb_disp():
        img = edge_boost(p.rgb_full) if p.edges else p.rgb_full
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def th_disp():
        g = norm8(p.th_full)
        return cv2.cvtColor(edge_boost(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)),
                            cv2.COLOR_BGR2GRAY) if p.edges else g

    im1 = ax1.imshow(rgb_disp(), extent=[0, rw, rh, 0], interpolation="nearest")
    im2 = ax2.imshow(th_disp(), extent=[0, th_w, th_h, 0],
                     cmap=["gray", "jet", "inferno"][p.cmap], interpolation="nearest")
    ax1.set_title("RGB (full res coords)")
    ax2.set_title("THERMAL")
    for a in (ax1, ax2):
        a.set_anchor("C")
    marks = []

    def redraw():
        for m in marks:
            m.remove()
        marks.clear()
        for ax, pts in ((ax1, p.pts_rgb), (ax2, p.pts_th)):
            for i, (x, y) in enumerate(pts):
                c = f"C{i % 10}"
                marks.append(ax.plot(x, y, "+", color=c, ms=14, mew=2)[0])
                marks.append(ax.annotate(str(i + 1), (x, y), color=c,
                                         xytext=(6, 6), textcoords="offset points",
                                         fontsize=11, weight="bold"))
        n = min(len(p.pts_rgb), len(p.pts_th))
        side = "RGB (left)" if p.expect_rgb else "THERMAL (right)"
        fig.suptitle(f"pairs: {n}   next click: {side}   "
                     f"[u]ndo  [c]map  [e]dges  [p]review  [s]ave  [q]uit"
                     f"    -- use the toolbar magnifier to zoom before clicking",
                     fontsize=11)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes not in (ax1, ax2) or ev.xdata is None:
            return
        if fig.canvas.toolbar and fig.canvas.toolbar.mode:
            return                      # zoom/pan active -- not a pick
        want = ax1 if p.expect_rgb else ax2
        if ev.inaxes is not want:
            print(f"  next point goes on the "
                  f"{'RGB (left)' if p.expect_rgb else 'THERMAL (right)'} panel")
            return
        (p.pts_rgb if p.expect_rgb else p.pts_th).append([ev.xdata, ev.ydata])
        print(f"  {'rgb' if p.expect_rgb else 'thermal':8s}"
              f"#{len(p.pts_rgb if p.expect_rgb else p.pts_th)}"
              f"  ({ev.xdata:.1f}, {ev.ydata:.1f})")
        p.expect_rgb = not p.expect_rgb
        redraw()

    def on_key(ev):
        if ev.key == "u":
            p.undo()
            redraw()
        elif ev.key == "c":
            p.cmap = (p.cmap + 1) % len(CMAPS)
            im2.set_cmap(["gray", "jet", "inferno"][p.cmap])
            fig.canvas.draw_idle()
        elif ev.key == "e":
            p.edges = not p.edges
            im1.set_data(rgb_disp())
            im2.set_data(th_disp())
            fig.canvas.draw_idle()
        elif ev.key in ("p", "s"):
            H = p.fit()
            if H is None:
                return
            if ev.key == "p":
                pv = cv2.cvtColor(p.preview(H), cv2.COLOR_BGR2RGB)
                f2 = plt.figure(figsize=(19, 4))
                plt.imshow(pv)
                plt.axis("off")
                plt.title("warp | thermal | 50/50 blend | thermal edges on warp")
                f2.show()
            else:
                save_points(p, name, p.rgb_full, p.th_full, out)
                plt.close("all")
        elif ev.key == "q":
            print("quit without saving")
            plt.close("all")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="dir containing rgb/ and thermal/")
    ap.add_argument("--frame", default="2000",
                    help="frame index into the sorted pair list, or a filename")
    ap.add_argument("--out", default="points.json")
    ap.add_argument("--panel-height", type=int, default=760)
    ap.add_argument("--zoom-src", type=int, default=140,
                    help="native-pixel width of the magnifier region")
    ap.add_argument("--zoom-out", type=int, default=420)
    ap.add_argument("--backend", choices=["auto", "cv2", "mpl"], default="auto",
                    help="auto picks cv2 if it has a GUI build, else matplotlib")
    args = ap.parse_args()

    rgb_dir = os.path.join(args.raw, "rgb")
    th_dir = os.path.join(args.raw, "thermal")
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    names = sorted(set(f for f in os.listdir(rgb_dir) if f.lower().endswith(exts)) &
                   set(f for f in os.listdir(th_dir) if f.lower().endswith(exts)))
    if not names:
        sys.exit(f"no paired frames under {args.raw}")
    if args.frame.isdigit() and args.frame not in [os.path.splitext(n)[0] for n in names]:
        name = names[min(int(args.frame), len(names) - 1)]
    else:
        name = next((n for n in names if os.path.splitext(n)[0] == args.frame
                     or n == args.frame), None)
        if name is None:
            sys.exit(f"frame {args.frame} not found")

    rgb = cv2.imread(os.path.join(rgb_dir, name), cv2.IMREAD_COLOR)
    th = cv2.imread(os.path.join(th_dir, name), cv2.IMREAD_UNCHANGED)
    if rgb is None or th is None:
        sys.exit(f"failed to read pair {name}")
    if th.ndim == 3:
        th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
    print(f"pair {name}: rgb {rgb.shape[1]}x{rgb.shape[0]}, "
          f"thermal {th.shape[1]}x{th.shape[0]}")
    print(__doc__.split("Keys")[1].split("Pick points")[0])

    p = Picker(rgb, th, args.panel_height, args.zoom_src, args.zoom_out)

    backend = args.backend
    if backend == "auto":
        try:
            cv2.namedWindow("_probe", cv2.WINDOW_AUTOSIZE)
            cv2.destroyWindow("_probe")
            backend = "cv2"
        except cv2.error:
            backend = "mpl"
    print(f"GUI backend: {backend}")
    if backend == "mpl":
        try:
            run_matplotlib(p, name, args.out)
        except ImportError as e:
            sys.exit(f"matplotlib GUI unavailable ({e}). Either install a Qt/Tk "
                     f"backend, or replace opencv-python-headless with "
                     f"opencv-python to use the cv2 backend.")
        return

    win, zwin = "pick_points", "zoom"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(zwin, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, p.on_mouse)

    while True:
        cv2.imshow(win, p.render())
        cv2.imshow(zwin, p.zoom())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            print("quit without saving")
            break
        elif k == ord("u"):
            p.undo()
        elif k == ord("c"):
            p.cmap = (p.cmap + 1) % len(CMAPS)
            p._build_panels()
        elif k == ord("e"):
            p.edges = not p.edges
            p._build_panels()
        elif k == ord("p"):
            H = p.fit()
            if H is not None:
                cv2.imshow("preview  [warp | thermal | blend | thermal-edges]",
                           p.preview(H))
        elif k == ord("s"):
            if p.fit() is None:
                continue
            save_points(p, name, rgb, th, args.out)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Generate a tiny synthetic day/thermal recording pair for end-to-end testing
of the pipeline (extract -> register -> train -> infer -> eval).

Scene: sky gradient + ground band + a moving dark 'UAV' rectangle with a red
'motor' dot. The thermal channel is a deterministic function of RGB (learnable
mapping) rendered at half resolution with a known affine misalignment, so the
manual-registration path can be verified against ground truth.

python tests/make_dummy_data.py --out /tmp/dummy
"""
import argparse
import json
import os

import cv2
import numpy as np

RGB_W, RGB_H = 320, 240
TH_W, TH_H = 160, 120
# rgb -> thermal ground-truth affine: (x, y) -> (0.5x - 8, 0.5y - 6)
M_TRUE = np.array([[0.5, 0.0, -8.0],
                   [0.0, 0.5, -6.0]], np.float32)


def render_rgb(t, n_frames):
    img = np.zeros((RGB_H, RGB_W, 3), np.uint8)
    for y in range(RGB_H):
        f = y / RGB_H
        img[y, :] = (np.array([235, 206, 135]) * (1 - f) +
                     np.array([250, 250, 250]) * f)  # BGR sky
    img[int(RGB_H * 0.8):, :] = (60, 100, 140)       # ground band
    # moving "UAV"
    cx = int(30 + (RGB_W - 80) * t / max(1, n_frames - 1))
    cy = int(60 + 40 * np.sin(2 * np.pi * t / 30))
    cv2.rectangle(img, (cx - 14, cy - 4), (cx + 14, cy + 4), (70, 70, 70), -1)
    cv2.rectangle(img, (cx - 3, cy - 10), (cx + 3, cy + 10), (80, 80, 80), -1)
    cv2.circle(img, (cx, cy), 3, (40, 40, 210), -1)  # red motor dot
    return img, (cx, cy)


def render_thermal(rgb, center):
    b, g, r = [rgb[..., i].astype(np.float32) for i in range(3)]
    base = 0.10 * b + 0.55 * g + 0.35 * r
    base = 255.0 - 0.6 * base                       # sky cold, ground warmer
    cx, cy = center
    cv2.rectangle(base, (cx - 14, cy - 10), (cx + 14, cy + 10), 190.0, -1)
    cv2.circle(base, (cx, cy), 4, 255.0, -1)        # hot motor
    base = cv2.GaussianBlur(base, (7, 7), 2.0)
    th = cv2.warpAffine(base, M_TRUE, (TH_W, TH_H),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    th += np.random.randn(TH_H, TH_W).astype(np.float32) * 2.0
    return np.clip(th, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(0)

    def open_writer(path, size, color):
        for fourcc in ("mp4v", "avc1", "MJPG"):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                args.fps, size, color)
            if w.isOpened():
                return w, True
        return None, False

    rgb_path = os.path.join(args.out, "rgb.mp4")
    th_path = os.path.join(args.out, "thermal.mp4")
    wr, ok_r = open_writer(rgb_path, (RGB_W, RGB_H), True)
    wt, ok_t = open_writer(th_path, (TH_W, TH_H), True)
    use_dirs = not (ok_r and ok_t)
    if use_dirs:
        print("VideoWriter unavailable; writing frame directories instead")
        rgb_path = os.path.join(args.out, "rgb_frames")
        th_path = os.path.join(args.out, "thermal_frames")
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(th_path, exist_ok=True)

    for t in range(args.frames):
        rgb, c = render_rgb(t, args.frames)
        th = render_thermal(rgb, c)
        if use_dirs:
            cv2.imwrite(os.path.join(rgb_path, f"{t:05d}.png"), rgb)
            cv2.imwrite(os.path.join(th_path, f"{t:05d}.png"), th)
        else:
            wr.write(rgb)
            wt.write(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    if not use_dirs:
        wr.release()
        wt.release()

    # ground-truth correspondences for manual registration
    rgb_pts = [[20, 20], [300, 20], [300, 220], [20, 220]]
    th_pts = [[0.5 * x - 8, 0.5 * y - 6] for x, y in rgb_pts]
    with open(os.path.join(args.out, "points.json"), "w") as f:
        json.dump({"rgb": rgb_pts, "thermal": th_pts}, f, indent=2)

    print(json.dumps({"rgb": rgb_path, "thermal": th_path,
                      "fps": args.fps, "frames": args.frames,
                      "frame_dirs": use_dirs}))


if __name__ == "__main__":
    main()

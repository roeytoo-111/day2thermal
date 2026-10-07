r"""
Confirm/reject the RGB-derived drone boxes from rgb_to_thermal_labels.py: one pre-drawn box per frame
(not a 6-candidate list), so gt_boxes.py review doesn't fit. This is a smaller, purpose-built reviewer.

Why review at all: the RGB cascade's "accepted"/"confirmed" boxes are proposals, not ground truth -- some
land on real drones, many don't (notebook 2026-10-01/02). Only reviewed rows are usable as training data.

Keys:
  y        confirm the box as-is
  n        no drone here (becomes "nothing")
  drag     redraw the box (then y to accept the new one)
  space    skip for now (re-asked next run)
  b        back to the previous box (within this run; re-decide it)
  p        play frame-10 .. frame+10 (motion tells a drone from a speck / noise), same crop window
  q        save and quit (resumable; already-reviewed rows are never re-shown)

Mouse buttons:
  CONFIRM   confirm the box as-is
  NOTHING   reject the box
  SKIP      skip for now
  BACK      go back to previous box
  PLAY      play frame-N .. frame+N
  QUIT      save and quit

Safety: --exclude-frame-min/max drops frames in that range before they're ever shown (e.g. a scored val
range) -- impossible to accidentally confirm a label there.
"""

import os
import sys
import csv
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_boxes import FrameReader   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--video", required=True)
    p.add_argument(
        "--boxes",
        required=True,
        help="rgb_to_thermal_labels.py output; reviewed in place"
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="review at most this many 'drone' rows (random)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exclude-frame-min",
        type=int,
        default=None,
        help="drop frame_id >= this before reviewing"
    )
    p.add_argument(
        "--exclude-frame-max",
        type=int,
        default=None,
        help="drop frame_id < this before reviewing"
    )
    p.add_argument(
        "--zoom",
        type=float,
        default=5,
        help="magnification of the box crop (shrunk automatically when the crop would not fit --max-w/--max-h)"
    )
    p.add_argument(
        "--max-w",
        type=int,
        default=1180,
        help="the zoomed crop is shrunk to fit this width (px); big drones otherwise overflow the screen"
    )
    p.add_argument(
        "--max-h",
        type=int,
        default=640,
        help="... and this height (px)"
    )
    p.add_argument(
        "--play-window",
        type=int,
        default=10,
        help="'p' plays frame-N .. frame+N"
    )
    p.add_argument(
        "--play-fps",
        type=float,
        default=8.0,
        help="playback speed for 'p'"
    )
    return p.parse_args()


def main():
    a = parse_args()
    base_zoom = a.zoom

    b = pd.read_csv(a.boxes)

    if "reviewed" not in b.columns:
        b["reviewed"] = ""

    b.reviewed = b.reviewed.fillna("")

    if a.exclude_frame_min is not None:
        excl = b.frame_id >= a.exclude_frame_min
        print(
            f"excluding {excl.sum()} frames >= "
            f"{a.exclude_frame_min} (never shown, never reviewed)"
        )
        b = b[~excl]

    if a.exclude_frame_max is not None:
        excl = b.frame_id < a.exclude_frame_max
        print(
            f"excluding {excl.sum()} frames < "
            f"{a.exclude_frame_max} (never shown, never reviewed)"
        )
        b = b[~excl]

    todo = b[(b.verdict == "drone") & (b.reviewed == "")]

    if a.sample and len(todo) > a.sample:
        todo = todo.sample(a.sample, random_state=a.seed)

    todo = todo.sort_values("frame_id")

    print(
        f"{len(todo)} boxes to review "
        f"({(b.verdict == 'drone').sum()} total, "
        f"{(b.reviewed != '').sum()} already done)"
    )

    if len(todo) == 0:
        return

    fr = FrameReader(
        a.video,
        needed=todo.frame_id.astype(int).tolist()
    )

    # Re-read the complete file so excluded rows remain untouched.
    full = pd.read_csv(a.boxes)

    if "reviewed" not in full.columns:
        full["reviewed"] = ""

    full.reviewed = full.reviewed.fillna("")

    idx_by_frame = {
        int(r.frame_id): i
        for i, r in full.iterrows()
    }

    def save():
        full.to_csv(a.boxes, index=False)

    drag = {
        "down": None,
        "cur": None,
        "box": None
    }

    # ---------------------------------------------------------
    # Mouse buttons
    # ---------------------------------------------------------

    BUTTONS = {
        "confirm": None,
        "nothing": None,
        "skip": None,
        "back": None,
        "play": None,
        "quit": None,
    }

    def draw_button(img, name, x0, y0, x1, y1):
        BUTTONS[name] = (x0, y0, x1, y1)

        cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            (80, 80, 80),
            -1
        )

        cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            (220, 220, 220),
            1
        )

        label = {
            "confirm": "Y  CONFIRM",
            "nothing": "N  NOTHING",
            "skip": "SPACE  SKIP",
            "back": "B  BACK",
            "play": "P  PLAY",
            "quit": "Q  QUIT",
        }[name]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1

        tw, th = cv2.getTextSize(
            label,
            font,
            scale,
            thickness
        )[0]

        tx = x0 + (x1 - x0 - tw) // 2
        ty = y0 + (y1 - y0 + th) // 2

        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    def on_mouse(ev, x, y, flags, param):
        z = a.zoom

        # First handle button clicks.
        if ev == cv2.EVENT_LBUTTONDOWN:

            for name, rect in BUTTONS.items():
                if rect is None:
                    continue

                x0, y0, x1, y1 = rect

                if x0 <= x <= x1 and y0 <= y <= y1:
                    param["button"] = name
                    return

        # Otherwise handle box dragging.
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag["down"] = (x / z, y / z)

        elif ev == cv2.EVENT_MOUSEMOVE and drag["down"] is not None:
            drag["cur"] = (x / z, y / z)

        elif ev == cv2.EVENT_LBUTTONUP and drag["down"] is not None:

            (x0, y0), (x1, y1) = (
                drag["down"],
                (x / z, y / z)
            )

            if abs(x1 - x0) > 1 and abs(y1 - y0) > 1:
                drag["box"] = [
                    min(x0, x1),
                    min(y0, y1),
                    max(x0, x1),
                    max(y0, y1),
                ]

            drag["down"] = drag["cur"] = None

    cv2.namedWindow("confirm RGB-derived box")

    mouse_state = {"button": None}

    cv2.setMouseCallback(
        "confirm RGB-derived box",
        on_mouse,
        mouse_state
    )

    rows = list(todo.itertuples())

    decisions = {}
    # row index in `rows` ->
    # {"verdict": ..., "box": [x0,y0,x1,y1] in ORIGINAL image coords}

    def play(fi, x0c, y0c, x1c, y1c):
        """Replay frame-N .. frame+N through the same crop window."""

        delay = max(
            1,
            int(1000 / a.play_fps)
        )

        for d in range(
            -a.play_window,
            a.play_window + 1
        ):
            nb = fr.read(fi + d)

            if nb is None:
                continue

            c = nb[y0c:y1c, x0c:x1c]

            c = cv2.resize(
                c,
                (
                    int(round(c.shape[1] * a.zoom)),
                    int(round(c.shape[0] * a.zoom))
                ),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.putText(
                c,
                f"frame {fi + d} "
                f"({'+' if d >= 0 else ''}{d})",
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(
                "confirm RGB-derived box",
                c
            )

            if cv2.waitKey(delay) != -1:
                break

    i = 0

    while 0 <= i < len(rows):

        r = rows[i]

        fi = int(r.frame_id)

        im = fr.read(fi)

        if im is None:
            i += 1
            continue

        prev = decisions.get(i)

        box = (
            prev["box"]
            if prev
            else [
                int(round(v))
                for v in (
                    r.x0,
                    r.y0,
                    r.x1,
                    r.y1
                )
            ]
        )

        H, W = im.shape[:2]

        half = max(
            40,
            int(
                max(
                    box[2] - box[0],
                    box[3] - box[1]
                ) * 1.5
            )
        )

        cx = int(
            (box[0] + box[2]) / 2
        )

        cy = int(
            (box[1] + box[3]) / 2
        )

        x0c = max(cx - half, 0)
        y0c = max(cy - half, 0)

        x1c = min(cx + half, W)
        y1c = min(cy + half, H)

        crop0 = im[y0c:y1c, x0c:x1c]

        # effective zoom for THIS frame: the requested one, shrunk (never enlarged) so the window fits the screen
        a.zoom = min(base_zoom, a.max_w / max(crop0.shape[1], 1), a.max_h / max(crop0.shape[0], 1))

        drag["box"] = [
            box[0] - x0c,
            box[1] - y0c,
            box[2] - x0c,
            box[3] - y0c,
        ]

        n_y = sum(
            1
            for d in decisions.values()
            if d["verdict"] == "drone"
        )

        n_n = sum(
            1
            for d in decisions.values()
            if d["verdict"] == "nothing"
        )

        advance = None

        while advance is None:

            crop = cv2.resize(
                crop0,
                (
                    int(round(crop0.shape[1] * a.zoom)),
                    int(round(crop0.shape[0] * a.zoom))
                ),
                interpolation=cv2.INTER_NEAREST
            )

            bx = drag["box"]

            cv2.rectangle(
                crop,
                (
                    int(bx[0] * a.zoom),
                    int(bx[1] * a.zoom)
                ),
                (
                    int(bx[2] * a.zoom),
                    int(bx[3] * a.zoom)
                ),
                (0, 255, 255),
                1
            )

            # Instruction text.
            cv2.putText(
                crop,
                f"frame {fi} ({i + 1}/{len(rows)})",
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            # Reserve space at bottom for buttons.
            button_h = 38
            gap = 6
            margin = 6

            old_h, old_w = crop.shape[:2]

            canvas = np.zeros(
                (
                    old_h + button_h + margin * 2,
                    old_w,
                    3
                ),
                dtype=np.uint8
            )

            canvas[:old_h, :old_w] = crop

            # Status text.
            cv2.putText(
                canvas,
                f"[{n_y} confirmed / {n_n} rejected]",
                (6, old_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            # Button widths.
            widths = {
                "confirm": 115,
                "nothing": 115,
                "skip": 125,
                "back": 90,
                "play": 90,
                "quit": 90,
            }

            x = 6
            y0 = old_h + margin
            y1 = y0 + button_h - 2

            BUTTONS.clear()

            for name in [
                "confirm",
                "nothing",
                "skip",
                "back",
                "play",
                "quit",
            ]:
                w = widths[name]

                if x + w > old_w:
                    break

                draw_button(
                    canvas,
                    name,
                    x,
                    y0,
                    x + w,
                    y1
                )

                x += w + gap

            cv2.imshow(
                "confirm RGB-derived box",
                canvas
            )

            k = cv2.waitKey(30) & 0xFF

            # Mouse button click.
            button = mouse_state["button"]

            if button is not None:
                k = {
                    "confirm": ord("y"),
                    "nothing": ord("n"),
                    "skip": ord(" "),
                    "back": ord("b"),
                    "play": ord("p"),
                    "quit": ord("q"),
                }[button]

                mouse_state["button"] = None

            if drag["down"] is not None or k != 255:

                if k == ord("y"):

                    bx = drag["box"]

                    nb = [
                        bx[0] + x0c,
                        bx[1] + y0c,
                        bx[2] + x0c,
                        bx[3] + y0c,
                    ]

                    decisions[i] = {
                        "verdict": "drone",
                        "box": nb,
                    }

                    advance = 1

                elif k == ord("n"):

                    decisions[i] = {
                        "verdict": "nothing",
                        "box": box,
                    }

                    advance = 1

                elif k == ord("p"):

                    play(
                        fi,
                        x0c,
                        y0c,
                        x1c,
                        y1c
                    )

                elif k == ord("b"):

                    advance = -1

                elif k == ord(" "):

                    # Skip: do NOT add to decisions.
                    # Therefore reviewed stays empty and
                    # this row will appear again next run.
                    advance = 1

                elif k == ord("q"):

                    for j, d in decisions.items():

                        rr = rows[j]

                        full.loc[
                            idx_by_frame[int(rr.frame_id)],
                            [
                                "x0",
                                "y0",
                                "x1",
                                "y1",
                                "verdict",
                                "reviewed",
                            ],
                        ] = [
                            d["box"][0],
                            d["box"][1],
                            d["box"][2],
                            d["box"][3],
                            d["verdict"],
                            "y",
                        ]

                    save()

                    print(
                        f"saved {a.boxes}: "
                        f"{n_y} confirmed, "
                        f"{n_n} rejected this session"
                    )

                    return

        i = max(
            i + advance,
            0
        )

    # Save all decisions after finishing.
    for j, d in decisions.items():

        rr = rows[j]

        full.loc[
            idx_by_frame[int(rr.frame_id)],
            [
                "x0",
                "y0",
                "x1",
                "y1",
                "verdict",
                "reviewed",
            ],
        ] = [
            d["box"][0],
            d["box"][1],
            d["box"][2],
            d["box"][3],
            d["verdict"],
            "y",
        ]

    save()

    n_y = sum(
        1
        for d in decisions.values()
        if d["verdict"] == "drone"
    )

    n_n = sum(
        1
        for d in decisions.values()
        if d["verdict"] == "nothing"
    )

    print(
        f"saved {a.boxes}: "
        f"{n_y} confirmed, "
        f"{n_n} rejected this session"
    )


if __name__ == "__main__":
    main()

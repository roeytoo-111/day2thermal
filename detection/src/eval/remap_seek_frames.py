"""
Re-index labels made on SEEKED frames to SEQUENTIAL frame indices.

gt_boxes.py review / export-yolo read frames with cv2 CAP_PROP_POS_FRAMES (seek); run_yolo_inference.py
and gt_boxes.py propose decode sequentially. For constant-frame-rate files both give the same frame, but
for a variable-frame-rate recording (2026-07-30: nominal 50 fps, 40.9 fps actual -> dropped frames) a seek
to N lands on a different image (measured: sequential N-31 .. N+44; 2026-06-23: N-1 at times). Labels
then point at the wrong frame for anything that reads sequentially (scoring against inference JSONs).

For every labelled frame: seek to it exactly as the review tool did, then find the sequential index whose
decoded image is identical (MD5 of the raw pixels; decoding is deterministic). frame_id is rewritten to
that index; the old value is kept in seek_frame_id. Frames without an exact match are reported and left
unmatched (seq_match = 0) so they can be excluded.

    python3 src/eval/remap_seek_frames.py --video <thermal clip> --boxes <boxes.csv> [--window 200]
"""

import argparse
import hashlib

import cv2
import pandas as pd


def md5(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def remap_manifest(path, found):
    m = pd.read_csv(path)
    if "seek_frame_id" in m.columns:
        print(f"{path}: already remapped")
        return
    m["seek_frame_id"] = m.frame_id
    m["frame_id"] = m.frame_id.map(lambda f: found.get(int(f), int(f)))
    m.to_csv(path, index=False)
    print(f"{path}: frame_id remapped ({int((m.frame_id != m.seek_frame_id).sum())} changed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--window", type=int, default=200, help="search +-window sequential frames around the seek target")
    ap.add_argument("--manifest", default=None,
                    help="also rewrite this sample manifest's frame_id with the same mapping (keeps seek_frame_id)")
    a = ap.parse_args()
    b = pd.read_csv(a.boxes)
    if "seek_frame_id" in b.columns:
        raise SystemExit(f"{a.boxes} already remapped (has seek_frame_id).")
    targets = sorted(set(int(f) for f in b.frame_id))
    # 1) hash of the image the review tool showed for each labelled frame (seek)
    cap = cv2.VideoCapture(a.video)
    seek_hash = {}
    for f in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            seek_hash[f] = md5(im)
    # 2) one sequential pass: hash every frame within reach of any target
    lo, hi = max(min(targets) - a.window, 0), max(targets) + a.window
    want = {}
    for f, h in seek_hash.items():
        want.setdefault(h, []).append(f)
    found = {}
    cap = cv2.VideoCapture(a.video)
    i = 0
    while i <= hi:
        ok, im = cap.read()
        if not ok:
            break
        if i >= lo:
            h = md5(im)
            if h in want:
                for f in want[h]:
                    if abs(i - f) <= a.window and (f not in found or abs(i - f) < abs(found[f] - f)):
                        found[f] = i
        i += 1
    b["seek_frame_id"] = b.frame_id
    b["seq_match"] = b.frame_id.map(lambda f: int(int(f) in found))
    b["frame_id"] = b.frame_id.map(lambda f: found.get(int(f), int(f)))
    off = pd.Series({f: found[f] - f for f in found})
    b.to_csv(a.boxes, index=False)
    if a.manifest:
        remap_manifest(a.manifest, found)
    print(f"{a.boxes}: {len(found)}/{len(targets)} labelled frames matched exactly; "
          f"offset (seq - seek): {int((off != 0).sum())} changed, range {off.min():+d}..{off.max():+d}; "
          f"unmatched {len(targets) - len(found)} (seq_match=0)")


if __name__ == "__main__":
    main()

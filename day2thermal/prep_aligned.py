"""Turn a collection of ALREADY-REGISTERED RGB / IR pair folders into the
``train/val/{rgb,thermal}`` layout that ``train.py`` consumes.

Use this instead of ``register.py`` when the source dataset ships pixel-aligned
pairs (same footprint, same instant) organised as::

    root/<...>/<session>/RGB[-k]/<i>.png     root/<...>/<session>/NIR[-k]/<i>.png

i.e. one directory per recording session holding an RGB sub-folder and an IR
sub-folder whose files pair by identical filename. Sub-folders pair by the
suffix after the modality prefix (``RGB-1`` <-> ``NIR-1``, ``RGB`` <-> ``NIR``).

What it does
------------
* pairs files by name inside each (RGB, IR) sub-folder pair;
* crops both to their common size (top-left anchored) when they differ by a
  pixel or two -- a frequent off-by-one in source datasets. Anything larger
  than ``--max-size-diff`` is treated as a *mis*-registration and skipped;
* writes the IR frame as a single-channel 8-bit PNG (3-channel replicated
  greys are collapsed) and hard-links the RGB frame when it needs no crop, so
  the copy costs little disk;
* renames to ``<session>[_<k>]_<index:06d>.png`` so lexicographic order is
  session-then-frame order (the loaders sort names);
* splits **by session**, never inside one: consecutive frames of one flight
  are near-duplicates, so the only honest hold-out is a whole session
  (``--val-sessions`` explicit, or ``--val-frac`` picks evenly spaced ones);
* writes ``split.json`` next to the split with what went where.

Example
-------
python -m day2thermal.prep_aligned --root ir-rgb-dataset \
    --rgb-prefix RGB --ir-prefix NIR --out data/agri_rgb_nir/aligned \
    --val-sessions canola_06082019,drybean-30072020,Lentwheat_29082018,wheat_27072019
"""
import argparse
import os
import re
import shutil
import sys

import cv2
import numpy as np

from .utils import ensure_dir, save_json

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _numeric_sorted(names):
    def key(n):
        stem = os.path.splitext(n)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(names, key=key)


def _images(d):
    return _numeric_sorted(f for f in os.listdir(d)
                           if os.path.splitext(f)[1].lower() in IMG_EXTS)


def discover_sessions(root, rgb_prefix, ir_prefix):
    """Walk ``root``; a session is any directory that directly contains at
    least one ``<rgb_prefix>*`` and one ``<ir_prefix>*`` sub-directory.
    Returns {session_name: [(suffix, rgb_dir, ir_dir), ...]} sorted."""
    sessions = {}
    for dirpath, dirnames, _ in os.walk(root):
        rgb = {d[len(rgb_prefix):]: d for d in dirnames if d.startswith(rgb_prefix)}
        ir = {d[len(ir_prefix):]: d for d in dirnames if d.startswith(ir_prefix)}
        common = sorted(set(rgb) & set(ir))
        if not common:
            continue
        name = os.path.basename(dirpath.rstrip(os.sep))
        if name in sessions:
            sys.exit(f"duplicate session name '{name}' under {root}; "
                     f"rename one of the directories")
        sessions[name] = [(sfx.lstrip("-_"), os.path.join(dirpath, rgb[sfx]),
                           os.path.join(dirpath, ir[sfx])) for sfx in common]
        dirnames[:] = [d for d in dirnames if d not in rgb.values()
                       and d not in ir.values()]
    return dict(sorted(sessions.items()))


def choose_val(session_names, val_sessions, val_frac):
    if val_sessions:
        wanted = [s.strip() for s in val_sessions.split(",") if s.strip()]
        missing = [s for s in wanted if s not in session_names]
        if missing:
            sys.exit(f"--val-sessions not found: {missing}\n"
                     f"available: {session_names}")
        return set(wanted)
    n = len(session_names)
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    idx = set(np.linspace(0, n - 1, n_val, dtype=int)) if n_val else set()
    return {session_names[i] for i in idx}


def to_gray8(img):
    if img.ndim == 3:
        # replicated-grey 3-channel files are the common case; averaging is
        # exact for those and a sane fallback for genuinely coloured ones
        img = np.round(img.astype(np.float32).mean(axis=2)).astype(img.dtype)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root to walk")
    ap.add_argument("--out", required=True, help="output root (train/ val/ created)")
    ap.add_argument("--rgb-prefix", default="RGB")
    ap.add_argument("--ir-prefix", default="NIR")
    ap.add_argument("--val-sessions", default=None,
                    help="comma-separated session names to hold out")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of SESSIONS held out when --val-sessions is absent")
    ap.add_argument("--max-size-diff", type=int, default=4,
                    help="max |dh|,|dw| between RGB and IR still treated as aligned")
    ap.add_argument("--copy-rgb", action="store_true",
                    help="copy RGB instead of hard-linking")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sessions = discover_sessions(args.root, args.rgb_prefix, args.ir_prefix)
    if not sessions:
        sys.exit(f"no <{args.rgb_prefix}*>/<{args.ir_prefix}*> session dirs under {args.root}")
    names = list(sessions)
    val = choose_val(names, args.val_sessions, args.val_frac)

    report = {"root": os.path.abspath(args.root), "sessions": {},
              "val_sessions": sorted(val), "skipped": []}
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        ensure_dir(os.path.join(args.out, split, "rgb"))
        ensure_dir(os.path.join(args.out, split, "thermal"))

    for sess, parts in sessions.items():
        split = "val" if sess in val else "train"
        n_sess = 0
        for sfx, rgb_dir, ir_dir in parts:
            rgb_names = _images(rgb_dir)
            ir_names = set(_images(ir_dir))
            for i, fn in enumerate(rgb_names):
                if fn not in ir_names:
                    report["skipped"].append(f"{sess}/{sfx}/{fn}: no IR twin")
                    continue
                tag = f"{sess}_{sfx}" if sfx else sess
                out_name = f"{tag}_{i:06d}.png"
                if args.dry_run:
                    n_sess += 1
                    continue
                rgb = cv2.imread(os.path.join(rgb_dir, fn), cv2.IMREAD_COLOR)
                ir = cv2.imread(os.path.join(ir_dir, fn), cv2.IMREAD_UNCHANGED)
                if rgb is None or ir is None:
                    report["skipped"].append(f"{sess}/{sfx}/{fn}: unreadable")
                    continue
                dh, dw = ir.shape[0] - rgb.shape[0], ir.shape[1] - rgb.shape[1]
                if abs(dh) > args.max_size_diff or abs(dw) > args.max_size_diff:
                    report["skipped"].append(
                        f"{sess}/{sfx}/{fn}: size mismatch rgb{rgb.shape[:2]} ir{ir.shape[:2]}")
                    continue
                h = min(rgb.shape[0], ir.shape[0])
                w = min(rgb.shape[1], ir.shape[1])
                ir8 = to_gray8(ir)[:h, :w]
                cv2.imwrite(os.path.join(args.out, split, "thermal", out_name), ir8)
                dst_rgb = os.path.join(args.out, split, "rgb", out_name)
                if rgb.shape[0] != h or rgb.shape[1] != w:
                    cv2.imwrite(dst_rgb, rgb[:h, :w])
                else:
                    src = os.path.join(rgb_dir, fn)
                    if os.path.exists(dst_rgb):
                        os.remove(dst_rgb)
                    if args.copy_rgb:
                        shutil.copy2(src, dst_rgb)
                    else:
                        try:
                            os.link(src, dst_rgb)
                        except OSError:
                            shutil.copy2(src, dst_rgb)
                n_sess += 1
            print(f"{sess}/{sfx or '-'}: {len(rgb_names)} -> {split}", flush=True)
        report["sessions"][sess] = {"split": split, "pairs": n_sess}
        counts[split] += n_sess

    report["counts"] = counts
    if not args.dry_run:
        save_json(report, os.path.join(args.out, "split.json"))
    print(f"train {counts['train']} / val {counts['val']} pairs -> {args.out}"
          f"{' (dry run)' if args.dry_run else ''}; "
          f"skipped {len(report['skipped'])}; val sessions: {sorted(val)}")


if __name__ == "__main__":
    main()

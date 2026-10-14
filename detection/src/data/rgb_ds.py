r"""
Shared helpers for the RGB (Fixed_wing) dataset tools: rgb_audit.py, resplit_by_source.py, propose_rgb_label_fixes.py,
review_label_fixes.py.

  class names   from <data>/data.yaml if present, else the Fixed_wing_v4 / production order uav, airplane, bird.
                (Fixed_wing_v3 on disk is [bird, airplane, uav]; v4's zip has no yaml and uses uav = 0.)
  source key    the video / photo series an image was cut from, from the Roboflow file name:
                  DJI_0025_MP4-0218_jpg.rf.<hash>.jpg   -> DJI_0025_MP4
                  BirdsFlying_446_jpg.rf.<hash>.jpg     -> BirdsFlying
                  boson09_06_16_000684.jpg              -> session09_06_16 (our flight tiles: boson*, fp_boson*,
                                                                           miss*, rv_* of one session share a key)
                Frames of one video are near-duplicates, so a split must never cut through a source.
                Generic camera / download names (img_0029, image_12, frame_3, descarga...) are NOT a series: several
                unrelated subsets share them (v3 has 7,120 "img_*"), so they keep their own key; real sequences are
                then caught by the hash.
  fingerprint   coarse 64-bit dHash (9x8) for candidate search, confirmed by correlating 64x64 high-pass thumbnails.
                dHash alone matches any two skies with the same brightness gradient and chained 14k unrelated v3
                images into one group.
"""
import os
import re
import glob

import cv2
import numpy as np

DEFAULT_NAMES = ["uav", "airplane", "bird"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_names(data_dir):
    y = os.path.join(data_dir, "data.yaml")
    if os.path.exists(y):
        import yaml
        n = yaml.safe_load(open(y)).get("names")
        if isinstance(n, dict):
            n = [n[k] for k in sorted(n)]
        if n:
            return [str(x) for x in n]
    return list(DEFAULT_NAMES)


def find_root(path):
    """Accept the dataset folder or its single child (an extracted zip: <x>/Fixed_wing_v4_.../train)."""
    if any(os.path.isdir(os.path.join(path, s)) for s in ("train", "valid", "test")):
        return path
    subs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(subs) == 1:
        return find_root(subs[0])
    raise SystemExit(f"no train/valid/test under {path}")


def list_split(root, split):
    d = os.path.join(root, split, "images")
    if not os.path.isdir(d):
        return []
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(IMG_EXT))


def label_path(img_path, labels_override=None):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if labels_override:
        p = os.path.join(labels_override, stem + ".txt")
        if os.path.exists(p):
            return p
    return os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", stem + ".txt")


def read_labels(path):
    """[(cls, cx, cy, w, h)] normalised; polygons (Roboflow segment rows) reduced to their bounding box."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        v = line.split()
        if len(v) < 5:
            continue
        c, xs = int(float(v[0])), [float(x) for x in v[1:]]
        if len(xs) > 4:
            px, py = xs[0::2], xs[1::2]
            x0, x1, y0, y1 = min(px), max(px), min(py), max(py)
            xs = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        out.append((c, *xs[:4]))
    return out


_RF = re.compile(r"\.rf\.[0-9a-f]{8,}$")
_EXT = re.compile(r"_(jpe?g|png|webp|bmp)$", re.I)
_VID = re.compile(r"^(.*_(?:mp4|mov|avi|mkv))-\d+$", re.I)
_OURS = re.compile(r"^(?:rv_)?(?:fp_)?(?:boson|miss|trkfp|trk)([0-9_]+?)_\d{6}(?:_\d+)?$")   # our flight tiles -> one key per session
_GENERIC = re.compile(r"^(img|image|images|frame|photo|pic|picture|screenshot|descarga|download|unnamed|untitled|"
                      r"maxresdefault|hqdefault|\d+)$", re.I)
_TAIL = re.compile(r"^(.*?[A-Za-z].*?[A-Za-z].*?[A-Za-z].*?)[-_ ]+\d+[-_ ]*$")


def source_key(path, size=None):
    """size = (W, H) of the image if known: generic names on un-resized, non-square video frames (v4's 07-08 4K
    footage, frame_00001..., in train AND valid) are one video, keyed by prefix + resolution."""
    s = os.path.splitext(os.path.basename(path))[0]
    m = _OURS.match(s)
    if m:
        return "session" + m.group(1)
    s = _RF.sub("", s)
    while _EXT.search(s):
        s = _EXT.sub("", s)
    m = _VID.match(s)
    if m:
        return m.group(1)
    m = _TAIL.match(s)      # one trailing frame / burst number, only if >= 3 letters remain (not "000000000247")
    if m and not _GENERIC.match(m.group(1).strip("-_ ")):
        return m.group(1)
    if m and size and size[0] != size[1]:
        return f"{m.group(1).strip('-_ ')}@{size[0]}x{size[1]}"
    return s


def frame_number(path):
    """Frame / sequence number of a video-frame file name (DJI_0025_MP4-0218_jpg.rf.x -> 218, frame_00008 -> 8), NaN if none."""
    s = _RF.sub("", os.path.splitext(os.path.basename(path))[0])
    while _EXT.search(s):
        s = _EXT.sub("", s)
    m = re.search(r"_(\d{6})(?:_\d+)?$", s) or re.search(r"(\d+)\D*$", s)   # our tiles: <session>_<frame6>[_k]
    return float(m.group(1)) if m else float("nan")


def dhash(gray):
    g = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def fingerprint(gray):
    """(coarse uint64 dHash, 64x64 high-pass thumbnail as float16, its std) of the CENTRAL 60% of a grayscale image:
    Roboflow's rotation augmentation leaves black corners whose edges match between any two images rotated by the
    same angle (they chained 4,381 unrelated v3 sky images)."""
    H, W = gray.shape[:2]
    gray = gray[int(0.2 * H):int(0.8 * H), int(0.2 * W):int(0.8 * W)]
    t = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    h = t - cv2.GaussianBlur(t, (0, 0), 3)
    h = (h - h.mean()) / (h.std() + 1e-6)
    return dhash(gray), h.ravel().astype(np.float16), float((t - cv2.GaussianBlur(t, (0, 0), 3)).std())


def confirm_pairs(pairs, hp_a, hp_b, e_a, e_b, ncc_min=0.9, e_min=6.0):
    """Keep candidate pairs whose high-pass thumbnails correlate (>= ncc_min) and both carry detail (std >= e_min).
    Measured on v3: Roboflow copies of one picture 79% >= 0.9; different sources sharing a sky gradient + horizon
    reach 0.7-0.83 at detail std < 5 (dHash alone matched them)."""
    if not len(pairs):
        return pairs
    pairs = pairs[(e_a[pairs[:, 0]] >= e_min) & (e_b[pairs[:, 1]] >= e_min)]
    keep = np.zeros(len(pairs), bool)
    for k in range(0, len(pairs), 20000):          # batched: 400k pairs x 4096 floats would not fit in RAM
        i, j = pairs[k:k + 20000, 0], pairs[k:k + 20000, 1]
        keep[k:k + 20000] = (hp_a[i].astype(np.float32) * hp_b[j].astype(np.float32)).mean(1) >= ncc_min
    return pairs[keep]


_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming_pairs(a, b, thr, same=False, chunk=2048):
    """Index pairs (i, j) with popcount(a[i] ^ b[j]) <= thr. a, b: uint64 arrays. same=True -> i < j only."""
    a = np.asarray(a, np.uint64); b = np.asarray(b, np.uint64); out = []
    for i0 in range(0, len(a), chunk):
        x = a[i0:i0 + chunk, None] ^ b[None, :]
        d = _POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)
        ii, jj = np.nonzero(d <= thr)
        ii = ii + i0
        if same:
            k = ii < jj; ii, jj = ii[k], jj[k]
        out.append(np.stack([ii, jj], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def box_px(lab, W, H, imgsz=640):
    """sqrt(w*h) of a label in MODEL-INPUT px (letterboxed to imgsz), as in the fw4 README size tables."""
    r = imgsz / max(W, H)
    return float(np.sqrt(lab[3] * W * lab[4] * H) * r)


SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]


def size_bin(px):
    for lo, hi in SIZE_BINS:
        if lo <= px < hi:
            return f"{lo}-{int(hi)}" if hi < 1e9 else f">={lo}"
    return "?"

"""Utility functions shared across the day2thermal pipeline.

This module is deliberately torch-free: the data preparation stages
(frame extraction, registration) and evaluation only need numpy + OpenCV,
so they can run on machines without PyTorch installed.
"""
import json
import os

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory):
    """Sorted list of image filenames in a directory."""
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class ThermalNorm:
    """Converts thermal images between raw pixels, degrees Celsius and the
    normalized [-1, 1] range used by the networks.

    Modes
    -----
    ``rel8``
        8-bit thermal video (the AGC output of most cameras). Pixels are
        *relative* intensities; normalization is ``x / 127.5 - 1``.
        Use this when your thermal recordings are ordinary 8-bit video.

    ``abs16``
        Radiometric 16-bit imagery. Raw counts are first converted to Celsius
        via ``celsius = raw * raw_scale + raw_offset`` (FLIR "TLinear" default:
        0.04 K/count, offset -273.15) and then mapped linearly from
        ``[tmin, tmax]`` C to ``[-1, 1]``. This follows ThermalGAN's idea of
        predicting *calibrated temperatures* rather than display intensities.
    """

    def __init__(self, mode="rel8", raw_scale=0.04, raw_offset=-273.15,
                 tmin=-20.0, tmax=80.0):
        assert mode in ("rel8", "abs16"), mode
        self.mode = mode
        self.raw_scale = float(raw_scale)
        self.raw_offset = float(raw_offset)
        self.tmin = float(tmin)
        self.tmax = float(tmax)

    # ---- raw image -> normalized float32 in [-1, 1] ----------------------
    def to_norm(self, img):
        img = np.asarray(img)
        if self.mode == "rel8":
            if img.dtype == np.uint16:
                img = (img / 257.0).astype(np.uint8)  # crude 16->8 fallback
            x = img.astype(np.float32) / 127.5 - 1.0
        else:
            celsius = img.astype(np.float32) * self.raw_scale + self.raw_offset
            x = self.celsius_to_norm(celsius)
        return np.clip(x, -1.0, 1.0)

    def celsius_to_norm(self, celsius):
        c = (np.asarray(celsius, dtype=np.float32) - self.tmin) / (self.tmax - self.tmin)
        return c * 2.0 - 1.0

    def norm_to_celsius(self, norm):
        c = (np.asarray(norm, dtype=np.float32) + 1.0) / 2.0
        return c * (self.tmax - self.tmin) + self.tmin

    def norm_to_uint8(self, norm):
        x = (np.clip(np.asarray(norm, dtype=np.float32), -1, 1) + 1.0) * 127.5
        return x.astype(np.uint8)

    def norm_to_uint16(self, norm):
        """Inverse of ``to_norm`` for abs16; for rel8 returns a 16-bit stretch."""
        if self.mode == "abs16":
            celsius = self.norm_to_celsius(norm)
            raw = (celsius - self.raw_offset) / self.raw_scale
            return np.clip(raw, 0, 65535).astype(np.uint16)
        x = (np.clip(np.asarray(norm, dtype=np.float32), -1, 1) + 1.0) * 32767.5
        return x.astype(np.uint16)

    def celsius_span(self):
        return self.tmax - self.tmin

    def to_dict(self):
        return dict(mode=self.mode, raw_scale=self.raw_scale,
                    raw_offset=self.raw_offset, tmin=self.tmin, tmax=self.tmax)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def read_thermal(path):
    """Read a thermal frame preserving bit depth, returned as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img  # BGR uint8


def rgb_to_norm(img_bgr):
    """BGR uint8 HWC -> RGB float32 CHW in [-1, 1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb / 127.5 - 1.0).transpose(2, 0, 1)


def norm_to_rgb_uint8(chw):
    """RGB float32 CHW [-1,1] -> RGB uint8 HWC."""
    x = np.clip(chw, -1, 1).transpose(1, 2, 0)
    return ((x + 1.0) * 127.5).astype(np.uint8)


def make_grid_rows(rows):
    """rows: list of rows; each row is a list of HxW (gray) or HxWx3 (RGB)
    uint8 images. Returns one big BGR uint8 image ready for cv2.imwrite."""
    fixed_rows = []
    for row in rows:
        imgs = []
        h = max(im.shape[0] for im in row)
        for im in row:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if im.shape[0] != h:
                im = cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h))
            imgs.append(im)
        fixed_rows.append(np.concatenate(imgs, axis=1))
    w = max(r.shape[1] for r in fixed_rows)
    fixed_rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=0)
        for r in fixed_rows
    ]
    return np.concatenate(fixed_rows, axis=0)

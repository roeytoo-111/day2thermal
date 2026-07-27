"""Paired RGB/thermal dataset.

Expected layout (produced by ``register.py``)::

    root/
      train/rgb/000000.png       train/thermal/000000.png
      val/rgb/...                val/thermal/...

RGB and thermal files pair by identical filename. Thermal files may be 8-bit
(rel8 mode) or radiometric 16-bit (abs16 mode).

Each sample returns:
    A     3xHxW  RGB in [-1, 1]
    B     1xHxW  thermal in [-1, 1]
    T     scalar temperature condition in [-1, 1]
          (from --temps-csv "filename,celsius" if given; otherwise the mean of
           B, a background-temperature proxy in the spirit of ThermalGAN's
           temperature vector)
    name  file stem
"""
import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import list_images, read_rgb, read_thermal, rgb_to_norm


class PairedDayThermalDataset(Dataset):
    def __init__(self, root, phase, thermal_norm, load_size=286, crop_size=256,
                 augment=True, temps_csv=None):
        self.rgb_dir = os.path.join(root, phase, "rgb")
        self.th_dir = os.path.join(root, phase, "thermal")
        rgb = set(list_images(self.rgb_dir))
        th = set(list_images(self.th_dir))
        self.names = sorted(rgb & th)
        if not self.names:
            raise RuntimeError(f"no paired images under {root}/{phase}")
        self.tn = thermal_norm
        self.load_size = load_size
        self.crop_size = crop_size
        self.augment = augment
        self.temps = None
        if temps_csv:
            self.temps = {}
            with open(temps_csv) as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[0].strip():
                        stem = os.path.splitext(row[0].strip())[0]
                        self.temps[stem] = float(row[1])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        a = read_rgb(os.path.join(self.rgb_dir, name))
        b_raw = read_thermal(os.path.join(self.th_dir, name))
        b = self.tn.to_norm(b_raw)  # HxW float32 in [-1, 1]

        cs = self.crop_size
        if self.load_size and self.load_size > 0:
            # pix2pix convention: squash-resize to load_size, then crop
            ls = max(self.load_size, cs)
            a = cv2.resize(a, (ls, ls), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (ls, ls), interpolation=cv2.INTER_AREA)
        else:
            # native resolution: only upscale if smaller than the crop
            h, w = b.shape
            if min(h, w) < cs:
                s = cs / min(h, w)
                new = (max(cs, int(round(w * s))), max(cs, int(round(h * s))))
                a = cv2.resize(a, new, interpolation=cv2.INTER_LINEAR)
                b = cv2.resize(b, new, interpolation=cv2.INTER_LINEAR)

        h, w = b.shape
        if self.augment:
            x = random.randint(0, w - cs)
            y = random.randint(0, h - cs)
        else:
            x = (w - cs) // 2
            y = (h - cs) // 2
        a = a[y:y + cs, x:x + cs]
        b = b[y:y + cs, x:x + cs]

        if self.augment and random.random() < 0.5:
            a = a[:, ::-1].copy()
            b = b[:, ::-1].copy()

        if self.augment and random.random() < 0.8:
            # photometric jitter on RGB only; thermal must stay calibrated
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta = random.uniform(-15.0, 15.0)
            a = np.clip(a.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        A = torch.from_numpy(rgb_to_norm(a))
        B = torch.from_numpy(np.ascontiguousarray(b)).unsqueeze(0)

        stem = os.path.splitext(name)[0]
        if self.temps is not None and stem in self.temps:
            t = float(self.tn.celsius_to_norm(self.temps[stem]))
        else:
            t = float(B.mean())
        return {"A": A, "B": B,
                "T": torch.tensor(t, dtype=torch.float32),
                "name": stem}


def temporal_chunk_split(names, val_frac, chunk):
    """Contiguous-chunk train/val split (same rule as register.py) so that
    near-duplicate neighbouring video frames never straddle the split."""
    nch = max(1, (len(names) + chunk - 1) // chunk)
    n_val = max(1, int(round(val_frac * nch))) if val_frac > 0 else 0
    val_chunks = set(np.linspace(0, nch - 1, n_val, dtype=int)) if n_val else set()
    train = [n for i, n in enumerate(names) if (i // chunk) not in val_chunks]
    val = [n for i, n in enumerate(names) if (i // chunk) in val_chunks]
    return train, val


class UnpairedDayThermalDataset(Dataset):
    """Unpaired RGB / thermal sampler for CycleGAN-style training.

    Needs **no registration**: A and B are drawn independently, so the two
    folders only have to depict the same kind of scene -- not the same instant
    and not the same geometry.

    Layout is the raw extractor output::

        root/rgb/000000.png   root/thermal/000000.png

    Scale matching matters even though the frames are unpaired. A 256px crop
    out of a 3840-wide RGB frame covers ~7% of the field of view, while the
    same crop out of a 640-wide thermal frame covers 40%; the discriminator
    would then be separating the two domains by *zoom level* rather than by
    modality. ``rgb_width`` therefore downscales RGB to roughly the thermal
    frame's angular sampling before cropping (default: the thermal width).
    """

    def __init__(self, rgb_dir, th_dir, thermal_norm, crop_size,
                 rgb_names, th_names, rgb_width=0, augment=True):
        self.rgb_dir = rgb_dir
        self.th_dir = th_dir
        self.tn = thermal_norm
        self.cs = crop_size
        self.rgb_names = list(rgb_names)
        self.th_names = list(th_names)
        if not self.rgb_names or not self.th_names:
            raise RuntimeError(f"empty unpaired split: {len(self.rgb_names)} rgb / "
                               f"{len(self.th_names)} thermal")
        self.augment = augment
        if rgb_width and rgb_width > 0:
            self.rgb_width = rgb_width
        else:
            probe = read_thermal(os.path.join(th_dir, self.th_names[0]))
            self.rgb_width = probe.shape[1]

    def __len__(self):
        return len(self.rgb_names)

    def _crop(self, img):
        cs = self.cs
        h, w = img.shape[:2]
        if min(h, w) < cs:
            s = cs / min(h, w)
            img = cv2.resize(img, (max(cs, int(round(w * s))),
                                   max(cs, int(round(h * s)))),
                             interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
        if self.augment:
            x, y = random.randint(0, w - cs), random.randint(0, h - cs)
        else:
            x, y = (w - cs) // 2, (h - cs) // 2
        img = img[y:y + cs, x:x + cs]
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1].copy()
        return img

    def __getitem__(self, idx):
        a_name = self.rgb_names[idx]
        # A and B are deliberately decoupled -- that is the whole point of the
        # unpaired setting. Deterministic pairing in val keeps samples stable.
        b_name = (random.choice(self.th_names) if self.augment
                  else self.th_names[idx % len(self.th_names)])

        a = read_rgb(os.path.join(self.rgb_dir, a_name))
        h, w = a.shape[:2]
        if w != self.rgb_width:
            new_h = max(1, int(round(h * self.rgb_width / w)))
            a = cv2.resize(a, (self.rgb_width, new_h), interpolation=cv2.INTER_AREA)
        a = self._crop(a)
        if self.augment and random.random() < 0.8:
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta = random.uniform(-15.0, 15.0)
            a = np.clip(a.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        b = self.tn.to_norm(read_thermal(os.path.join(self.th_dir, b_name)))
        b = self._crop(b)

        return {"A": torch.from_numpy(rgb_to_norm(a)),
                "B": torch.from_numpy(np.ascontiguousarray(b)).unsqueeze(0),
                "T": torch.tensor(0.0, dtype=torch.float32),
                "name": os.path.splitext(a_name)[0]}

"""PyTorch helpers: seeding, Gaussian blur, safe padding, tensor->image."""
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_requires_grad(nets, flag):
    if not isinstance(nets, (list, tuple)):
        nets = [nets]
    for net in nets:
        if net is None:
            continue
        for p in net.parameters():
            p.requires_grad = flag


def gaussian_kernel1d(sigma, device, dtype):
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return k / k.sum(), radius


def gaussian_blur(x, sigma):
    """Separable Gaussian blur for NCHW tensors (reflect padding).

    Used to build the low-frequency thermal base S = blur(B): our
    annotation-free surrogate for ThermalGAN's 'thermal segmentation'."""
    if sigma <= 0:
        return x
    k, r = gaussian_kernel1d(sigma, x.device, x.dtype)
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


def _safe_reflect_pad(x, pw, ph):
    """Reflect-pad right/bottom by (pw, ph), chunked so each single pad stays
    smaller than the current spatial size (a hard requirement of F.pad)."""
    while pw > 0 or ph > 0:
        h, w = x.shape[-2:]
        cw = min(pw, w - 1)
        ch = min(ph, h - 1)
        x = F.pad(x, (0, cw, 0, ch), mode="reflect")
        pw -= cw
        ph -= ch
    return x


def pad_to_multiple(x, multiple):
    """Pad NCHW tensor on the right/bottom so H and W are multiples of
    ``multiple``. Returns (padded, (orig_h, orig_w))."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = _safe_reflect_pad(x, pw, ph)
    return x, (h, w)


def unpad(x, hw):
    h, w = hw
    return x[..., :h, :w]


class ImagePool:
    """CycleGAN's history buffer: the discriminator sees a mix of the current
    generator output and older ones. Without it D chases the generator's most
    recent mode and the pair oscillates. ``size=0`` disables it."""

    def __init__(self, size=50):
        self.size = size
        self.items = []

    def query(self, images):
        if self.size <= 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.items) < self.size:
                self.items.append(img)
                out.append(img)
            elif random.random() > 0.5:
                i = random.randint(0, self.size - 1)
                out.append(self.items[i].clone())
                self.items[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def t_rgb_to_uint8(t):
    """3xHxW tensor in [-1,1] -> HxWx3 RGB uint8."""
    a = t.detach().float().cpu().clamp(-1, 1).numpy()
    return ((a.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)


def t_thermal_to_uint8(t):
    """1xHxW tensor in [-1,1] -> HxW uint8."""
    a = t.detach().float().cpu().clamp(-1, 1).numpy()[0]
    return ((a + 1.0) * 127.5).astype(np.uint8)

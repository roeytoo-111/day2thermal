"""Loss functions."""
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """LSGAN (default, more stable) or vanilla (BCE-with-logits, as in the
    original pix2pix / ThermalGAN era). Accepts a tensor or a list of tensors
    (multi-scale discriminator)."""

    def __init__(self, mode="lsgan"):
        super().__init__()
        self.mode = mode
        if mode == "lsgan":
            self.crit = nn.MSELoss()
        elif mode == "vanilla":
            self.crit = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"unknown gan mode '{mode}'")

    def _one(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.crit(pred, target)

    def forward(self, pred, is_real):
        if isinstance(pred, (list, tuple)):
            return sum(self._one(p, is_real) for p in pred) / len(pred)
        return self._one(pred, is_real)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

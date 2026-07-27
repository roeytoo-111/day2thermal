"""Network architectures.

* ``UnetGenerator``  -- pix2pix-style U-Net. With ``extra_head=True`` the
  outermost upsampling is followed by an extra 3x3 conv before the final
  projection, mirroring ThermalGAN's "addition of one convolutional layer and
  one deconvolutional layer" to improve output fidelity.
* ``NLayerDiscriminator`` -- 70x70 PatchGAN.
* ``MultiScaleDiscriminator`` -- pix2pixHD-style multi-scale PatchGAN
  (helps with high-resolution frames and small targets such as distant UAVs).
* ``TemperatureEncoder`` -- small conv encoder producing (mu, logvar); used by
  the optional VAE branch (BicycleGAN heritage kept by ThermalGAN).
"""
import functools

import torch
import torch.nn as nn


def get_norm_layer(name):
    if name == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False,
                                 track_running_stats=False)
    if name == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True,
                                 track_running_stats=True)
    if name == "none":
        return lambda c: nn.Identity()
    raise ValueError(f"unknown norm '{name}'")


def init_weights(net, gain=0.02):
    def init_fn(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and m.weight is not None and \
                ("Conv" in cls or "Linear" in cls):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in cls:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_fn)
    return net


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=None,
                 use_dropout=False, use_bias=True, extra_head=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if outermost:
            if extra_head:
                # ThermalGAN-style extra conv after the last deconv
                up = [uprelu,
                      nn.ConvTranspose2d(inner_nc * 2, inner_nc, 4, 2, 1),
                      norm_layer(inner_nc),
                      nn.ReLU(True),
                      nn.Conv2d(inner_nc, outer_nc, 3, 1, 1),
                      nn.Tanh()]
            else:
                up = [uprelu,
                      nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1),
                      nn.Tanh()]
            model = [downconv, submodule] + up
        elif innermost:
            up = [uprelu,
                  nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=use_bias),
                  norm_layer(outer_nc)]
            model = [downrelu, downconv] + up
        else:
            up = [uprelu,
                  nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=use_bias),
                  norm_layer(outer_nc)]
            model = [downrelu, downconv, norm_layer(inner_nc), submodule] + up
            if use_dropout:
                model = model + [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """U-Net with ``num_downs`` stride-2 encodings. Feed inputs whose H and W
    are multiples of 2**num_downs (training crops should be powers of two)."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm="instance", use_dropout=False, extra_head=True):
        super().__init__()
        assert num_downs >= 5, "num_downs must be >= 5 (crop size >= 32)"
        norm_layer = get_norm_layer(norm)
        use_bias = norm in ("instance", "none")

        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None,
                                        norm_layer=norm_layer, innermost=True,
                                        use_bias=use_bias)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=block,
                                            norm_layer=norm_layer,
                                            use_dropout=use_dropout,
                                            use_bias=use_bias)
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block,
                                        norm_layer=norm_layer, use_bias=use_bias)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block,
                                        norm_layer=norm_layer, use_bias=use_bias)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=block,
                                        norm_layer=norm_layer, use_bias=use_bias)
        block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                        submodule=block, outermost=True,
                                        norm_layer=norm_layer, use_bias=use_bias,
                                        extra_head=extra_head)
        self.model = block

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (defaults reproduce the 70x70 receptive field)."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm="instance"):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        use_bias = norm in ("instance", "none")
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf = 1
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_prev, ndf * nf, kw, 2, padw, bias=use_bias),
                    norm_layer(ndf * nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_prev, ndf * nf, kw, 1, padw, bias=use_bias),
                norm_layer(ndf * nf), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf, 1, kw, 1, padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm="instance",
                 num_scales=2):
        super().__init__()
        self.nets = nn.ModuleList(
            NLayerDiscriminator(input_nc, ndf, n_layers, norm)
            for _ in range(num_scales)
        )
        self.down = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outs = []
        for i, net in enumerate(self.nets):
            outs.append(net(x))
            if i + 1 < len(self.nets):
                x = self.down(x)
        return outs


class TemperatureEncoder(nn.Module):
    """Predicts a latent vector (mu, logvar) from a low-frequency thermal map.
    Optional VAE branch: keeps a piece of ThermalGAN's BicycleGAN heritage and
    can also serve as a latent-temperature consistency term."""

    def __init__(self, input_nc=1, nz=8, ndf=64):
        super().__init__()
        layers = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for _ in range(4):
            nxt = min(mult * 2, 8)
            layers += [nn.Conv2d(ndf * mult, ndf * nxt, 4, 2, 1),
                       nn.LeakyReLU(0.2, True)]
            mult = nxt
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(ndf * mult, nz)
        self.fc_logvar = nn.Linear(ndf * mult, nz)

    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


def define_G(input_nc, output_nc, ngf, num_downs, norm="instance",
             use_dropout=False, extra_head=True):
    return init_weights(UnetGenerator(input_nc, output_nc, num_downs, ngf,
                                      norm, use_dropout, extra_head))


def define_D(input_nc, ndf=64, n_layers=3, norm="instance", num_scales=1):
    if num_scales > 1:
        net = MultiScaleDiscriminator(input_nc, ndf, n_layers, norm, num_scales)
    else:
        net = NLayerDiscriminator(input_nc, ndf, n_layers, norm)
    return init_weights(net)


def define_E(input_nc=1, nz=8, ndf=64):
    return init_weights(TemperatureEncoder(input_nc, nz, ndf))

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FORMAL_MODEL_BACKBONE


def gn(channels: int) -> nn.GroupNorm:
    g = min(8, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class RGBHeatmapUNet(nn.Module):
    def __init__(self, base: int = 32, hm_size: int = 128, img_size: int = 512):
        super().__init__()
        self.hm_size = hm_size
        self.img_size = img_size
        self.enc1 = ConvBlock(3, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(base * 4, base * 4)
        self.down3 = nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1, bias=False)
        self.enc4 = ConvBlock(base * 8, base * 8)
        self.down4 = nn.Conv2d(base * 8, base * 8, 3, stride=2, padding=1, bias=False)
        self.bottleneck = ConvBlock(base * 8, base * 8)
        self.up4 = nn.Conv2d(base * 8, base * 8, 1)
        self.dec4 = ConvBlock(base * 16, base * 4)
        self.up3 = nn.Conv2d(base * 4, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.hm_head = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(base * 2, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))
        d4 = F.interpolate(self.up4(b), size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = F.interpolate(self.up3(d4), size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        hm_logits = self.hm_head(d3)
        if hm_logits.shape[-1] != self.hm_size:
            hm_logits = F.interpolate(hm_logits, size=(self.hm_size, self.hm_size), mode="bilinear", align_corners=False)
        return hm_logits


def create_model(args) -> nn.Module:
    if args.model_backbone != FORMAL_MODEL_BACKBONE:
        raise ValueError(f"The released experiment uses model_backbone={FORMAL_MODEL_BACKBONE!r}")
    return RGBHeatmapUNet(base=args.base_channels, hm_size=args.hm_size, img_size=args.img_size)

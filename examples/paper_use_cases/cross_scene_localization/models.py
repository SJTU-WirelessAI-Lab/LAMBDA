from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class ResNet18Heatmap(nn.Module):
    """ResNet18 encoder + lightweight deconvolution heatmap head.

    This is intended as a stronger, standard vision baseline. With
    pretrained=True, the encoder is initialized from ImageNet weights. The output
    is a single-channel heatmap at hm_size.
    """
    def __init__(self, hm_size: int = 128, img_size: int = 512, pretrained: bool = False):
        super().__init__()
        self.hm_size = hm_size
        self.img_size = img_size
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception as e:
            if pretrained:
                raise RuntimeError(
                    "Requested --model-backbone resnet18_imagenet, but ImageNet weights could not be loaded. "
                    "For paper experiments this must not silently fall back to random init. "
                    "Please pre-download/cache torchvision ResNet18 weights or use --model-backbone resnet18_scratch."
                ) from e
            backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False), gn(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), gn(128), nn.GELU(),  # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False), gn(64), nn.GELU(),    # 32 -> 64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False), gn(32), nn.GELU(),     # 64 -> 128
            nn.Conv2d(32, 1, 1),
        )
        # Initialize newly added head.
        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        hm_logits = self.head(x)
        if hm_logits.shape[-1] != self.hm_size:
            hm_logits = F.interpolate(hm_logits, size=(self.hm_size, self.hm_size), mode="bilinear", align_corners=False)
        return hm_logits


def create_model(args) -> nn.Module:
    if args.model_backbone == "rgb_unet":
        return RGBHeatmapUNet(base=args.base_channels, hm_size=args.hm_size, img_size=args.img_size)
    if args.model_backbone == "resnet18_scratch":
        return ResNet18Heatmap(hm_size=args.hm_size, img_size=args.img_size, pretrained=False)
    if args.model_backbone == "resnet18_imagenet":
        return ResNet18Heatmap(hm_size=args.hm_size, img_size=args.img_size, pretrained=True)
    raise ValueError(f"Unknown model_backbone: {args.model_backbone}")

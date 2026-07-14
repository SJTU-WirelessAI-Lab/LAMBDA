from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class RGBBeamNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 64,
        pretrained: bool = True,
        dropout: float = 0.25,
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        if backbone == "resnet18":
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                net = models.resnet18(weights=weights)
            except Exception:
                net = models.resnet18(pretrained=pretrained)
            in_dim = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
        elif backbone == "resnet50_paper":
            # Match the public paper repository: build a ResNet-50 with the
            # target number of beam classes, initialize the replacement fc
            # layer with Xavier normal and zero bias, then load ImageNet
            # weights with fc skipped.
            net = models.resnet50(weights=None, num_classes=n_classes)
            in_dim = net.fc.in_features
            init_beam_head(net.fc, official_resnet50=True)
            if pretrained:
                state = torch.hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                    progress=True,
                )
                state = {k: v for k, v in state.items() if not k.startswith("fc.")}
                net.load_state_dict(state, strict=False)
        else:
            raise ValueError(f"Unknown backbone {backbone!r}")
        self.net = net
        self.backbone = backbone

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["image"])


def model_file_stem(backbone: str) -> str:
    if backbone == "resnet50_paper":
        return "rgb60_resnet50_paper"
    return "rgb60_resnet18"


def assert_model_structure(model: nn.Module, n_classes: int) -> None:
    backbone = getattr(model, "backbone", "")
    if backbone != "resnet50_paper":
        return
    net = getattr(model, "net", None)
    head = getattr(net, "fc", None)
    if not isinstance(head, nn.Linear):
        raise RuntimeError(
            "Paper ResNet-50 requires a single Linear task head; "
            f"got {type(head).__name__}."
        )
    if head.in_features != 2048 or head.out_features != n_classes:
        raise RuntimeError(
            "Paper ResNet-50 task head must be Linear(2048, n_classes); "
            f"got Linear({head.in_features}, {head.out_features})."
        )


def init_beam_head(module: nn.Module, official_resnet50: bool = False) -> None:
    if official_resnet50 and isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif hasattr(module, "reset_parameters"):
        module.reset_parameters()


def reset_beam_head(model: nn.Module) -> None:
    """Reinitialize only the task head, keeping the visual backbone as-is."""
    head = getattr(getattr(model, "net", None), "fc", None)
    if head is None:
        raise RuntimeError("Model does not expose net.fc task head")
    official_resnet50 = getattr(model, "backbone", "") == "resnet50_paper"
    for module in head.modules():
        init_beam_head(module, official_resnet50=official_resnet50)

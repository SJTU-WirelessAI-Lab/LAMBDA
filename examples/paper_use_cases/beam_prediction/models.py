from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import FORMAL_BACKBONE


class RGBBeamNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 64,
        backbone: str = FORMAL_BACKBONE,
    ) -> None:
        super().__init__()
        if backbone != FORMAL_BACKBONE:
            raise ValueError(f"The released experiment uses backbone={FORMAL_BACKBONE!r}")
        # Formal model: ImageNet-initialized ResNet-50 with a freshly
        # initialized 64-way classification head.
        net = models.resnet50(weights=None, num_classes=n_classes)
        init_beam_head(net.fc)
        state = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            progress=True,
        )
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        net.load_state_dict(state, strict=False)
        self.net = net
        self.backbone = backbone

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["image"])


def model_file_stem(backbone: str) -> str:
    if backbone != FORMAL_BACKBONE:
        raise ValueError(f"The released experiment uses backbone={FORMAL_BACKBONE!r}")
    return "rgb60_resnet50_paper"


def assert_model_structure(model: nn.Module, n_classes: int) -> None:
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


def init_beam_head(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
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
    for module in head.modules():
        init_beam_head(module)

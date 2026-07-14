from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def modified_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos_inds = (targets >= 1.0 - eps).float()
    neg_inds = (targets < 1.0 - eps).float()
    neg_weights = torch.pow(1.0 - targets, beta)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    num_pos = pos_inds.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


@torch.no_grad()
def decode_heatmaps_local(logits: torch.Tensor, img_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = torch.sigmoid(logits[:, 0])
    b, h, w = prob.shape
    flat = prob.view(b, -1)
    conf, idx = flat.max(dim=1)
    ys = (idx // w).long()
    xs = (idx % w).long()
    coords = []
    for i in range(b):
        x0 = int(xs[i].item())
        y0 = int(ys[i].item())
        x_min = max(0, x0 - 1)
        x_max = min(w - 1, x0 + 1)
        y_min = max(0, y0 - 1)
        y_max = min(h - 1, y0 + 1)
        patch = prob[i, y_min:y_max + 1, x_min:x_max + 1]
        yy, xx = torch.meshgrid(
            torch.arange(y_min, y_max + 1, device=prob.device, dtype=torch.float32),
            torch.arange(x_min, x_max + 1, device=prob.device, dtype=torch.float32),
            indexing="ij",
        )
        weight = patch + 1e-6
        x = (xx * weight).sum() / weight.sum()
        y = (yy * weight).sum() / weight.sum()
        coords.append((x.item(), y.item()))
    hm_to_img = img_size / float(w)
    u_img = np.array([x * hm_to_img for x, _ in coords], dtype=np.float32)
    v_img = np.array([y * hm_to_img for _, y in coords], dtype=np.float32)
    return u_img, v_img, conf.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def decode_heatmaps_topk_local(
    logits: torch.Tensor,
    img_size: int,
    topk: int = 10,
    nms_kernel: int = 9,
) -> List[List[Dict[str, float]]]:
    """Decode heatmap local top-k peaks.

    Returns a list with length B. Each element is a list of candidate dicts
    sorted by heatmap confidence rank. Coordinates are in the 512 model crop.
    """
    prob = torch.sigmoid(logits[:, 0])
    b, h, w = prob.shape
    if nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be odd.")

    pooled = F.max_pool2d(prob[:, None], kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)[:, 0]
    keep = prob == pooled
    masked = prob.masked_fill(~keep, -1.0)

    flat = masked.view(b, -1)
    k = min(int(topk), flat.shape[1])
    confs, idxs = torch.topk(flat, k=k, dim=1)

    hm_to_img = img_size / float(w)
    out: List[List[Dict[str, float]]] = []
    for bi in range(b):
        peaks: List[Dict[str, float]] = []
        for r in range(k):
            conf = float(confs[bi, r].item())
            idx = int(idxs[bi, r].item())
            y0 = idx // w
            x0 = idx % w

            # Same 3x3 weighted local refinement as argmax decoding.
            x_min = max(0, x0 - 1)
            x_max = min(w - 1, x0 + 1)
            y_min = max(0, y0 - 1)
            y_max = min(h - 1, y0 + 1)
            patch = prob[bi, y_min:y_max + 1, x_min:x_max + 1]
            yy, xx = torch.meshgrid(
                torch.arange(y_min, y_max + 1, device=prob.device, dtype=torch.float32),
                torch.arange(x_min, x_max + 1, device=prob.device, dtype=torch.float32),
                indexing="ij",
            )
            weight = patch + 1e-6
            xr = float((xx * weight).sum().item() / weight.sum().item())
            yr = float((yy * weight).sum().item() / weight.sum().item())

            peaks.append({
                "rank": float(r + 1),
                "u_img": float(xr * hm_to_img),
                "v_img": float(yr * hm_to_img),
                "hm_x": float(xr),
                "hm_y": float(yr),
                "confidence": conf,
            })
        out.append(peaks)
    return out

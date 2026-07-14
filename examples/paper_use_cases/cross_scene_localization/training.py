from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import CROP_SIZE
from .data import collate_heatmap
from .decoding import decode_heatmaps_local, modified_focal_loss
from .metrics import compute_pixel_metrics


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
                      collate_fn=collate_heatmap)


def run_train_epoch(model, loader, optimizer, device, grad_clip: float) -> Dict[str, float]:
    model.train()
    total = 0.0
    steps = 0
    for x, hm, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        hm = hm.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = modified_focal_loss(logits, hm)
        if torch.isfinite(loss):
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item())
            steps += 1
    return {"train_loss": total / max(steps, 1)}


@torch.no_grad()
def evaluate_2d_only(model, loader, device, img_size: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    errs = []
    for x, _, metas in tqdm(loader, desc="eval2d", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        u_pred, v_pred, conf = decode_heatmaps_local(logits, img_size)
        for i, m in enumerate(metas):
            e = float(np.linalg.norm([u_pred[i] - float(m["u_gt_img"]), v_pred[i] - float(m["v_gt_img"])]))
            errs.append(e)
            rows.append({
                "scene": m["scene"], "fid": m["fid"], "height": m["height"],
                "u_gt_img": float(m["u_gt_img"]), "v_gt_img": float(m["v_gt_img"]),
                "u_pred_img": float(u_pred[i]), "v_pred_img": float(v_pred[i]),
                "center_error_px_512": e,
                "center_error_px_raw": e * (CROP_SIZE / img_size),
                "peak_confidence": float(conf[i]),
            })
    return compute_pixel_metrics(np.array(errs, dtype=np.float32)), pd.DataFrame(rows)

from __future__ import annotations

import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


def to_device(batch: Dict[str, object], device: str) -> Dict[str, object]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(1))
    pred = logits.topk(k, dim=1).indices
    return (pred == target.unsqueeze(1)).any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    n, t1, t3, t5, loss_sum = 0, 0.0, 0.0, 0.0, 0.0
    within1, within3, abs_err_sum, signed_err_sum = 0.0, 0.0, 0.0, 0.0
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        target = batch["label"]
        loss = F.cross_entropy(logits, target)
        pred1 = logits.argmax(dim=1)
        err = (pred1 - target).float()
        abs_err = err.abs()
        bs = target.size(0)
        loss_sum += float(loss.item()) * bs
        t1 += topk_accuracy(logits, target, 1) * bs
        t3 += topk_accuracy(logits, target, 3) * bs
        t5 += topk_accuracy(logits, target, 5) * bs
        within1 += (abs_err <= 1).float().sum().item()
        within3 += (abs_err <= 3).float().sum().item()
        abs_err_sum += abs_err.sum().item()
        signed_err_sum += err.sum().item()
        n += bs
    return {
        "loss": loss_sum / max(n, 1),
        "top1": t1 / max(n, 1),
        "top3": t3 / max(n, 1),
        "top5": t5 / max(n, 1),
        "within_1_beam": within1 / max(n, 1),
        "within_3_beams": within3 / max(n, 1),
        "mae_beams": abs_err_sum / max(n, 1),
        "bias_beams": signed_err_sum / max(n, 1),
        "n": n,
    }


def train_model(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    log_prefix: str = "",
    amp: bool = True,
    log_interval: int = 200,
    epoch_log_interval: int = 1,
    restore_best: bool = False,
    best_metric: str = "top1",
    scheduler_name: str = "cosine",
    lr_milestones: Sequence[int] = (4, 8, 12),
    lr_gamma: float = 0.1,
    optimizer_name: str = "adamw",
) -> Optional[Dict[str, float]]:
    model.to(device)
    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name!r}")
    if scheduler_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    elif scheduler_name == "paper_step":
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(lr_milestones), gamma=lr_gamma)
    else:
        raise ValueError(f"Unknown scheduler {scheduler_name!r}")
    weights = class_weights.to(device) if class_weights is not None else None
    use_amp = bool(amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score: Optional[float] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Optional[Dict[str, float]] = None
    for ep in range(epochs):
        model.train()
        loss_sum, n_sum, top1_sum = 0.0, 0, 0.0
        t0 = time.time()
        for step, batch in enumerate(loader_train, start=1):
            batch = to_device(batch, device)
            target = batch["label"]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch)
                loss = F.cross_entropy(logits, target, weight=weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = target.size(0)
            loss_sum += float(loss.item()) * bs
            n_sum += bs
            top1_sum += topk_accuracy(logits, target, 1) * bs
            if log_interval > 0 and step % log_interval == 0:
                seen = min(step * loader_train.batch_size, len(loader_train.dataset))
                ips = n_sum / max(time.time() - t0, 1e-6)
                print(
                    f"{log_prefix}epoch {ep+1}/{epochs} step {step}/{len(loader_train)} "
                    f"seen={seen}/{len(loader_train.dataset)} "
                    f"loss={loss_sum/max(n_sum,1):.4f} "
                    f"top1={top1_sum/max(n_sum,1):.4f} "
                    f"img/s={ips:.1f}",
                    flush=True,
                )
        sched.step()
        msg = (
            f"{log_prefix}epoch {ep+1}/{epochs} "
            f"loss={loss_sum/max(n_sum,1):.4f} "
            f"top1={top1_sum/max(n_sum,1):.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        should_log_epoch = (
            epoch_log_interval > 0
            and (ep == 0 or (ep + 1) == epochs or (ep + 1) % epoch_log_interval == 0)
        )
        val = evaluate(model, loader_val, device) if loader_val is not None and (restore_best or should_log_epoch) else None
        if val is not None and restore_best:
            if best_metric not in val:
                raise KeyError(f"best_metric {best_metric!r} not found in validation metrics: {sorted(val)}")
            score = -float(val[best_metric]) if best_metric in {"loss", "mae_beams"} else float(val[best_metric])
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = dict(val)
                best_metrics["epoch"] = ep + 1
                best_metrics["best_metric"] = best_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if should_log_epoch:
            if val is not None:
                best_note = ""
                if restore_best and best_metrics is not None:
                    best_note = f" | best {best_metric}=epoch{int(best_metrics['epoch'])}:{best_metrics[best_metric]:.4f}"
                msg += (
                    f" | val loss={val['loss']:.4f} top1={val['top1']:.4f} "
                    f"top3={val['top3']:.4f} top5={val['top5']:.4f} "
                    f"mae={val['mae_beams']:.2f} beams{best_note}"
                )
            print(msg, flush=True)
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restored best validation model from epoch {int(best_metrics['epoch'])}: "
            f"{best_metric}={best_metrics[best_metric]:.4f}",
            flush=True,
        )
    return best_metrics


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def class_weights_from_labels(labels: np.ndarray, n_classes: int = 64) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    return torch.from_numpy(weights)


def labels_from_dataset(ds: Dataset) -> np.ndarray:
    if isinstance(ds, ConcatDataset):
        return np.concatenate([labels_from_dataset(d) for d in ds.datasets], axis=0)
    if hasattr(ds, "labels"):
        return np.asarray(getattr(ds, "labels"), dtype=np.int64)
    raise TypeError(f"Cannot extract labels from dataset type {type(ds)}")


def label_summary(labels: np.ndarray) -> Dict[str, object]:
    uniq, counts = np.unique(labels, return_counts=True)
    top = sorted(zip(counts.tolist(), uniq.tolist()), reverse=True)[:8]
    return {
        "n": int(len(labels)),
        "unique": int(len(uniq)),
        "min": int(labels.min()) if len(labels) else None,
        "max": int(labels.max()) if len(labels) else None,
        "top_counts": [{"beam": int(b), "count": int(c)} for c, b in top],
    }


def make_loader_kwargs(num_workers: int, device: str) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return kwargs

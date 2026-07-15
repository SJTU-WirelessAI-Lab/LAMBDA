from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .artifacts import save_checkpoint
from .models import reset_beam_head
from .training import evaluate, label_summary, make_loader_kwargs, train_model


def sample_stratified_indices(labels: np.ndarray, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int64)
    idx_all = np.arange(len(labels))
    classes = np.unique(labels)
    chosen: List[int] = []
    per_class = max(1, k // max(len(classes), 1))
    for cls in classes:
        cls_idx = idx_all[labels == cls].copy()
        rng.shuffle(cls_idx)
        chosen.extend(cls_idx[:per_class].tolist())
    chosen = sorted(set(chosen))
    if len(chosen) < k:
        pool = np.setdiff1d(idx_all, np.array(chosen, dtype=np.int64), assume_unique=False)
        extra = rng.choice(pool, size=min(k - len(chosen), len(pool)), replace=False)
        chosen.extend(extra.tolist())
    if len(chosen) > k:
        chosen = rng.choice(np.array(chosen, dtype=np.int64), size=k, replace=False).tolist()
    return sorted(int(i) for i in chosen)


def sample_distribution_matched_indices(labels: np.ndarray, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int64)
    idx_all = np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)

    if k <= len(classes):
        chosen: List[int] = []
        for cls in classes:
            cls_idx = idx_all[labels == cls].copy()
            chosen.append(int(rng.choice(cls_idx)))
        if len(chosen) > k:
            chosen = rng.choice(np.array(chosen, dtype=np.int64), size=k, replace=False).tolist()
        return sorted(int(i) for i in chosen)

    expected = counts.astype(np.float64) / float(len(labels)) * float(k)
    quotas = np.floor(expected).astype(np.int64)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)

    while int(quotas.sum()) > k:
        reducible = np.where(quotas > 1)[0]
        if len(reducible) == 0:
            break
        surplus = expected[reducible] - quotas[reducible]
        j = reducible[int(np.argmin(surplus))]
        quotas[j] -= 1

    while int(quotas.sum()) < k:
        addable = np.where(quotas < counts)[0]
        if len(addable) == 0:
            break
        remainder = expected[addable] - quotas[addable]
        j = addable[int(np.argmax(remainder))]
        quotas[j] += 1

    chosen = []
    for cls, q in zip(classes, quotas):
        cls_idx = idx_all[labels == cls].copy()
        rng.shuffle(cls_idx)
        chosen.extend(cls_idx[: int(q)].tolist())
    return sorted(int(i) for i in chosen)


def sample_few_shot_indices(labels: np.ndarray, k: int, seed: int, strategy: str) -> List[int]:
    if strategy == "stratified":
        return sample_stratified_indices(labels, k, seed)
    if strategy == "coverage_distribution":
        return sample_distribution_matched_indices(labels, k, seed)
    raise ValueError(f"Unknown few-shot sampling strategy {strategy!r}")


def indices_hash(indices: Sequence[int]) -> str:
    arr = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def load_few_shot_manifest(path: str) -> Dict[int, List[int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    raw_indices = raw.get("indices", raw)
    return {int(k): [int(i) for i in v] for k, v in raw_indices.items()}


def save_few_shot_manifest(path: str, labels: np.ndarray, indices_by_k: Dict[int, List[int]]) -> None:
    out = {
        "dataset": "DeepSense scenario23",
        "n_frames": int(len(labels)),
        "label_summary": label_summary(labels),
        "indices": {str(k): [int(i) for i in v] for k, v in sorted(indices_by_k.items())},
        "selection": {
            str(k): {
                **few_shot_selection_summary(labels, v),
                "indices_hash": indices_hash(v),
            }
            for k, v in sorted(indices_by_k.items())
        },
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)


def few_shot_selection_summary(labels: np.ndarray, chosen: Sequence[int], n_classes: int = 64) -> Dict[str, object]:
    labels = labels.astype(np.int64)
    chosen_labels = labels[np.asarray(chosen, dtype=np.int64)]
    full_counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    chosen_counts = np.bincount(chosen_labels, minlength=n_classes).astype(np.float64)
    full_probs = full_counts / max(float(full_counts.sum()), 1.0)
    chosen_probs = chosen_counts / max(float(chosen_counts.sum()), 1.0)
    present = full_counts > 0
    l1 = float(np.abs(chosen_probs[present] - full_probs[present]).sum())
    missing = np.where((full_counts > 0) & (chosen_counts == 0))[0].astype(int).tolist()
    return {
        "selected_unique": int(np.count_nonzero(chosen_counts)),
        "dataset_unique": int(np.count_nonzero(full_counts)),
        "missing_present_labels": missing,
        "distribution_l1_present": l1,
        "top_selected_counts": [
            {"beam": int(i), "count": int(c)}
            for i, c in sorted(enumerate(chosen_counts.astype(int)), key=lambda x: x[1], reverse=True)[:8]
            if c > 0
        ],
    }


def choose_finetune_schedule(k: int, base_epochs: int, base_lr: float, batch_size: int) -> Tuple[int, float]:
    steps_per_epoch = max(1, math.ceil(k / max(1, min(batch_size, k))))
    min_steps = 160
    max_epochs = max(base_epochs, 1) * 12
    epochs = max(base_epochs, int(math.ceil(min_steps / steps_per_epoch)))
    epochs = min(epochs, max_epochs)
    if k <= 64:
        lr = base_lr * 0.25
    elif k <= 256:
        lr = base_lr * 0.5
    else:
        lr = base_lr
    return epochs, lr


def few_shot_finetune_deepsense(
    base_model: nn.Module,
    deepsense_train_aug: Dataset,
    deepsense_eval_fixed: Dataset,
    k_shots: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    device: str,
    seed: int = 0,
    weight_decay: float = 1e-4,
    amp: bool = True,
    log_interval: int = 0,
    auto_schedule: bool = True,
    reset_head: bool = False,
    sampling_strategy: str = "stratified",
    manifest_indices: Optional[Dict[int, List[int]]] = None,
    scheduler_name: str = "cosine",
    lr_milestones: Sequence[int] = (4, 8, 12),
    lr_gamma: float = 0.1,
    optimizer_name: str = "adamw",
    checkpoint_stem: str = "rgb60_resnet50_paper",
    save_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    labels = np.asarray(getattr(deepsense_eval_fixed, "labels"), dtype=np.int64)
    idx_all = np.arange(len(labels))
    loader_kwargs = make_loader_kwargs(num_workers, device)
    results: Dict[str, Dict[str, float]] = {}

    for k in k_shots:
        k = int(k)
        if k <= 0 or k >= len(labels):
            print(f"  [few-shot K={k}] skipped; expected 0 < K < {len(labels)}")
            continue
        ft_epochs, ft_lr = choose_finetune_schedule(k, epochs, lr, batch_size) if auto_schedule else (epochs, lr)
        if manifest_indices is not None and k in manifest_indices:
            chosen = sorted(int(i) for i in manifest_indices[k])
            selection_source = "manifest"
        else:
            chosen = sample_few_shot_indices(labels, k, seed + k, sampling_strategy)
            selection_source = "generated"
        rest = sorted(np.setdiff1d(idx_all, np.array(chosen, dtype=np.int64)).tolist())
        selection_info = few_shot_selection_summary(labels, chosen)
        selection_info["indices_hash"] = indices_hash(chosen)
        selection_info["source"] = selection_source

        ft_loader = DataLoader(
            Subset(deepsense_train_aug, chosen),
            batch_size=min(batch_size, k),
            shuffle=True,
            **loader_kwargs,
        )
        eval_loader = DataLoader(
            Subset(deepsense_eval_fixed, rest),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        print(
            f"  [few-shot K={k}] train={len(chosen)} eval={len(rest)} "
            f"epochs={ft_epochs} lr={ft_lr:.2e} sampling={sampling_strategy} "
            f"unique={selection_info['selected_unique']}/{selection_info['dataset_unique']} "
            f"dist_l1={selection_info['distribution_l1_present']:.3f}"
        )
        model = deepcopy(base_model)
        if reset_head:
            reset_beam_head(model)
        train_model(
            model,
            ft_loader,
            None,
            epochs=ft_epochs,
            lr=ft_lr,
            device=device,
            weight_decay=weight_decay,
            class_weights=None,
            log_prefix=f"  [few-shot K={k}] ",
            amp=amp,
            log_interval=log_interval,
            epoch_log_interval=max(1, ft_epochs // 4),
            scheduler_name=scheduler_name,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            optimizer_name=optimizer_name,
        )
        metrics = evaluate(model, eval_loader, device)
        metrics["k_shot"] = k
        metrics["epochs_used"] = ft_epochs
        metrics["lr_used"] = ft_lr
        metrics["n_finetune"] = len(chosen)
        metrics["n_eval"] = len(rest)
        metrics["reset_head"] = bool(reset_head)
        metrics["sampling_strategy"] = sampling_strategy
        metrics["selection"] = selection_info
        results[str(k)] = metrics
        print(
            f"  K={k:>5d}  top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} "
            f"top5={metrics['top5']:.4f} within1={metrics['within_1_beam']:.4f} "
            f"within3={metrics['within_3_beams']:.4f} "
            f"mae={metrics['mae_beams']:.2f} bias={metrics['bias_beams']:.2f} "
            f"n={metrics['n']}"
        )
        if save_dir is not None:
            save_checkpoint(save_dir / f"{checkpoint_stem}_deepsense_fewshot_k{k}.pt", model, argparse.Namespace(), metrics)
    return results

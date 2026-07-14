from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .data import DeepSenseRGB60BeamDataset, LambdaRGB60BeamDataset
from .geometry import PhotoULACodebook
from .training import label_summary, labels_from_dataset


def build_train_dataset(
    args,
    codebook: PhotoULACodebook,
    augment: bool,
    log: bool = True,
) -> Tuple[Dataset, np.ndarray, List[Dict[str, object]]]:
    datasets: List[Dataset] = []
    per_scene: List[Dict[str, object]] = []
    for scene in args.train_scenes:
        if log:
            print(f"Loading LAMBDA train scene {scene} ...")
        ds = LambdaRGB60BeamDataset(
            args.data_root,
            scene,
            codebook,
            stride=args.stride,
            img_size=args.img_size,
            augment=augment,
            center_codebook_crop=args.center_codebook_crop,
            rgb_hfov=args.rgb_hfov,
            cache_dir=args.cache_dir,
            limit=args.limit_train_per_scene,
            label_mode=args.lambda_label_mode,
            codebook_frame=args.lambda_codebook_frame,
        )
        datasets.append(ds)
        info = {"scene": scene, **label_summary(ds.labels)}
        angle = ds.strongest_angles
        info.update(
            {
                "strongest_angle_p01": float(np.percentile(angle, 1)),
                "strongest_angle_p99": float(np.percentile(angle, 99)),
                "inside_codebook_pct": float(100.0 * np.mean((angle >= -45.0) & (angle <= 45.0))),
            }
        )
        per_scene.append(info)
        if log:
            print(
                f"  {scene}: n={info['n']} unique={info['unique']}/64 "
                f"label_range=[{info['min']},{info['max']}] "
                f"inside90={info['inside_codebook_pct']:.2f}%"
            )
    if len(datasets) == 1:
        train_ds = datasets[0]
    else:
        train_ds = ConcatDataset(datasets)
    labels = labels_from_dataset(train_ds)
    return train_ds, labels, per_scene


def build_test_dataset(args) -> Tuple[Dataset, np.ndarray, Dict[str, object]]:
    if args.test_dataset != "deepsense":
        raise ValueError("Only --test_dataset deepsense is implemented for this experiment.")
    print("Loading DeepSense scenario23 test set ...")
    ds = DeepSenseRGB60BeamDataset(
        args.deepsense_root,
        csv_name=args.deepsense_csv,
        stride=args.test_stride,
        img_size=args.img_size,
        augment=False,
        center_codebook_crop=args.center_codebook_crop,
        rgb_hfov=args.rgb_hfov,
        codebook_fov=args.codebook_fov,
        label_source=args.deepsense_label_source,
        limit=args.limit_test,
    )
    labels = ds.labels
    info = {"scene": ds.scene, **label_summary(labels)}
    print(f"  DeepSense: n={info['n']} unique={info['unique']}/64 label_range=[{info['min']},{info['max']}]")
    return ds, labels, info


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    summary: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "summary": summary,
        },
        path,
    )


def load_beam_head(model: nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    own = model.state_dict()
    head_keys = [k for k in own.keys() if k.startswith("net.fc.")]
    missing = [k for k in head_keys if k not in state]
    if missing:
        raise RuntimeError(f"Head checkpoint missing keys: {missing}")
    copied = {}
    for k in head_keys:
        if own[k].shape != state[k].shape:
            raise RuntimeError(f"Shape mismatch for {k}: model={tuple(own[k].shape)} ckpt={tuple(state[k].shape)}")
        copied[k] = state[k]
    model.load_state_dict(copied, strict=False)
    print(f"Loaded beam head from checkpoint: {ckpt_path} ({len(copied)} tensors)")

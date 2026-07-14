from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import ExperimentSpec
from .data import RGBHeatmapDataset
from .evaluation import (
    evaluate_gt2d_lidar,
    evaluate_model_with_lidar,
    evaluate_model_with_lidar_topk_lidar,
    make_prediction_montage,
)
from .geometry import seed_everything
from .metrics import baseline_pixel_results
from .models import create_model
from .sampling import build_train_val_test_samples
from .training import evaluate_2d_only, make_loader, run_train_epoch


def _copy_model_state(model) -> Dict[str, torch.Tensor]:
    inner = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}


def _load_model_state(model, state: Dict[str, torch.Tensor], device) -> None:
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.load_state_dict({k: v.to(device) for k, v in state.items()})


def summarize_row(exp_name: str, method: str, metrics: Dict[str, float], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {"experiment": exp_name, "method": method}
    row.update(metrics)
    if extra:
        row.update(extra)
    return row


def get_scale_run_tag(args) -> str:
    if getattr(args, "scale_unit", "count_per_height") == "percent_uv":
        pct = getattr(args, "train_scale_percent", None)
        if pct is None:
            pct = 100.0
        pct_str = (f"{float(pct):g}").replace(".", "p")
        return f"train_pct_{pct_str}"
    val = getattr(args, "train_max_samples_per_height", None)
    return f"train_count_{val if val is not None else 'all'}"


def run_one_method(exp: ExperimentSpec, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]], args, device) -> Dict[str, Any]:
    seed_everything(args.seed)
    test_ds = RGBHeatmapDataset(test_samples, args.img_size, args.hm_size, args.hm_sigma, augment=False, allow_hflip=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = create_model(args).to(device)
    if torch.cuda.device_count() > 1 and not args.no_dataparallel:
        model = nn.DataParallel(model)

    if args.control_mode == "untrained":
        print(f"[{exp.name} | rgb_lidar_untrained] evaluating random-initialized model")
        test_metrics, test_details = evaluate_model_with_lidar(model, test_loader, device, args.img_size, args, "pred2d_lidar_untrained")
        method_name = "rgb_lidar_untrained"
        method_dir = os.path.join(args.out_dir, exp.name, f"{get_scale_run_tag(args)}_test_scale_{args.test_max_samples_per_height}_seed_{args.seed}_{args.model_backbone}", method_name)
        os.makedirs(method_dir, exist_ok=True)
        pd.DataFrame([{"epoch": 0, "control_mode": "untrained"}]).to_csv(os.path.join(method_dir, "train_val_curve.csv"), index=False)
        details_path = os.path.join(method_dir, "test_details.csv")
        test_details.to_csv(details_path, index=False)
        make_prediction_montage(test_details, os.path.join(method_dir, "test_prediction_montage.jpg"), max_images=args.montage_samples)
        return summarize_row(exp.name, method_name, test_metrics, {
            "control_mode": "untrained", "model_backbone": args.model_backbone, "sampling_mode": args.scale_unit, "train_scale_percent": args.train_scale_percent, "train_n": len(train_samples), "val_n": len(val_samples), "test_n": len(test_samples), "train_scale_per_height": args.train_max_samples_per_height, "test_scale_per_height": args.test_max_samples_per_height, "test_seed": args.test_seed, "val_seed": args.val_seed, "split_seed": args.split_seed, "subset_seed": args.subset_seed, "seed": args.seed, "best_epoch": 0, "selection_metric": "none_untrained",
            "test_details_path": details_path,
        })

    if args.control_mode != "normal":
        raise ValueError(f"run_one_method only handles normal/untrained; got {args.control_mode}")

    train_ds = RGBHeatmapDataset(train_samples, args.img_size, args.hm_size, args.hm_sigma, augment=True, allow_hflip=args.allow_hflip)
    val_ds = RGBHeatmapDataset(val_samples, args.img_size, args.hm_size, args.hm_sigma, augment=False, allow_hflip=False)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    best_score = float("inf")
    best_epoch = 0
    stopper = 0
    curves = []

    for ep in range(1, args.epochs + 1):
        losses = run_train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_metrics, _ = evaluate_2d_only(model, val_loader, device, args.img_size)
        score = val_metrics["mean_px"] if args.selection_metric == "mean_px" else val_metrics["p90_px"]
        row = {"epoch": ep, **losses, "selection_score": score, "control_mode": "normal"}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        curves.append(row)
        print(f"[{exp.name} | rgb_lidar | Ep {ep:02d}] loss={losses['train_loss']:.4f} "
              f"val_center={val_metrics['mean_px']:.2f}px val_PCK@8={val_metrics.get('pck@8', 0):.3f}")
        if score < best_score:
            best_score = score
            best_val_metrics = val_metrics
            best_state = _copy_model_state(model)
            best_epoch = ep
            stopper = 0
        else:
            stopper += 1
        if stopper >= args.early_stop:
            print(f"  early stop at epoch {ep}; best val epoch={best_epoch}")
            break

    assert best_state is not None and best_val_metrics is not None
    _load_model_state(model, best_state, device)
    result_rows: List[Dict[str, Any]] = []

    # Original single-peak argmax decoding.
    test_metrics, test_details = evaluate_model_with_lidar(model, test_loader, device, args.img_size, args, "pred2d_lidar_trained_argmax")

    method_name = "rgb_lidar"
    method_dir = os.path.join(args.out_dir, exp.name, f"{get_scale_run_tag(args)}_test_scale_{args.test_max_samples_per_height}_seed_{args.seed}_{args.model_backbone}", method_name)
    os.makedirs(method_dir, exist_ok=True)
    pd.DataFrame(curves).to_csv(os.path.join(method_dir, "train_val_curve.csv"), index=False)
    details_path = os.path.join(method_dir, "test_details.csv")
    test_details.to_csv(details_path, index=False)
    make_prediction_montage(test_details, os.path.join(method_dir, "test_prediction_montage.jpg"), max_images=args.montage_samples)
    common_extra = {
        "control_mode": "normal",
        "model_backbone": args.model_backbone,
        "sampling_mode": args.scale_unit,
        "train_scale_percent": args.train_scale_percent,
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "test_n": len(test_samples),
        "train_scale_per_height": args.train_max_samples_per_height,
        "test_scale_per_height": args.test_max_samples_per_height,
        "test_seed": args.test_seed,
        "val_seed": args.val_seed,
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": f"val_{args.selection_metric}",
        "best_selection_score": best_score,
        "best_val_mean_px": best_val_metrics.get("mean_px", float("nan")),
        "topk_lidar_k": getattr(args, "topk_lidar_k", float("nan")),
        "topk_lidar_nms_kernel": getattr(args, "topk_lidar_nms_kernel", float("nan")),
        "decode_mode": "argmax",
    }
    result_rows.append(summarize_row(exp.name, method_name, test_metrics, {**common_extra, "test_details_path": details_path}))

    # Deployable top-k heatmap + LiDAR-valid reranking.
    if args.eval_topk_lidar:
        topk_metrics, topk_details = evaluate_model_with_lidar_topk_lidar(model, test_loader, device, args.img_size, args, "pred2d_lidar_trained_topk_lidar")
        method_name = "rgb_lidar_topk_lidar"
        method_dir = os.path.join(args.out_dir, exp.name, f"{get_scale_run_tag(args)}_test_scale_{args.test_max_samples_per_height}_seed_{args.seed}_{args.model_backbone}", method_name)
        os.makedirs(method_dir, exist_ok=True)
        pd.DataFrame(curves).to_csv(os.path.join(method_dir, "train_val_curve.csv"), index=False)
        topk_details_path = os.path.join(method_dir, "test_details.csv")
        topk_details.to_csv(topk_details_path, index=False)
        topk_candidates = topk_details.attrs.get("topk_candidates", None)
        if isinstance(topk_candidates, pd.DataFrame):
            topk_candidates.to_csv(os.path.join(method_dir, "topk_candidates.csv"), index=False)
        make_prediction_montage(topk_details, os.path.join(method_dir, "test_prediction_montage.jpg"), max_images=args.montage_samples)
        result_rows.append(summarize_row(exp.name, method_name, topk_metrics, {
            **common_extra,
            "decode_mode": "topk_lidar_first_valid_else_argmax",
            "test_details_path": topk_details_path,
        }))

    return result_rows


def run_gt2d_lidar(exp: ExperimentSpec, test_samples: List[Dict[str, Any]], args, train_n: Optional[int] = None, val_n: Optional[int] = None) -> Dict[str, Any]:
    metrics, details = evaluate_gt2d_lidar(test_samples, args.img_size, args, source="gt2d_lidar")
    method_dir = os.path.join(args.out_dir, exp.name, f"{get_scale_run_tag(args)}_test_scale_{args.test_max_samples_per_height}_seed_{args.seed}_{args.model_backbone}", "gt2d_lidar")
    os.makedirs(method_dir, exist_ok=True)
    details_path = os.path.join(method_dir, "test_details.csv")
    details.to_csv(details_path, index=False)
    make_prediction_montage(details, os.path.join(method_dir, "test_prediction_montage.jpg"), max_images=args.montage_samples)
    return summarize_row(exp.name, "gt2d_lidar", metrics, {
        "control_mode": "gt2d_lidar",
        "model_backbone": args.model_backbone,
        "sampling_mode": args.scale_unit, "train_scale_percent": args.train_scale_percent, "train_n": train_n, "val_n": val_n, "test_n": len(test_samples), "train_scale_per_height": args.train_max_samples_per_height, "test_scale_per_height": args.test_max_samples_per_height, "test_seed": args.test_seed, "val_seed": args.val_seed, "split_seed": args.split_seed, "subset_seed": args.subset_seed,
        "seed": args.seed,
        "best_epoch": 0,
        "selection_metric": "none_gt2d_lidar",
        "test_details_path": details_path,
    })


def run_experiment(exp: ExperimentSpec, args, device) -> List[Dict[str, Any]]:
    train_samples, val_samples, test_samples = build_train_val_test_samples(exp, args)
    rows: List[Dict[str, Any]] = []
    exp_dir = os.path.join(args.out_dir, exp.name, f"{get_scale_run_tag(args)}_test_scale_{args.test_max_samples_per_height}_seed_{args.seed}_{args.model_backbone}")
    os.makedirs(exp_dir, exist_ok=True)

    if args.audit_sampling_only:
        audit = getattr(args, "_last_sampling_audit", {}) or {}
        row = summarize_row(exp.name, "sampling_audit_only", {}, {
            "control_mode": "audit",
            "model_backbone": args.model_backbone,
            "sampling_mode": args.scale_unit,
            "train_scale_percent": args.train_scale_percent,
            "train_scale_per_height": args.train_max_samples_per_height,
            "test_scale_per_height": args.test_max_samples_per_height,
            "seed": args.seed,
            "val_seed": args.val_seed,
            "split_seed": args.split_seed,
            "subset_seed": args.subset_seed,
            "test_seed": args.test_seed,
            **audit,
        })
        return [row]

    g_metrics, g_df = baseline_pixel_results(train_samples, test_samples, by_height=False)
    h_metrics, h_df = baseline_pixel_results(train_samples, test_samples, by_height=True)
    g_df.to_csv(os.path.join(exp_dir, "global_mean_pixel_baseline_test_details.csv"), index=False)
    h_df.to_csv(os.path.join(exp_dir, "height_mean_pixel_baseline_test_details.csv"), index=False)
    rows.append(summarize_row(exp.name, "global_mean_pixel_baseline", g_metrics, {"selection_metric": "none", "control_mode": "baseline", "model_backbone": args.model_backbone, "sampling_mode": args.scale_unit, "train_scale_percent": args.train_scale_percent, "train_n": len(train_samples), "val_n": len(val_samples), "test_n": len(test_samples), "train_scale_per_height": args.train_max_samples_per_height, "test_scale_per_height": args.test_max_samples_per_height, "test_seed": args.test_seed, "val_seed": args.val_seed, "split_seed": args.split_seed, "subset_seed": args.subset_seed, "seed": args.seed}))
    rows.append(summarize_row(exp.name, "height_mean_pixel_baseline", h_metrics, {"selection_metric": "none", "control_mode": "baseline", "model_backbone": args.model_backbone, "sampling_mode": args.scale_unit, "train_scale_percent": args.train_scale_percent, "train_n": len(train_samples), "val_n": len(val_samples), "test_n": len(test_samples), "train_scale_per_height": args.train_max_samples_per_height, "test_scale_per_height": args.test_max_samples_per_height, "test_seed": args.test_seed, "val_seed": args.val_seed, "split_seed": args.split_seed, "subset_seed": args.subset_seed, "seed": args.seed}))
    print(f"[{exp.name} | global_mean_pixel_baseline | TEST] center={g_metrics['mean_px']:.2f}px PCK@16={g_metrics['pck@16']:.3f}")
    print(f"[{exp.name} | height_mean_pixel_baseline | TEST] center={h_metrics['mean_px']:.2f}px PCK@16={h_metrics['pck@16']:.3f}")

    # Always include GT-2D + LiDAR upper bound unless explicitly skipped.
    if not args.no_gt2d_lidar:
        rows.append(run_gt2d_lidar(exp, test_samples, args, train_n=len(train_samples), val_n=len(val_samples)))

    def _append_result(obj):
        if isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)

    if args.control_mode in ("normal", "untrained"):
        _append_result(run_one_method(exp, train_samples, val_samples, test_samples, args, device))
    elif args.control_mode == "all":
        original_mode = args.control_mode
        args.control_mode = "normal"
        _append_result(run_one_method(exp, train_samples, val_samples, test_samples, args, device))
        args.control_mode = "untrained"
        _append_result(run_one_method(exp, train_samples, val_samples, test_samples, args, device))
        args.control_mode = original_mode
    elif args.control_mode == "gt2d_lidar":
        # Already evaluated above. No model run.
        pass
    else:
        raise ValueError(f"Unknown control mode: {args.control_mode}")
    return rows

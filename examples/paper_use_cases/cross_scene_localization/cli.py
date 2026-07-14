from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from .config import (
    CROP_U_MAX,
    CROP_U_MIN,
    CROP_V_MAX,
    CROP_V_MIN,
    DATASET_ROOT_CONFIGURED,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENTS,
)
from .experiment import run_experiment
from .geometry import seed_everything
from .sampling import get_experiment_by_name


def build_parser():
    p = argparse.ArgumentParser(description="V19.7 RGB heatmap + LiDAR with model-seed protocol, restored augmentation, and top-k LiDAR reranking.")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--experiments", nargs="+", default=["cross_scene_block1_square1_mid_70_80_90_100_110"], choices=[e.name for e in EXPERIMENTS])
    p.add_argument("--control-mode", default="normal", choices=["normal", "untrained", "gt2d_lidar", "all"], help="Use all to run GT2D upper bound + trained normal + untrained in one invocation.")
    p.add_argument("--no-gt2d-lidar", action="store_true", help="Do not evaluate the GT 2D + LiDAR upper bound.")

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--seed-list", default="", help="Optional comma-separated seeds for stability runs, e.g. 2026,2027,2028. Overrides --seed when non-empty.")
    p.add_argument("--split-seed", type=int, default=2026, help="Fixed seed for train-pool/validation split. Keep fixed across --seed-list for model-seed-only repeats.")
    p.add_argument("--subset-seed", type=int, default=2026, help="Fixed seed for nested train subset ordering. Keep fixed across --seed-list and scales.")
    p.add_argument("--scale-list", default="", help="For --scale-unit percent_uv, comma-separated percentages, e.g. 10,25,50,100. For count_per_height, comma-separated samples per height, e.g. 80,200,400,800.")
    p.add_argument("--scale-unit", default="percent_uv", choices=["percent_uv", "count_per_height"], help="percent_uv uses height+v_bin+u_bin stratified nested sampling by percentage. count_per_height keeps the V19.2 height-only count protocol.")
    p.add_argument("--uv-u-bins", type=int, default=4, help="Number of u_img bins for V19.7 joint height+image-position stratification.")
    p.add_argument("--uv-v-bins", type=int, default=4, help="Number of v_img bins for V19.7 joint height+image-position stratification.")
    p.add_argument("--min-stratum-size-for-val", type=int, default=5, help="A stratum must have at least this many samples before a validation sample is carved out.")
    p.add_argument("--min-one-per-nonempty-stratum", action=argparse.BooleanOptionalAction, default=True, help="When percent scale is >0, take at least one sample from every non-empty train stratum.")
    p.add_argument("--audit-sampling-only", action="store_true", help="Build train/val/test samples and write sampling audit CSVs, then exit without training/evaluation.")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--hm-size", type=int, default=128)
    p.add_argument("--hm-sigma", type=float, default=1.8)
    p.add_argument("--frame-stride", type=int, default=6, help="Default 6 keeps RGB/pose fids aligned to LiDAR fids saved every 3 frames.")
    p.add_argument("--max-samples-per-height", type=int, default=1200, help="Backward-compatible alias for --train-max-samples-per-height when the latter is not set. Use -1 for no train cap.")
    p.add_argument("--train-max-samples-per-height", type=int, default=None, help="Cap only the training/validation source samples per height. This is what --scale-list controls.")
    p.add_argument("--test-max-samples-per-height", type=int, default=-1, help="Cap only the test samples per height. Default -1 means no cap, so all test scales/seeds use the same test set.")
    p.add_argument("--test-seed", type=int, default=999999, help="Fixed seed used only when test-max-samples-per-height is capped; keep fixed across scale/seed runs.")
    p.add_argument("--val-seed", type=int, default=None, help="Backward-compatible alias for --split-seed. If omitted, --split-seed is used.")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--require-lidar-file", action=argparse.BooleanOptionalAction, default=True,
                   help="Require exact lidar_{fid}.pcd. Default true for clean RGB/LiDAR alignment.")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--model-backbone", default="rgb_unet", choices=["rgb_unet", "resnet18_scratch", "resnet18_imagenet"], help="rgb_unet is the original v18 heatmap model; resnet18_imagenet is the stronger pretrained baseline.")
    p.add_argument("--selection-metric", default="mean_px", choices=["mean_px", "p90_px"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-dataparallel", action="store_true")
    p.add_argument("--allow-hflip", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable horizontal flip augmentation. Default true, restored from V19.5. Use --no-allow-hflip or --no-hflip to disable.")
    p.add_argument("--no-hflip", action="store_true", help="Backward-compatible alias for --no-allow-hflip.")

    p.add_argument("--lidar-pixel-radius", type=float, default=12.0, help="Raw-image pixel radius for LiDAR projection association. Default 12 from top-k sweep audit.")
    p.add_argument("--lidar-range-strategy", default="median", choices=["minrange", "p10range", "median", "nearest_pixel"])
    p.add_argument("--lidar-min-range", type=float, default=1.0)
    p.add_argument("--lidar-max-range", type=float, default=150.0)
    p.add_argument("--eval-topk-lidar", action=argparse.BooleanOptionalAction, default=True,
                   help="Also evaluate deployable top-k heatmap + first LiDAR-valid reranking.")
    p.add_argument("--topk-lidar-k", type=int, default=10, help="Number of local heatmap peaks used by top-k LiDAR reranking.")
    p.add_argument("--topk-lidar-nms-kernel", type=int, default=9, help="Odd local-NMS kernel on heatmap grid for top-k peak extraction.")
    p.add_argument("--montage-samples", type=int, default=60)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not DATASET_ROOT_CONFIGURED:
        parser.error("Set LAMBDA_DATA_ROOT to the prepared public dataset root.")
    if args.val_seed is None:
        args.val_seed = args.split_seed
    else:
        args.split_seed = args.val_seed
    if args.no_hflip:
        args.allow_hflip = False
    # Backward compatibility: if train cap is not explicitly set, use --max-samples-per-height.
    if args.train_max_samples_per_height is None:
        args.train_max_samples_per_height = args.max_samples_per_height
    if args.train_max_samples_per_height is not None and args.train_max_samples_per_height < 0:
        args.train_max_samples_per_height = None
    if args.test_max_samples_per_height is not None and args.test_max_samples_per_height < 0:
        args.test_max_samples_per_height = None

    os.makedirs(args.out_dir, exist_ok=True)
    args.train_scale_percent = None
    scale_values = [args.train_max_samples_per_height]
    if args.scale_unit == "percent_uv":
        scale_values = [100.0]
    if args.scale_list.strip():
        # scale_list controls TRAIN subset size only. Full source -> fixed val + train pool first; test set remains fixed.
        if args.scale_unit == "percent_uv":
            scale_values = [None if x.strip().lower() in ("none", "all", "-1") else float(x.strip()) for x in args.scale_list.split(",") if x.strip()]
        else:
            scale_values = [None if x.strip().lower() in ("none", "all", "-1") else int(x.strip()) for x in args.scale_list.split(",") if x.strip()]
    seed_values = [args.seed]
    if args.seed_list.strip():
        seed_values = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device} | GPUs={torch.cuda.device_count()}")
    print(f"Output dir: {args.out_dir}")
    print(f"Frame stride: {args.frame_stride}; LiDAR exact-fid required={args.require_lidar_file}")
    print(f"Crop-visible range: u=[{CROP_U_MIN},{CROP_U_MAX}), v=[{CROP_V_MIN},{CROP_V_MAX})")
    print(f"LiDAR association: radius={args.lidar_pixel_radius}px raw, strategy={args.lidar_range_strategy}, range=[{args.lidar_min_range},{args.lidar_max_range}]m")
    print(f"Model backbone: {args.model_backbone}; nested_train_scales={scale_values}; scale_unit={args.scale_unit}; uv_bins=({args.uv_u_bins},{args.uv_v_bins}); test_cap={args.test_max_samples_per_height}; model_seeds={seed_values}; split_seed={args.split_seed}; subset_seed={args.subset_seed}; test_seed={args.test_seed}; allow_hflip={args.allow_hflip}")

    all_rows = []
    original_seed = args.seed
    original_scale = args.train_max_samples_per_height
    for scale in scale_values:
        if args.scale_unit == "percent_uv":
            args.train_scale_percent = 100.0 if scale is None else float(scale)
            args.train_max_samples_per_height = None
        else:
            args.train_scale_percent = None
            args.train_max_samples_per_height = scale
        for seed in seed_values:
            args.seed = seed
            seed_everything(args.seed)
            for exp_name in args.experiments:
                exp = get_experiment_by_name(exp_name)
                print(f"\n\n================ Experiment: {exp.name} | train_scale={scale} | test_cap={args.test_max_samples_per_height} | seed={seed} | backbone={args.model_backbone} ================")
                all_rows.extend(run_experiment(exp, args, device))
    args.seed = original_seed
    args.train_max_samples_per_height = original_scale

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.out_dir, "summary_v19_7_rgb_lidar.csv")
    summary.to_csv(summary_path, index=False)

    show_cols = [
        "experiment", "method", "decode_mode", "control_mode", "model_backbone", "sampling_mode", "train_scale_percent", "train_n", "val_n", "test_n", "train_scale_per_height", "test_scale_per_height", "seed", "val_seed", "split_seed", "subset_seed", "test_seed",
        "mean_px", "median_px", "p90_px", "pck@4", "pck@8", "pck@16",
        "lidar_candidate_rate", "lidar_valid_rate", "lidar_candidate_count_mean",
        "success_at_0p5m", "success_at_1p0m", "success_at_2p0m",
        "range_lidar_mae_m", "range_lidar_p50_m", "range_lidar_p90_m",
        "xyz_lidar_mean_m", "xyz_lidar_p50_m", "xyz_lidar_p90_m",
        "xyz_error_pred2d_gt_range_m_mean", "xyz_error_pred2d_gt_range_m_p50", "xyz_error_pred2d_gt_range_m_p90",
        "topk_used_lidar_candidate_rate", "topk_selected_rank_mean", "topk_valid_candidate_count_mean",
        "topk_lidar_k", "topk_lidar_nms_kernel", "best_epoch", "selection_metric",
    ]
    existing = [c for c in show_cols if c in summary.columns]
    print("\n========== V19.7 RGB+LiDAR Summary ==========")
    print(summary[existing].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved summary: {summary_path}")

from __future__ import annotations

import argparse
import os


def _bootstrap_data_root() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root")
    args, _ = parser.parse_known_args()
    if args.data_root:
        os.environ["LAMBDA_DATA_ROOT"] = os.path.expanduser(args.data_root)


_bootstrap_data_root()


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
    FORMAL_EXPERIMENT_NAME,
    FORMAL_LIDAR_RANGE_STRATEGY,
    FORMAL_MODEL_BACKBONE,
    FORMAL_SCALE_UNIT,
    FORMAL_SELECTION_METRIC,
)
from .experiment import run_experiment
from .geometry import seed_everything, validate_sensor_configuration
from .sampling import get_experiment_by_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="v20 RGB-LiDAR localization from Block 1 to Square 1."
    )
    p.add_argument(
        "--data-root",
        default=os.environ.get("LAMBDA_DATA_ROOT"),
        help="Prepared LAMBDA dataset root. Falls back to LAMBDA_DATA_ROOT.",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--experiments",
        nargs="+",
        default=[FORMAL_EXPERIMENT_NAME],
        choices=[e.name for e in EXPERIMENTS],
    )
    p.add_argument("--control-mode", default="normal", choices=["normal"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seed-list",
        default="",
        help="Optional comma-separated model seeds for repeated runs. Overrides --seed.",
    )
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--subset-seed", type=int, default=2026)
    p.add_argument("--test-seed", type=int, default=999999)
    p.add_argument("--scale-list", default="5,10,25,50,75,100")
    p.add_argument("--scale-unit", default=FORMAL_SCALE_UNIT, choices=[FORMAL_SCALE_UNIT])
    p.add_argument("--uv-u-bins", type=int, default=4)
    p.add_argument("--uv-v-bins", type=int, default=4)
    p.add_argument("--min-stratum-size-for-val", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-max-samples-per-height", type=int, default=500)

    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--hm-size", type=int, default=128)
    p.add_argument("--hm-sigma", type=float, default=1.8)
    p.add_argument("--frame-stride", type=int, default=6)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--model-backbone", default=FORMAL_MODEL_BACKBONE, choices=[FORMAL_MODEL_BACKBONE])
    p.add_argument("--selection-metric", default=FORMAL_SELECTION_METRIC, choices=[FORMAL_SELECTION_METRIC])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-dataparallel", action="store_true")

    p.add_argument("--lidar-pixel-radius", type=float, default=12.0)
    p.add_argument(
        "--lidar-range-strategy",
        default=FORMAL_LIDAR_RANGE_STRATEGY,
        choices=[FORMAL_LIDAR_RANGE_STRATEGY],
    )
    p.add_argument("--lidar-min-range", type=float, default=1.0)
    p.add_argument("--lidar-max-range", type=float, default=150.0)
    p.add_argument("--eval-topk-lidar", dest="eval_topk_lidar", action="store_true")
    p.add_argument("--no-eval-topk-lidar", dest="eval_topk_lidar", action="store_false")
    p.set_defaults(eval_topk_lidar=True)
    p.add_argument("--topk-lidar-k", type=int, default=10)
    p.add_argument("--topk-lidar-nms-kernel", type=int, default=9)
    p.add_argument("--montage-samples", type=int, default=0)
    return p


def _parse_scale_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("--scale-list must contain at least one percentage.")
    if any(value <= 0 or value > 100 for value in values):
        raise ValueError("Each --scale-list percentage must be in (0, 100].")
    return values


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not DATASET_ROOT_CONFIGURED:
        parser.error("Pass --data-root or set LAMBDA_DATA_ROOT to the prepared dataset root.")

    try:
        scale_values = _parse_scale_list(args.scale_list)
    except ValueError as exc:
        parser.error(str(exc))
    seed_values = [args.seed]
    if args.seed_list.strip():
        seed_values = [int(item.strip()) for item in args.seed_list.split(",") if item.strip()]

    selected_experiments = [get_experiment_by_name(name) for name in args.experiments]
    validate_sensor_configuration(
        [root for exp in selected_experiments for root in (exp.train_root, exp.test_root)]
    )

    # Fixed workflow values not exposed as alternative CLI branches.
    args.allow_hflip = True
    args.require_lidar_file = True
    args.train_max_samples_per_height = None
    args.train_scale_percent = None
    args.val_seed = args.split_seed

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device} | GPUs={torch.cuda.device_count()}")
    print(f"Output dir: {args.out_dir}")
    print(f"Frame stride: {args.frame_stride}; LiDAR exact-fid required=True")
    print(f"Crop-visible range: u=[{CROP_U_MIN},{CROP_U_MAX}), v=[{CROP_V_MIN},{CROP_V_MAX})")
    print(
        f"LiDAR association: radius={args.lidar_pixel_radius}px raw, "
        f"strategy={args.lidar_range_strategy}, range=[{args.lidar_min_range},{args.lidar_max_range}]m"
    )
    print(
        f"Configuration: backbone={args.model_backbone}; global_train_scales={scale_values}; "
        f"test_cap={args.test_max_samples_per_height}; model_seeds={seed_values}; "
        f"split_seed={args.split_seed}; subset_seed={args.subset_seed}; test_seed={args.test_seed}"
    )

    all_rows = []
    for scale in scale_values:
        args.train_scale_percent = scale
        for seed in seed_values:
            args.seed = seed
            seed_everything(seed)
            for exp in selected_experiments:
                print(
                    f"\n\n================ Experiment: {exp.name} | global_train_pct={scale:g} "
                    f"| test_cap={args.test_max_samples_per_height} | seed={seed} "
                    f"| backbone={args.model_backbone} ================"
                )
                all_rows.extend(run_experiment(exp, args, device))

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.out_dir, "summary_v20_rgb_lidar_global_sampling.csv")
    summary.to_csv(summary_path, index=False)

    show_cols = [
        "experiment", "method", "decode_mode", "control_mode", "model_backbone",
        "sampling_mode", "train_scale_percent", "train_n", "val_n", "test_n",
        "test_scale_per_height", "seed", "split_seed", "subset_seed", "test_seed",
        "mean_px", "median_px", "p90_px", "pck@4", "pck@8", "pck@16",
        "lidar_candidate_rate", "lidar_valid_rate", "lidar_candidate_count_mean",
        "success_at_0p5m", "success_at_1p0m", "success_at_2p0m",
        "range_lidar_mae_m", "range_lidar_p50_m", "range_lidar_p90_m",
        "xyz_lidar_mean_m", "xyz_lidar_p50_m", "xyz_lidar_p90_m",
        "topk_used_lidar_candidate_rate", "topk_selected_rank_mean",
        "topk_valid_candidate_count_mean", "topk_lidar_k", "topk_lidar_nms_kernel",
        "best_epoch", "selection_metric",
    ]
    existing = [column for column in show_cols if column in summary.columns]
    print("\n========== v20 RGB+LiDAR Global-Sampling Summary ==========")
    print(summary[existing].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()

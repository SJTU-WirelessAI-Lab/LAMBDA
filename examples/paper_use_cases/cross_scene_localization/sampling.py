from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import EXPERIMENTS, SCENE_NAMES, ExperimentSpec
from .data import build_scene_index, print_dataset_summary


def get_experiment_by_name(name: str) -> ExperimentSpec:
    for e in EXPERIMENTS:
        if e.name == name:
            return e
    raise ValueError(f"Unknown experiment: {name}")


def _stable_group_seed(seed: int, *parts: str) -> int:
    """Stable deterministic seed independent of Python's randomized hash()."""
    acc = int(seed) & 0x7FFFFFFF
    for part in parts:
        for ch in str(part):
            acc = (acc * 131 + ord(ch)) % 2147483647
    return acc


def get_uv_bin(value: float, img_size: int, n_bins: int) -> int:
    if n_bins <= 1:
        return 0
    b = int(math.floor(float(value) / float(img_size) * n_bins))
    return int(max(0, min(n_bins - 1, b)))


def get_uv_stratum_key(s: Dict[str, Any], args) -> Tuple[str, str, int, int]:
    """Scene + height + v_bin + u_bin.

    v_bin is intentionally placed before u_bin because the observed V19.2
    distribution shift was dominated by vertical image location.
    """
    u_bin = get_uv_bin(float(s["u_img"]), args.img_size, int(args.uv_u_bins))
    v_bin = get_uv_bin(float(s["v_img"]), args.img_size, int(args.uv_v_bins))
    return (str(s["scene"]), str(s["height"]), int(v_bin), int(u_bin))


def split_fixed_val_train_pool_by_height_uv(
    samples: List[Dict[str, Any]],
    val_ratio: float,
    split_seed: int,
    args,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fixed validation split stratified by scene/height/v_bin/u_bin.

    This keeps validation distribution tied to the same joint height and image
    location strata used for nested training subsets. For very small strata, we
    keep all samples in the train pool to avoid destroying rare bins.
    """
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        grouped[get_uv_stratum_key(s, args)].append(s)

    train_pool: List[Dict[str, Any]] = []
    val_out: List[Dict[str, Any]] = []
    for key, hs in sorted(grouped.items()):
        scene, h, v_bin, u_bin = key
        hs = sorted(hs, key=lambda x: int(x["fid_int"]))
        rng = random.Random(_stable_group_seed(split_seed, scene, h, str(v_bin), str(u_bin), "val_split_uv"))
        rng.shuffle(hs)
        n = len(hs)
        if n < int(args.min_stratum_size_for_val):
            train_pool.extend(hs)
            continue
        n_val = int(round(n * val_ratio))
        if n_val <= 0 and val_ratio > 0:
            n_val = 1
        n_val = min(n_val, n - 1)
        val_part = sorted(hs[:n_val], key=lambda x: int(x["fid_int"]))
        train_part = sorted(hs[n_val:], key=lambda x: int(x["fid_int"]))
        train_pool.extend(train_part)
        val_out.extend(val_part)
        print(
            f"  [fixed val split uv {scene} {h} vbin={v_bin} ubin={u_bin}] "
            f"train_pool={len(train_part)} val={len(val_part)} total={n}"
        )
    return train_pool, val_out


def select_nested_train_subset_global_percent(
    train_pool: List[Dict[str, Any]],
    scale_percent: Optional[float],
    subset_seed: int,
    args,
) -> List[Dict[str, Any]]:
    """Select a deterministic nested percentage of the complete train pool.

    For a fixed seed and pool, each smaller percentage is a strict prefix of
    every larger percentage. The fixed validation split remains stratified by
    scene, height, and image location before this global selection is applied.
    """
    ratio = 1.0 if scale_percent is None else float(scale_percent) / 100.0
    ratio = max(0.0, min(1.0, ratio))
    ordered = sorted(
        train_pool,
        key=lambda s: (str(s.get("scene", "")), str(s.get("height", "")), int(s.get("fid_int", 0))),
    )
    random.Random(_stable_group_seed(subset_seed, "nested_subset_global_percent")).shuffle(ordered)
    n_total = len(ordered)
    take_n = n_total if ratio >= 1.0 else int(round(n_total * ratio))
    take_n = max(0, min(take_n, n_total))
    selected = ordered[:take_n]
    random.Random(
        _stable_group_seed(subset_seed, "final_train_shuffle_global_percent", str(scale_percent))
    ).shuffle(selected)
    print(
        f"  [nested global subset] selected={len(selected)}/{n_total} "
        f"target_ratio={ratio:.4f} actual_ratio={len(selected) / max(n_total, 1):.4f}"
    )
    return selected


def _sample_audit_rows(samples: List[Dict[str, Any]], split: str, args, scale_value: Any) -> List[Dict[str, Any]]:
    rows = []
    for s in samples:
        u_bin = get_uv_bin(float(s["u_img"]), args.img_size, int(args.uv_u_bins))
        v_bin = get_uv_bin(float(s["v_img"]), args.img_size, int(args.uv_v_bins))
        rows.append({
            "split": split,
            "sampling_mode": args.scale_unit,
            "scale": scale_value,
            "seed": args.seed,
            "val_seed": args.val_seed,
            "split_seed": args.split_seed,
            "subset_seed": args.subset_seed,
            "test_seed": args.test_seed,
            "scene": s.get("scene", ""),
            "height": s.get("height", ""),
            "v_bin": int(v_bin),
            "u_bin": int(u_bin),
            "fid": int(s["fid_int"]),
            "u_img": float(s["u_img"]),
            "v_img": float(s["v_img"]),
            "range_gt": float(s.get("range_gt", np.nan)),
        })
    return rows


def _audit_summary_for_df(df: pd.DataFrame, split: str, scale_value: Any, args) -> Dict[str, Any]:
    out = {"split": split, "scale": scale_value, "seed": args.seed, "n": int(len(df))}
    if len(df) == 0:
        return out
    for col in ["u_img", "v_img", "range_gt"]:
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if x.size:
            out[f"{col}_min"] = float(np.min(x))
            out[f"{col}_p10"] = float(np.percentile(x, 10))
            out[f"{col}_mean"] = float(np.mean(x))
            out[f"{col}_p50"] = float(np.percentile(x, 50))
            out[f"{col}_p90"] = float(np.percentile(x, 90))
            out[f"{col}_max"] = float(np.max(x))
            out[f"{col}_std"] = float(np.std(x))
    return out


def write_sampling_audit(
    exp: ExperimentSpec,
    args,
    source_samples: List[Dict[str, Any]],
    train_pool: List[Dict[str, Any]],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Write CSV files auditing the v20 global-sampling distributions."""
    scale_value = args.train_scale_percent
    audit_dir = os.path.join(args.out_dir, exp.name, "sampling_audit", f"scale_{scale_value}_seed_{args.seed}_{args.model_backbone}")
    os.makedirs(audit_dir, exist_ok=True)

    rows = []
    rows.extend(_sample_audit_rows(source_samples, "source_full", args, scale_value))
    rows.extend(_sample_audit_rows(train_pool, "train_pool", args, scale_value))
    rows.extend(_sample_audit_rows(train_samples, "train", args, scale_value))
    rows.extend(_sample_audit_rows(val_samples, "val", args, scale_value))
    rows.extend(_sample_audit_rows(test_samples, "test", args, scale_value))
    all_df = pd.DataFrame(rows)
    all_df.to_csv(os.path.join(audit_dir, "sampling_audit_samples.csv"), index=False)

    summary_rows = []
    for split, g in all_df.groupby("split", dropna=False):
        summary_rows.append(_audit_summary_for_df(g, str(split), scale_value, args))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(audit_dir, "sampling_audit_summary.csv"), index=False)

    strata_cols = ["split", "scene", "height", "v_bin", "u_bin"]
    strata_df = all_df.groupby(strata_cols, dropna=False).size().reset_index(name="n")
    strata_df.to_csv(os.path.join(audit_dir, "sampling_audit_strata_counts.csv"), index=False)

    train_df = all_df[all_df["split"] == "train"].copy()
    test_df = all_df[all_df["split"] == "test"].copy()
    compare = {
        "sampling_mode": args.scale_unit,
        "scale": scale_value,
        "seed": args.seed,
        "val_seed": args.val_seed,
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "test_seed": args.test_seed,
        "train_n": int(len(train_df)),
        "val_n": int((all_df["split"] == "val").sum()),
        "test_n": int(len(test_df)),
    }
    if len(train_df) and len(test_df):
        tu_min, tu_max = test_df["u_img"].min(), test_df["u_img"].max()
        tv_min, tv_max = test_df["v_img"].min(), test_df["v_img"].max()
        compare.update({
            "train_u_mean": float(train_df["u_img"].mean()),
            "test_u_mean": float(test_df["u_img"].mean()),
            "abs_u_mean_diff": float(abs(train_df["u_img"].mean() - test_df["u_img"].mean())),
            "train_v_mean": float(train_df["v_img"].mean()),
            "test_v_mean": float(test_df["v_img"].mean()),
            "abs_v_mean_diff": float(abs(train_df["v_img"].mean() - test_df["v_img"].mean())),
            "train_u_p10": float(train_df["u_img"].quantile(0.10)),
            "test_u_p10": float(test_df["u_img"].quantile(0.10)),
            "train_u_p90": float(train_df["u_img"].quantile(0.90)),
            "test_u_p90": float(test_df["u_img"].quantile(0.90)),
            "train_v_p10": float(train_df["v_img"].quantile(0.10)),
            "test_v_p10": float(test_df["v_img"].quantile(0.10)),
            "train_v_p90": float(train_df["v_img"].quantile(0.90)),
            "test_v_p90": float(test_df["v_img"].quantile(0.90)),
            "train_fraction_in_test_u_range": float(((train_df["u_img"] >= tu_min) & (train_df["u_img"] <= tu_max)).mean()),
            "train_fraction_in_test_v_range": float(((train_df["v_img"] >= tv_min) & (train_df["v_img"] <= tv_max)).mean()),
            "train_fraction_in_test_uv_box": float(((train_df["u_img"] >= tu_min) & (train_df["u_img"] <= tu_max) & (train_df["v_img"] >= tv_min) & (train_df["v_img"] <= tv_max)).mean()),
        })
    pd.DataFrame([compare]).to_csv(os.path.join(audit_dir, "sampling_audit_train_vs_test_compare.csv"), index=False)
    print(f"  [sampling audit] saved to {audit_dir}")
    return {"sampling_audit_dir": audit_dir, **compare}


def build_train_val_test_samples(exp: ExperimentSpec, args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Build the full Block 1 source set before applying the fixed validation
    # split. Every scale therefore shares the same validation and test sets.
    source_samples, source_skipped = build_scene_index(
        exp.train_root,
        exp.train_heights,
        args.img_size,
        args.frame_stride,
        None,
        args.seed,
        require_lidar_file=args.require_lidar_file,
    )
    print_dataset_summary(
        f"{exp.name}_source_full_{SCENE_NAMES.get(exp.train_root, os.path.basename(exp.train_root))}",
        source_samples,
        source_skipped,
    )

    test_samples, test_skipped = build_scene_index(exp.test_root, exp.test_heights, args.img_size, args.frame_stride,
                                                   args.test_max_samples_per_height, args.test_seed,
                                                   require_lidar_file=args.require_lidar_file)
    print_dataset_summary(f"{exp.name}_source_full_before_fixed_val_split", source_samples, source_skipped)

    train_pool, val_samples = split_fixed_val_train_pool_by_height_uv(
        source_samples, args.val_ratio, args.split_seed, args
    )
    train_samples = select_nested_train_subset_global_percent(
        train_pool, args.train_scale_percent, args.subset_seed, args
    )

    print_dataset_summary(f"{exp.name}_train_pool_full_after_fixed_val", train_pool, Counter({"after_fixed_val_split": len(train_pool)}))
    print_dataset_summary(f"{exp.name}_train_nested_global_pct_{args.train_scale_percent}", train_samples, Counter({"nested_subset_from_train_pool": len(train_samples)}))
    print_dataset_summary(f"{exp.name}_val_fixed", val_samples, Counter({"fixed_val_split": len(val_samples)}))
    print_dataset_summary(f"{exp.name}_test_fixed", test_samples, test_skipped)

    args._last_sampling_audit = write_sampling_audit(exp, args, source_samples, train_pool, train_samples, val_samples, test_samples)
    return train_samples, val_samples, test_samples

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_pixel_metrics(errors: np.ndarray, prefix: str = "") -> Dict[str, float]:
    vals = np.array(errors, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        out = {f"{prefix}mean_px": float("nan"), f"{prefix}median_px": float("nan"), f"{prefix}p90_px": float("nan")}
        for k in [2, 4, 8, 16, 32]:
            out[f"{prefix}pck@{k}"] = float("nan")
        return out
    out = {
        f"{prefix}mean_px": float(np.mean(vals)),
        f"{prefix}median_px": float(np.median(vals)),
        f"{prefix}p90_px": float(np.percentile(vals, 90)),
        f"{prefix}p95_px": float(np.percentile(vals, 95)),
        f"{prefix}max_px": float(np.max(vals)),
    }
    for k in [2, 4, 8, 16, 32]:
        out[f"{prefix}pck@{k}"] = float(np.mean(vals <= k))
    return out


def add_stats(metrics: Dict[str, float], name: str, values: List[float]) -> None:
    vals = np.array([v for v in values if np.isfinite(v)], dtype=np.float32)
    if vals.size == 0:
        metrics[f"{name}_mean"] = float("nan")
        metrics[f"{name}_p50"] = float("nan")
        metrics[f"{name}_p90"] = float("nan")
        return
    metrics[f"{name}_mean"] = float(np.mean(vals))
    metrics[f"{name}_p50"] = float(np.percentile(vals, 50))
    metrics[f"{name}_p90"] = float(np.percentile(vals, 90))
    metrics[f"{name}_p95"] = float(np.percentile(vals, 95))
    metrics[f"{name}_max"] = float(np.max(vals))


def baseline_pixel_results(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]], by_height: bool) -> Tuple[Dict[str, float], pd.DataFrame]:
    global_uv = np.array([[s["u_img"], s["v_img"]] for s in train_samples], dtype=np.float32).mean(axis=0)
    height_uv: Dict[str, np.ndarray] = {}
    for h in sorted(set(s["height"] for s in train_samples)):
        arr = np.array([[s["u_img"], s["v_img"]] for s in train_samples if s["height"] == h], dtype=np.float32)
        height_uv[h] = arr.mean(axis=0)
    rows = []
    errs = []
    for s in test_samples:
        pred = height_uv.get(s["height"], global_uv) if by_height else global_uv
        err = float(np.linalg.norm(pred - np.array([s["u_img"], s["v_img"]], dtype=np.float32)))
        errs.append(err)
        rows.append({
            "scene": s["scene"], "fid": s["fid"], "height": s["height"],
            "u_gt_img": s["u_img"], "v_gt_img": s["v_img"],
            "u_pred_img": float(pred[0]), "v_pred_img": float(pred[1]),
            "center_error_px_512": err,
        })
    return compute_pixel_metrics(np.array(errs, dtype=np.float32)), pd.DataFrame(rows)

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from .config import BLOCK_ROOT, CROP_SIZE, CROP_U_MAX, CROP_U_MIN, IMG_H, SQUARE_ROOT
from .decoding import decode_heatmaps_local, decode_heatmaps_topk_local
from .geometry import img_to_raw_uv, load_camera_pose, ray_range_to_world
from .lidar import associate_lidar_range
from .metrics import add_stats, compute_pixel_metrics


def _evaluate_lidar_for_metas(
    metas: List[Dict[str, Any]],
    u_query_img: np.ndarray,
    v_query_img: np.ndarray,
    conf: np.ndarray,
    img_size: int,
    args,
    source: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    center_errs = []
    candidate_flags = []
    valid_flags = []
    candidate_counts = []
    xyz_lidar_errs = []
    range_lidar_errs = []
    xyz_pred2d_gt_range_errs = []
    selected_pixel_dists = []
    overall_success_thresholds = [0.5, 1.0, 2.0]
    overall_success_counts = {thr: 0 for thr in overall_success_thresholds}

    cam_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for i, m in enumerate(tqdm(metas, desc=f"lidar-{source}", leave=False)):
        scene_root = m["scene_root"]
        if scene_root not in cam_cache:
            cam_cache[scene_root] = load_camera_pose(scene_root)
        cam_world, R_cam = cam_cache[scene_root]

        u_pred_img = float(u_query_img[i])
        v_pred_img = float(v_query_img[i])
        u_pred_raw, v_pred_raw = img_to_raw_uv(u_pred_img, v_pred_img, img_size)
        center_err = float(np.linalg.norm([u_pred_img - float(m["u_gt_img"]), v_pred_img - float(m["v_gt_img"])]))
        center_errs.append(center_err)

        gt_range = float(m["range_gt"])
        xyz_pred2d_gt_range = ray_range_to_world(u_pred_raw, v_pred_raw, gt_range, cam_world, R_cam)
        xyz_gt_world = np.array(m["uav_world"], dtype=np.float32)
        xyz_err_2d = float(np.linalg.norm(xyz_pred2d_gt_range - xyz_gt_world))
        xyz_pred2d_gt_range_errs.append(xyz_err_2d)

        assoc = {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "u_lidar_raw": np.nan, "v_lidar_raw": np.nan}
        lidar_path = m.get("lidar_path")
        if lidar_path and os.path.exists(lidar_path):
            try:
                assoc = associate_lidar_range(
                    scene_root=scene_root,
                    lidar_path=lidar_path,
                    u_raw=u_pred_raw,
                    v_raw=v_pred_raw,
                    cam_world=cam_world,
                    R_cam_ned=R_cam,
                    pixel_radius=args.lidar_pixel_radius,
                    min_range=args.lidar_min_range,
                    max_range=args.lidar_max_range,
                    strategy=args.lidar_range_strategy,
                )
            except Exception as e:
                assoc = {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "u_lidar_raw": np.nan, "v_lidar_raw": np.nan, "error": repr(e)}

        candidate = int(assoc.get("candidate_count", 0)) > 0
        valid = bool(assoc.get("valid", False))
        candidate_flags.append(candidate)
        valid_flags.append(valid)
        candidate_counts.append(int(assoc.get("candidate_count", 0)))

        xyz_err_lidar = float("nan")
        range_err_lidar = float("nan")
        if valid:
            range_lidar = float(assoc["range_lidar"])
            xyz_lidar = ray_range_to_world(u_pred_raw, v_pred_raw, range_lidar, cam_world, R_cam)
            xyz_err_lidar = float(np.linalg.norm(xyz_lidar - xyz_gt_world))
            range_err_lidar = abs(range_lidar - gt_range)
            xyz_lidar_errs.append(xyz_err_lidar)
            range_lidar_errs.append(range_err_lidar)
            for _thr in overall_success_thresholds:
                if xyz_err_lidar <= _thr:
                    overall_success_counts[_thr] += 1
            if np.isfinite(assoc.get("selected_pixel_dist_raw", np.nan)):
                selected_pixel_dists.append(float(assoc["selected_pixel_dist_raw"]))

        rows.append({
            "source": source,
            "scene": m["scene"],
            "scene_root": scene_root,
            "fid": m["fid"],
            "height": m["height"],
            "u_gt_img": float(m["u_gt_img"]),
            "v_gt_img": float(m["v_gt_img"]),
            "u_pred_img": u_pred_img,
            "v_pred_img": v_pred_img,
            "u_gt_raw": float(m["u_gt_raw"]),
            "v_gt_raw": float(m["v_gt_raw"]),
            "u_pred_raw": u_pred_raw,
            "v_pred_raw": v_pred_raw,
            "center_error_px_512": center_err,
            "center_error_px_raw": center_err * (CROP_SIZE / img_size),
            "peak_confidence": float(conf[i]) if len(conf) else float("nan"),
            "range_gt": gt_range,
            "lidar_path": lidar_path,
            "lidar_candidate": candidate,
            "lidar_valid": valid,
            "lidar_candidate_count": int(assoc.get("candidate_count", 0)),
            "range_lidar": float(assoc.get("range_lidar", np.nan)),
            "range_error_lidar_m": range_err_lidar,
            "u_lidar_raw": float(assoc.get("u_lidar_raw", np.nan)),
            "v_lidar_raw": float(assoc.get("v_lidar_raw", np.nan)),
            "selected_pixel_dist_raw": float(assoc.get("selected_pixel_dist_raw", np.nan)),
            "xyz_error_pred2d_gt_range_m": xyz_err_2d,
            "xyz_error_pred2d_lidar_range_m": xyz_err_lidar,
            "association_error": assoc.get("error", ""),
        })

    metrics = compute_pixel_metrics(np.array(center_errs, dtype=np.float32))
    n = max(len(metas), 1)
    metrics["lidar_candidate_rate"] = float(np.sum(candidate_flags) / n)
    metrics["lidar_valid_rate"] = float(np.sum(valid_flags) / n)
    metrics["lidar_candidate_count_mean"] = float(np.mean(candidate_counts)) if candidate_counts else float("nan")
    # Overall success counts invalid LiDAR association as failure over all test samples.
    for _thr in overall_success_thresholds:
        key = str(_thr).replace(".", "p")
        metrics[f"success_at_{key}m"] = float(overall_success_counts[_thr] / n)
    add_stats(metrics, "xyz_error_pred2d_gt_range_m", xyz_pred2d_gt_range_errs)
    add_stats(metrics, "xyz_error_pred2d_lidar_range_m", xyz_lidar_errs)
    add_stats(metrics, "range_error_lidar_m", range_lidar_errs)
    add_stats(metrics, "selected_lidar_pixel_dist_raw", selected_pixel_dists)

    # Friendly aliases for summary tables.
    metrics["xyz_lidar_mean_m"] = metrics.get("xyz_error_pred2d_lidar_range_m_mean", float("nan"))
    metrics["xyz_lidar_p50_m"] = metrics.get("xyz_error_pred2d_lidar_range_m_p50", float("nan"))
    metrics["xyz_lidar_p90_m"] = metrics.get("xyz_error_pred2d_lidar_range_m_p90", float("nan"))
    metrics["range_lidar_mae_m"] = metrics.get("range_error_lidar_m_mean", float("nan"))
    metrics["range_lidar_p50_m"] = metrics.get("range_error_lidar_m_p50", float("nan"))
    metrics["range_lidar_p90_m"] = metrics.get("range_error_lidar_m_p90", float("nan"))
    return metrics, pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model_with_lidar(model, loader, device, img_size: int, args, source: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    all_metas = []
    all_u = []
    all_v = []
    all_conf = []
    for x, _, metas in tqdm(loader, desc="eval-model", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        u_pred, v_pred, conf = decode_heatmaps_local(logits, img_size)
        all_metas.extend(metas)
        all_u.extend(list(u_pred))
        all_v.extend(list(v_pred))
        all_conf.extend(list(conf))
    return _evaluate_lidar_for_metas(all_metas, np.array(all_u, dtype=np.float32), np.array(all_v, dtype=np.float32), np.array(all_conf, dtype=np.float32), img_size, args, source)


@torch.no_grad()
def evaluate_model_with_lidar_topk_lidar(model, loader, device, img_size: int, args, source: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate deployable top-k LiDAR reranking.

    Rule:
      1. Extract heatmap top-k local peaks.
      2. Check candidates in heatmap-confidence order.
      3. Select the first LiDAR-valid candidate.
      4. If no top-k candidate is LiDAR-valid, fall back to rank-1 argmax.

    This uses no ground-truth information during prediction.
    """
    model.eval()
    all_metas: List[Dict[str, Any]] = []
    all_u: List[float] = []
    all_v: List[float] = []
    all_conf: List[float] = []
    selected_ranks: List[int] = []
    used_topk_lidar_flags: List[bool] = []
    topk_valid_counts: List[int] = []
    topk_candidate_rows: List[Dict[str, Any]] = []

    cam_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for x, _, metas in tqdm(loader, desc="eval-model-topk-lidar", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        topk_batch = decode_heatmaps_topk_local(
            logits,
            img_size=img_size,
            topk=args.topk_lidar_k,
            nms_kernel=args.topk_lidar_nms_kernel,
        )

        for m, peaks in zip(metas, topk_batch):
            scene_root = m["scene_root"]
            if scene_root not in cam_cache:
                cam_cache[scene_root] = load_camera_pose(scene_root)
            cam_world, R_cam = cam_cache[scene_root]

            lidar_path = m.get("lidar_path")
            first_valid_peak: Optional[Dict[str, float]] = None
            valid_count = 0

            for p in peaks:
                assoc = {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "selected_pixel_dist_raw": np.nan}
                if lidar_path and os.path.exists(lidar_path):
                    try:
                        u_raw, v_raw = img_to_raw_uv(float(p["u_img"]), float(p["v_img"]), img_size)
                        assoc = associate_lidar_range(
                            scene_root=scene_root,
                            lidar_path=lidar_path,
                            u_raw=u_raw,
                            v_raw=v_raw,
                            cam_world=cam_world,
                            R_cam_ned=R_cam,
                            pixel_radius=args.lidar_pixel_radius,
                            min_range=args.lidar_min_range,
                            max_range=args.lidar_max_range,
                            strategy=args.lidar_range_strategy,
                        )
                    except Exception as e:
                        assoc = {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "error": repr(e)}

                is_valid = bool(assoc.get("valid", False))
                if is_valid:
                    valid_count += 1
                    if first_valid_peak is None:
                        first_valid_peak = p

                topk_candidate_rows.append({
                    "source": source,
                    "scene": m["scene"],
                    "fid": m["fid"],
                    "height": m["height"],
                    "rank": int(p["rank"]),
                    "u_gt_img": float(m["u_gt_img"]),
                    "v_gt_img": float(m["v_gt_img"]),
                    "u_pred_img": float(p["u_img"]),
                    "v_pred_img": float(p["v_img"]),
                    "center_error_px_512": float(np.linalg.norm([float(p["u_img"]) - float(m["u_gt_img"]), float(p["v_img"]) - float(m["v_gt_img"])])),
                    "peak_confidence": float(p["confidence"]),
                    "topk_lidar_candidate": int(assoc.get("candidate_count", 0)) > 0,
                    "topk_lidar_valid": is_valid,
                    "topk_lidar_candidate_count": int(assoc.get("candidate_count", 0)),
                    "topk_selected_pixel_dist_raw": float(assoc.get("selected_pixel_dist_raw", np.nan)),
                })

            if first_valid_peak is not None:
                chosen = first_valid_peak
                used = True
            else:
                chosen = peaks[0]
                used = False

            all_metas.append(m)
            all_u.append(float(chosen["u_img"]))
            all_v.append(float(chosen["v_img"]))
            all_conf.append(float(chosen["confidence"]))
            selected_ranks.append(int(chosen["rank"]))
            used_topk_lidar_flags.append(bool(used))
            topk_valid_counts.append(int(valid_count))

    metrics, details = _evaluate_lidar_for_metas(
        all_metas,
        np.array(all_u, dtype=np.float32),
        np.array(all_v, dtype=np.float32),
        np.array(all_conf, dtype=np.float32),
        img_size,
        args,
        source,
    )

    details["topk_selected_rank"] = selected_ranks
    details["topk_used_lidar_candidate"] = used_topk_lidar_flags
    details["topk_valid_candidate_count"] = topk_valid_counts
    metrics["topk_used_lidar_candidate_rate"] = float(np.mean(used_topk_lidar_flags)) if used_topk_lidar_flags else float("nan")
    metrics["topk_selected_rank_mean"] = float(np.mean(selected_ranks)) if selected_ranks else float("nan")
    metrics["topk_valid_candidate_count_mean"] = float(np.mean(topk_valid_counts)) if topk_valid_counts else float("nan")

    # Store candidate-level diagnostics as an attribute. Caller can persist it.
    details.attrs["topk_candidates"] = pd.DataFrame(topk_candidate_rows)
    return metrics, details


def evaluate_gt2d_lidar(samples: List[Dict[str, Any]], img_size: int, args, source: str = "gt2d_lidar") -> Tuple[Dict[str, float], pd.DataFrame]:
    metas = []
    for s in samples:
        metas.append({
            "scene_root": s["scene_root"], "scene": s["scene"], "fid": s["fid"], "fid_int": s["fid_int"],
            "height": s["height"], "rgb_path": s["rgb_path"], "lidar_path": s.get("lidar_path"),
            "u_gt_img": float(s["u_img"]), "v_gt_img": float(s["v_img"]),
            "u_gt_raw": float(s["u_raw"]), "v_gt_raw": float(s["v_raw"]),
            "range_gt": float(s["range_gt"]), "xyz_cv": s["xyz_cv"].astype(np.float32),
            "uav_world": s["uav_world"].astype(np.float32),
        })
    u = np.array([m["u_gt_img"] for m in metas], dtype=np.float32)
    v = np.array([m["v_gt_img"] for m in metas], dtype=np.float32)
    conf = np.ones(len(metas), dtype=np.float32)
    return _evaluate_lidar_for_metas(metas, u, v, conf, img_size, args, source)


def make_prediction_montage(details: pd.DataFrame, out_path: str, max_images: int = 60, tile_w: int = 320) -> None:
    if details.empty:
        return
    df = details.copy()
    if len(df) > max_images:
        df = df.sample(max_images, random_state=2026)
    tiles = []
    for _, r in df.iterrows():
        scene = str(r["scene"])
        if scene == "Block_1":
            root = BLOCK_ROOT
        else:
            root = SQUARE_ROOT
        fid = str(r["fid"]).zfill(6)
        img_path = os.path.join(root, "cam", f"img_{fid}.png")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB").crop((CROP_U_MIN, 0, CROP_U_MAX, IMG_H)).resize((tile_w, tile_w))
        draw = ImageDraw.Draw(img)
        scale_img = tile_w / 512.0
        scale_raw = tile_w / CROP_SIZE
        ug, vg = float(r["u_gt_img"]) * scale_img, float(r["v_gt_img"]) * scale_img
        up, vp = float(r["u_pred_img"]) * scale_img, float(r["v_pred_img"]) * scale_img
        draw.ellipse((ug - 4, vg - 4, ug + 4, vg + 4), outline=(0, 255, 0), width=2)
        draw.line((up - 6, vp, up + 6, vp), fill=(255, 0, 0), width=2)
        draw.line((up, vp - 6, up, vp + 6), fill=(255, 0, 0), width=2)
        if np.isfinite(float(r.get("u_lidar_raw", np.nan))) and np.isfinite(float(r.get("v_lidar_raw", np.nan))):
            ul = (float(r["u_lidar_raw"]) - CROP_U_MIN) * scale_raw
            vl = float(r["v_lidar_raw"]) * scale_raw
            draw.line((ul - 6, vl, ul + 6, vl), fill=(0, 160, 255), width=2)
            draw.line((ul, vl - 6, ul, vl + 6), fill=(0, 160, 255), width=2)
        txt = f"{scene} {r['height']} {fid} 2D={float(r['center_error_px_512']):.1f}px Lvalid={int(bool(r.get('lidar_valid', False)))}"
        draw.rectangle((0, 0, tile_w, 24), fill=(0, 0, 0))
        draw.text((4, 4), txt, fill=(255, 255, 255))
        tiles.append(img)
    if not tiles:
        return
    cols = 5
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_w), (20, 20, 20))
    for i, im in enumerate(tiles):
        canvas.paste(im, ((i % cols) * tile_w, (i // cols) * tile_w))
    canvas.save(out_path)

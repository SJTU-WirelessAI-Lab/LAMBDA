from __future__ import annotations

import bisect
import glob
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import CX, CY, FX, FY, IMG_H, IMG_W
from .geometry import pcd_to_world, world_to_camcv


@lru_cache(maxsize=4)
def read_pcd_xyz_cached(path_str: str) -> np.ndarray:
    path = Path(path_str)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"No DATA line in PCD: {path}")
            header_lines.append(line)
            if line.strip().lower().startswith(b"data"):
                break
        payload = f.read()
    header = {}
    for raw in header_lines:
        s = raw.decode("ascii", errors="ignore").strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        header[parts[0].upper()] = parts[1:]
    fields = header.get("FIELDS", [])
    npoints = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    data_type = header.get("DATA", [""])[0].lower()
    if fields[:3] != ["x", "y", "z"] or data_type != "binary":
        raise ValueError(f"Only binary x/y/z float32 PCD supported: {path}, fields={fields}, data={data_type}")
    pts = np.frombuffer(payload[:npoints * 12], dtype=np.float32).reshape(npoints, 3).copy()
    return pts[np.isfinite(pts).all(axis=1)]


def lidar_file_for_fid(scene_root: str, fid: int) -> Optional[str]:
    p = os.path.join(scene_root, "lidar", f"lidar_{fid:06d}.pcd")
    return p if os.path.exists(p) else None


def associate_lidar_range(
    scene_root: str,
    lidar_path: str,
    u_raw: float,
    v_raw: float,
    cam_world: np.ndarray,
    R_cam_ned: np.ndarray,
    pixel_radius: float,
    min_range: float,
    max_range: float,
    strategy: str,
) -> Dict[str, Any]:
    """Associate projected LiDAR points around a raw-pixel query point.

    Returns candidate rate information and selected range. Range is measured from
    RoofCam origin along the OpenCV ray, consistent with AirSim DepthPerspective.
    """
    pts_local = read_pcd_xyz_cached(lidar_path)
    pts_world = pcd_to_world(pts_local, scene_root)
    xyz_cv = world_to_camcv(pts_world, cam_world, R_cam_ned)
    z = xyz_cv[:, 2]
    valid_z = z > 1e-6
    ranges = np.linalg.norm(xyz_cv, axis=1).astype(np.float32)
    valid_range = (ranges >= min_range) & (ranges <= max_range)
    valid = valid_z & valid_range
    if not np.any(valid):
        return {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "u_lidar_raw": np.nan, "v_lidar_raw": np.nan}

    idx_valid = np.nonzero(valid)[0]
    xyzv = xyz_cv[idx_valid]
    rv = ranges[idx_valid]
    u = FX * xyzv[:, 0] / xyzv[:, 2] + CX
    v = FY * xyzv[:, 1] / xyzv[:, 2] + CY
    in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    if not np.any(in_img):
        return {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "u_lidar_raw": np.nan, "v_lidar_raw": np.nan}
    idx2 = idx_valid[in_img]
    u2 = u[in_img]
    v2 = v[in_img]
    r2 = rv[in_img]
    pix_dist = np.sqrt((u2 - float(u_raw)) ** 2 + (v2 - float(v_raw)) ** 2)
    cand = pix_dist <= float(pixel_radius)
    if not np.any(cand):
        return {"valid": False, "candidate_count": 0, "range_lidar": np.nan, "u_lidar_raw": np.nan, "v_lidar_raw": np.nan}

    cand_indices = np.nonzero(cand)[0]
    cr = r2[cand]
    cu = u2[cand]
    cv = v2[cand]
    cpix = pix_dist[cand]

    if strategy == "minrange":
        j = int(np.argmin(cr))
        selected_range = float(cr[j])
    elif strategy == "p10range":
        selected_range = float(np.percentile(cr, 10))
        j = int(np.argmin(np.abs(cr - selected_range)))
    elif strategy == "median":
        selected_range = float(np.median(cr))
        j = int(np.argmin(np.abs(cr - selected_range)))
    elif strategy == "nearest_pixel":
        j = int(np.argmin(cpix))
        selected_range = float(cr[j])
    else:
        raise ValueError(f"Unknown lidar range strategy: {strategy}")

    return {
        "valid": True,
        "candidate_count": int(len(cand_indices)),
        "range_lidar": selected_range,
        "u_lidar_raw": float(cu[j]),
        "v_lidar_raw": float(cv[j]),
        "selected_pixel_dist_raw": float(cpix[j]),
    }

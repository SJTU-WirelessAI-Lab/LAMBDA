from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .config import (
    CROP_SIZE,
    CROP_U_MAX,
    CROP_U_MIN,
    CROP_V_MAX,
    CROP_V_MIN,
    CX,
    CY,
    FX,
    FY,
    IMG_H,
    IMG_W,
    SCENE_CAM_PARAMS,
    SCENE_LIDAR_PARAMS,
    SCENE_LIDAR_SETTINGS_WORLD,
    SCENE_LIDAR_TRANSFORM_MODE,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def euler_to_R_ned(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y),  math.cos(y), 0],
                   [0,            0,           1]], dtype=np.float32)
    Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                   [0,           1, 0],
                   [-math.sin(p), 0, math.cos(p)]], dtype=np.float32)
    Rx = np.array([[1, 0,           0],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r),  math.cos(r)]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def world_to_ned(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], -p[1], -p[2]], dtype=np.float32)


def build_R_ned(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    return euler_to_R_ned(pitch_deg, yaw_deg, roll_deg)


def _load_sensor_pose(scene_root: str, sensor_name: str) -> Tuple[np.ndarray, dict]:
    pose_path = os.path.join(scene_root, "sensors", f"{sensor_name}_world_pose.json")
    with open(pose_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("sensor_name") != sensor_name:
        raise ValueError(
            f"Expected sensor_name={sensor_name!r} in {pose_path}, "
            f"got {data.get('sensor_name')!r}."
        )
    pos = data["world_transform"]["position"]
    orientation = data["world_transform"]["orientation"]
    return np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32), orientation


def load_sensor_position(scene_root: str, sensor_name: str) -> np.ndarray:
    position, _ = _load_sensor_pose(scene_root, sensor_name)
    return position


def _quaternion_to_rotation(orientation: dict) -> np.ndarray:
    q = np.array(
        [orientation["w"], orientation["x"], orientation["y"], orientation["z"]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(q))
    if norm == 0:
        raise ValueError("Sensor orientation quaternion has zero norm.")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _sensor_pose_rotation(params: dict) -> np.ndarray:
    """Rotation order used by the released AirSim sensor-pose JSON files."""
    pitch = math.radians(float(params["pitch"]))
    yaw = math.radians(float(params["yaw"]))
    roll = math.radians(float(params.get("roll", 0.0)))
    ry = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]],
        dtype=np.float64,
    )
    return ry @ rz @ rx


def _rotation_error_degrees(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = (float(np.trace(expected.T @ actual)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def validate_sensor_configuration(
    scene_roots: Sequence[str],
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_deg: float = 0.05,
    expected_image_size: Tuple[int, int] = (IMG_W, IMG_H),
) -> None:
    """Fail before training if prepared data no longer match audited sensor constants."""
    from PIL import Image

    errors = []
    for scene_root in dict.fromkeys(scene_roots):
        try:
            cam_position, cam_orientation = _load_sensor_pose(scene_root, "RoofCam")
            lidar_position, lidar_orientation = _load_sensor_pose(scene_root, "RoofLidar")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{scene_root}: cannot read sensor poses: {exc}")
            continue

        expected_lidar_position = SCENE_LIDAR_SETTINGS_WORLD[scene_root]
        lidar_position_error = float(np.linalg.norm(lidar_position - expected_lidar_position))
        if lidar_position_error > position_tolerance_m:
            errors.append(
                f"{scene_root}: RoofLidar position differs from the audited setting by "
                f"{lidar_position_error:.6f} m"
            )

        colocation_error = float(np.linalg.norm(cam_position - lidar_position))
        if colocation_error > position_tolerance_m:
            errors.append(
                f"{scene_root}: RoofCam and RoofLidar are not co-located "
                f"({colocation_error:.6f} m apart)"
            )

        cam_rotation_error = _rotation_error_degrees(
            _quaternion_to_rotation(cam_orientation),
            _sensor_pose_rotation(SCENE_CAM_PARAMS[scene_root]),
        )
        if cam_rotation_error > orientation_tolerance_deg:
            errors.append(
                f"{scene_root}: RoofCam orientation differs from the audited setting by "
                f"{cam_rotation_error:.6f} deg"
            )

        lidar_rotation_error = _rotation_error_degrees(
            _quaternion_to_rotation(lidar_orientation),
            _sensor_pose_rotation(SCENE_LIDAR_PARAMS[scene_root]),
        )
        if lidar_rotation_error > orientation_tolerance_deg:
            errors.append(
                f"{scene_root}: RoofLidar orientation differs from the audited setting by "
                f"{lidar_rotation_error:.6f} deg"
            )

        image_path = next(Path(scene_root, "cam").glob("img_*.png"), None)
        if image_path is None:
            errors.append(f"{scene_root}: no extracted cam/img_*.png frame found")
        else:
            with Image.open(image_path) as image:
                if image.size != expected_image_size:
                    errors.append(
                        f"{scene_root}: {image_path.name} has size {image.size}, "
                        f"expected {expected_image_size}"
                    )

    if errors:
        raise RuntimeError("Sensor configuration check failed:\n- " + "\n- ".join(errors))
    print(f"Sensor configuration check: PASS ({len(dict.fromkeys(scene_roots))} scenes)")


def load_camera_pose(scene_root: str) -> Tuple[np.ndarray, np.ndarray]:
    cam_world = load_sensor_position(scene_root, "RoofCam")
    cfg = SCENE_CAM_PARAMS[scene_root]
    R_ned = build_R_ned(cfg["pitch"], cfg["yaw"], cfg.get("roll", 0.0))
    return cam_world, R_ned


def load_lidar_pose_from_settings(scene_root: str) -> Tuple[np.ndarray, np.ndarray, str]:
    lidar_world = SCENE_LIDAR_SETTINGS_WORLD[scene_root]
    cfg = SCENE_LIDAR_PARAMS[scene_root]
    R_ned = build_R_ned(cfg["pitch"], cfg["yaw"], cfg.get("roll", 0.0))
    mode = SCENE_LIDAR_TRANSFORM_MODE[scene_root]
    return lidar_world, R_ned, mode


def world_to_camcv(points_world: np.ndarray, cam_world: np.ndarray, R_ned: np.ndarray) -> np.ndarray:
    """World coordinates -> OpenCV camera coordinates.

    OpenCV camera coordinates: x right, y down, z forward.
    """
    pts = np.asarray(points_world, dtype=np.float32)
    one = pts.ndim == 1
    pts2 = pts[None, :] if one else pts
    pts_ned = np.stack([pts2[:, 0], -pts2[:, 1], -pts2[:, 2]], axis=1).astype(np.float32)
    cam_ned = world_to_ned(cam_world)
    diff_ned = pts_ned - cam_ned[None, :]
    # row-vector equivalent of R_ned.T @ diff used in V16
    cc = diff_ned @ R_ned
    xyz_cv = np.stack([cc[:, 1], cc[:, 2], cc[:, 0]], axis=1).astype(np.float32)
    return xyz_cv[0] if one else xyz_cv


def camcv_to_world(xyz_cv: np.ndarray, cam_world: np.ndarray, R_ned: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_cv, dtype=np.float32)
    x_cv, y_cv, z_cv = xyz[0], xyz[1], xyz[2]
    cc = np.array([z_cv, x_cv, y_cv], dtype=np.float32)
    cam_ned = world_to_ned(cam_world)
    # inverse of cc_row = diff_ned @ R_ned
    diff_ned = cc @ R_ned.T
    p_ned = cam_ned + diff_ned
    return np.array([p_ned[0], -p_ned[1], -p_ned[2]], dtype=np.float32)


def project_to_pixel_raw(xyz_cv: np.ndarray, require_crop_visible: bool = True) -> Optional[Tuple[float, float]]:
    if float(xyz_cv[2]) < 1.0:
        return None
    u = FX * float(xyz_cv[0]) / float(xyz_cv[2]) + CX
    v = FY * float(xyz_cv[1]) / float(xyz_cv[2]) + CY
    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        return None
    if require_crop_visible and not (CROP_U_MIN <= u < CROP_U_MAX and CROP_V_MIN <= v < CROP_V_MAX):
        return None
    return float(u), float(v)


def raw_uv_to_img(u_raw: float, v_raw: float, img_size: int) -> Tuple[float, float]:
    u_crop = u_raw - CROP_U_MIN
    v_crop = v_raw
    return float(u_crop / CROP_SIZE * img_size), float(v_crop / CROP_SIZE * img_size)


def img_to_raw_uv(u_img: float, v_img: float, img_size: int) -> Tuple[float, float]:
    u_raw = CROP_U_MIN + u_img / img_size * CROP_SIZE
    v_raw = v_img / img_size * CROP_SIZE
    return float(u_raw), float(v_raw)


def raw_pixel_to_camcv_ray(u_raw: float, v_raw: float) -> np.ndarray:
    ray = np.array([(u_raw - CX) / FX, (v_raw - CY) / FY, 1.0], dtype=np.float32)
    return ray / (np.linalg.norm(ray) + 1e-9)


def ray_range_to_world(u_raw: float, v_raw: float, range_m: float, cam_world: np.ndarray, R_cam_ned: np.ndarray) -> np.ndarray:
    ray = raw_pixel_to_camcv_ray(u_raw, v_raw)
    xyz_cv = ray * float(range_m)
    return camcv_to_world(xyz_cv, cam_world, R_cam_ned)


def pcd_to_world(points_local: np.ndarray, scene_root: str) -> np.ndarray:
    lidar_world, R_lidar_ned, mode = load_lidar_pose_from_settings(scene_root)
    pts = np.asarray(points_local, dtype=np.float32)
    base = lidar_world.astype(np.float32)
    if mode == "settings_local_world_identity":
        return base[None, :] + pts
    if mode == "settings_local_world_yflip_zflip":
        return base[None, :] + np.stack([pts[:, 0], -pts[:, 1], -pts[:, 2]], axis=1).astype(np.float32)
    if mode == "settings_local_ned_identity":
        base_ned = world_to_ned(base)
        p_ned = base_ned[None, :] + pts
        return np.stack([p_ned[:, 0], -p_ned[:, 1], -p_ned[:, 2]], axis=1).astype(np.float32)
    if mode == "settings_local_ned_euler":
        base_ned = world_to_ned(base)
        p_ned = base_ned[None, :] + pts @ R_lidar_ned.T
        return np.stack([p_ned[:, 0], -p_ned[:, 1], -p_ned[:, 2]], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported LiDAR transform mode for {scene_root}: {mode}")

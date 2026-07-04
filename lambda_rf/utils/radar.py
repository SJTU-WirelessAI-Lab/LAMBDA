from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lambda_rf.utils.array_csi import (
    ARRAY_MODEL_FAR_FIELD,
    ARRAY_MODEL_SPHERICAL,
    normalize_array_model,
    quaternion_xyzw_to_rotation_matrix,
)


C_M_S = 299_792_458.0


def parse_frame_index(path: str | Path, prefix: str = "csi") -> int | None:
    name = Path(path).name
    match = re.search(rf"{re.escape(prefix)}_(\d+)", name)
    if not match:
        match = re.search(r"(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def parse_shape(value: str | tuple[int, int] | list[int], name: str = "shape") -> tuple[int, int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"{name} must be ROWS,COLS or ROWSxCOLS, got {value!r}")
    rows, cols = int(parts[0]), int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name} values must be positive, got {value!r}")
    return rows, cols


def virtual_array_positions(
    carrier_frequency_hz: float,
    shape: tuple[int, int] | list[int] = (4, 4),
    spacing_wavelengths: float = 0.5,
) -> np.ndarray:
    rows, cols = parse_shape(shape, "array_shape")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be > 0")
    if spacing_wavelengths <= 0.0:
        raise ValueError("spacing_wavelengths must be > 0")

    wavelength = C_M_S / float(carrier_frequency_hz)
    spacing = wavelength * float(spacing_wavelengths)
    y_range = np.arange(cols, dtype=np.float64) * spacing
    z_range = np.arange(rows, dtype=np.float64) * spacing
    y_range -= np.mean(y_range)
    z_range -= np.mean(z_range)
    y_grid, z_grid = np.meshgrid(y_range, z_range)
    x_grid = np.zeros_like(y_grid)
    return np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)


def rotation_matrix_zyx_degrees(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a local-to-world matrix using ZYX yaw/pitch/roll angles."""
    y, p, r = np.radians([yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _orientation_to_rotation(data: dict[str, Any], source: Path) -> np.ndarray:
    orientation = None
    if isinstance(data.get("world_transform"), dict):
        orientation = data["world_transform"].get("orientation")
    if orientation is None:
        orientation = data.get("orientation")
    if orientation is None and isinstance(data.get("rotation"), dict):
        orientation = data["rotation"].get("orientation") or data["rotation"].get("quaternion")
    if orientation is None:
        raise KeyError(f"No supported orientation quaternion found in {source}")

    if isinstance(orientation, dict):
        quat = np.asarray(
            [
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
                float(orientation["w"]),
            ],
            dtype=np.float64,
        )
    else:
        quat = np.asarray(orientation, dtype=np.float64).reshape(-1)
    return quaternion_xyzw_to_rotation_matrix(quat)


class IMURotationLoader:
    """Load per-frame body-to-world rotations from `imu_*.json` files."""

    def __init__(self, imu_dir: str | Path | None):
        self.imu_dir = Path(imu_dir).expanduser().resolve() if imu_dir else None
        self.rotations: list[np.ndarray] = []
        if self.imu_dir is None:
            return
        files = sorted(self.imu_dir.glob("imu_*.json"))
        if not files:
            raise FileNotFoundError(f"No imu_*.json files found in {self.imu_dir}")
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                self.rotations.append(_orientation_to_rotation(json.load(handle), path))

    def get(self, frame_idx: int | None) -> np.ndarray:
        if not self.rotations:
            return np.eye(3, dtype=np.float64)
        if frame_idx is None:
            return self.rotations[0]
        idx = min(max(int(frame_idx), 0), len(self.rotations) - 1)
        return self.rotations[idx]


class ConstantRCSModel:
    def __init__(self, rcs_m2: float = 1.0):
        if rcs_m2 < 0.0:
            raise ValueError("rcs_m2 must be >= 0")
        self.rcs_m2 = float(rcs_m2)

    def get_rcs(self, theta_deg: float, phi_deg: float) -> complex:
        return complex(self.rcs_m2, 0.0)


class H5RCSModel:
    """H5 RCS pattern reader for the bundled AirSim default-drone model."""

    def __init__(self, h5_path: str | Path):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("H5 RCS files require h5py. Install with: pip install 'lambda-rf[radar]'") from exc

        path = Path(h5_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RCS H5 file not found: {path}")

        with h5py.File(path, "r") as handle:
            self.theta_axis = np.asarray(handle["axes"]["theta"][:], dtype=np.float64).reshape(-1)
            self.phi_axis = np.asarray(handle["axes"]["phi"][:], dtype=np.float64).reshape(-1)
            et_real = np.asarray(handle["E_theta"]["real"][:], dtype=np.float64)
            et_imag = np.asarray(handle["E_theta"]["imag"][:], dtype=np.float64)
        if et_real.shape == (self.phi_axis.size, self.theta_axis.size):
            et_real = et_real.T
            et_imag = et_imag.T
        if et_real.shape != (self.theta_axis.size, self.phi_axis.size):
            raise ValueError(f"Unexpected RCS grid shape {et_real.shape}")
        self.values = et_real + 1j * et_imag

    def get_rcs(self, theta_deg: float, phi_deg: float) -> complex:
        theta = float(np.clip(theta_deg, self.theta_axis[0], self.theta_axis[-1]))
        phi = float(phi_deg) % 360.0
        if phi > self.phi_axis[-1]:
            phi = self.phi_axis[-1]

        ti = int(np.searchsorted(self.theta_axis, theta, side="right") - 1)
        pi = int(np.searchsorted(self.phi_axis, phi, side="right") - 1)
        ti = int(np.clip(ti, 0, self.theta_axis.size - 2))
        pi = int(np.clip(pi, 0, self.phi_axis.size - 2))

        t0, t1 = self.theta_axis[ti], self.theta_axis[ti + 1]
        p0, p1 = self.phi_axis[pi], self.phi_axis[pi + 1]
        wt = 0.0 if t1 == t0 else (theta - t0) / (t1 - t0)
        wp = 0.0 if p1 == p0 else (phi - p0) / (p1 - p0)

        v00 = self.values[ti, pi]
        v10 = self.values[ti + 1, pi]
        v01 = self.values[ti, pi + 1]
        v11 = self.values[ti + 1, pi + 1]
        return complex((1 - wt) * (1 - wp) * v00 + wt * (1 - wp) * v10 + (1 - wt) * wp * v01 + wt * wp * v11)

@dataclass(frozen=True)
class RadarSystem:
    f_c: float
    bandwidth: float = 2.0e9
    chirp_duration: float = 40.0e-6
    chirp_interval: float | None = None
    num_chirps: int = 64
    sample_rate: float = 204.8e6
    noise_floor_dbm: float | None = -100.0
    noise_figure_db: float = 0.0
    noise_bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        if self.f_c <= 0.0:
            raise ValueError("f_c must be > 0")
        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0")
        if self.chirp_duration <= 0.0:
            raise ValueError("chirp_duration must be > 0")
        if self.effective_chirp_interval < self.chirp_duration:
            raise ValueError("chirp_interval must be >= chirp_duration")
        if self.num_chirps <= 0:
            raise ValueError("num_chirps must be > 0")
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be > 0")

    @property
    def lambda_c(self) -> float:
        return C_M_S / self.f_c

    @property
    def effective_chirp_interval(self) -> float:
        return float(self.chirp_duration if self.chirp_interval is None else self.chirp_interval)

    @property
    def idle_time(self) -> float:
        return self.effective_chirp_interval - self.chirp_duration

    @property
    def num_samples(self) -> int:
        return int(round(self.chirp_duration * self.sample_rate))

    @property
    def slope(self) -> float:
        return self.bandwidth / self.chirp_duration

    @property
    def effective_noise_bandwidth_hz(self) -> float:
        return float(self.noise_bandwidth_hz or self.sample_rate)

    @property
    def noise_floor_effective_dbm(self) -> float:
        if self.noise_floor_dbm is not None:
            return float(self.noise_floor_dbm)
        return -174.0 + 10.0 * math.log10(self.effective_noise_bandwidth_hz) + float(self.noise_figure_db)

    @property
    def noise_power_w(self) -> float:
        return 10.0 ** ((self.noise_floor_effective_dbm - 30.0) / 10.0)

    @property
    def noise_std(self) -> float:
        return math.sqrt(self.noise_power_w / 2.0)

    def params_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.f_c,
                self.slope,
                self.sample_rate,
                self.chirp_duration,
                self.num_samples,
                self.effective_chirp_interval,
            ],
            dtype=np.float64,
        )


def _required(arrays: dict[str, np.ndarray], key: str, path: Path) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"{path} is missing required CSI field {key!r}")
    return arrays[key]


def load_csi_paths(npz_path: str | Path, fallback_carrier_frequency_hz: float | None = None) -> dict[str, Any] | None:
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    a_real = _required(arrays, "a_real", path)
    a_imag = _required(arrays, "a_imag", path)
    if a_real.shape != a_imag.shape:
        raise ValueError(f"{path}: a_real shape {a_real.shape} != a_imag shape {a_imag.shape}")
    if a_real.size == 0:
        return None

    a_complex = (a_real.astype(np.float64) + 1j * a_imag.astype(np.float64)).reshape(-1)
    num_paths = a_complex.size
    valid = arrays.get("valid")
    if valid is not None and np.asarray(valid).shape == a_real.shape:
        mask = np.asarray(valid, dtype=bool).reshape(-1)
    else:
        mask = np.ones(num_paths, dtype=bool)
    if not np.any(mask):
        return None

    def mask_path_array(key: str) -> np.ndarray | None:
        if key not in arrays:
            return None
        value = np.asarray(arrays[key])
        if value.shape[-1:] == (num_paths,):
            return value[..., mask]
        if value.ndim >= 2 and value.shape[-2:] == (num_paths, 3):
            return value[..., mask, :]
        return value

    result: dict[str, Any] = {
        "a": a_complex[mask],
        "tau": _required(arrays, "tau", path).reshape(-1)[mask].astype(np.float64),
        "theta_r": _required(arrays, "theta_r", path).reshape(-1)[mask].astype(np.float64),
        "phi_r": _required(arrays, "phi_r", path).reshape(-1)[mask].astype(np.float64),
        "theta_t": _required(arrays, "theta_t", path).reshape(-1)[mask].astype(np.float64),
        "phi_t": _required(arrays, "phi_t", path).reshape(-1)[mask].astype(np.float64),
        "doppler": arrays.get("doppler", np.zeros_like(a_real)).reshape(-1)[mask].astype(np.float64),
        "num_paths": int(np.count_nonzero(mask)),
        "uav_pos": arrays.get("uav_pos"),
        "uav_vel": arrays.get("uav_vel"),
        "tx_pos": arrays.get("tx_pos"),
        "interactions": mask_path_array("interactions"),
        "vertices": mask_path_array("vertices"),
        "objects": mask_path_array("objects"),
        "primitives": mask_path_array("primitives"),
        "timestamp": float(np.asarray(arrays.get("t", arrays.get("timestamp", 0.0))).reshape(-1)[0]),
        "carrier_frequency": float(
            np.asarray(arrays.get("carrier_frequency", fallback_carrier_frequency_hz or 0.0)).reshape(-1)[0]
        ),
    }
    if result["carrier_frequency"] <= 0.0:
        raise ValueError(f"{path}: carrier_frequency is missing and no fallback was provided")
    return result


def _direction_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    sin_theta = np.sin(theta)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=1)


def _position3(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"spherical-wave radar model requires {name}")
    position = np.asarray(value, dtype=np.float64).reshape(-1)
    if position.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 coordinates, got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return position


def _representative_path_steps(values: Any, num_paths: int, vector_dim: int | None = None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        if vector_dim is None:
            return np.zeros((0, num_paths), dtype=arr.dtype)
        return np.zeros((0, num_paths, vector_dim), dtype=np.float64)
    if vector_dim is None:
        if arr.shape[-1] != num_paths:
            raise ValueError(f"path-step array must end with path dimension {num_paths}, got {arr.shape}")
        if arr.ndim == 1:
            return arr.reshape(1, num_paths)
        if arr.ndim == 2:
            return arr
        steps = arr.shape[0]
        flat = arr.reshape(steps, int(np.prod(arr.shape[1:-1])), num_paths)
        out = flat[:, 0, :].copy()
        for step_idx in range(steps):
            for path_idx in range(num_paths):
                nonzero = np.flatnonzero(flat[step_idx, :, path_idx] != 0)
                if nonzero.size:
                    out[step_idx, path_idx] = flat[step_idx, nonzero[0], path_idx]
        return out

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim < 3 or arr.shape[-2:] != (num_paths, vector_dim):
        raise ValueError(f"vertex array must have shape (..., {num_paths}, {vector_dim}), got {arr.shape}")
    if arr.ndim == 3:
        return arr
    steps = arr.shape[0]
    flat = arr.reshape(steps, int(np.prod(arr.shape[1:-2])), num_paths, vector_dim)
    out = np.full((steps, num_paths, vector_dim), np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(flat), axis=-1)
    for step_idx in range(steps):
        for path_idx in range(num_paths):
            rows = np.flatnonzero(finite[step_idx, :, path_idx])
            if rows.size:
                out[step_idx, path_idx, :] = flat[step_idx, rows[0], path_idx, :]
    return out


def _vertices_for_path(vertices: np.ndarray | None, interactions: np.ndarray | None, path_idx: int) -> np.ndarray:
    if vertices is None:
        if interactions is not None and np.any(interactions[:, path_idx] != 0):
            raise ValueError("spherical-wave radar model requires vertices for NLOS paths")
        return np.zeros((0, 3), dtype=np.float64)

    path_vertices = vertices[:, path_idx, :]
    finite = np.all(np.isfinite(path_vertices), axis=1)
    if interactions is not None and interactions.shape[0] == path_vertices.shape[0]:
        mask = finite & (interactions[:, path_idx] != 0)
    else:
        mask = finite
    selected = path_vertices[mask]
    if interactions is not None and np.any(interactions[:, path_idx] != 0) and selected.size == 0:
        raise ValueError("spherical-wave radar model found an NLOS path without finite vertices")
    return selected


def radar_one_way_tau_by_antenna(
    csi_data: dict[str, Any],
    antenna_positions_m: np.ndarray,
    radar_world_to_local: np.ndarray | None = None,
) -> np.ndarray:
    """Return one-way path delay per radar antenna and path for spherical-wave synthesis."""
    tau = np.asarray(csi_data["tau"], dtype=np.float64).reshape(-1)
    num_paths = tau.size
    ant_local = np.asarray(antenna_positions_m, dtype=np.float64)
    if ant_local.ndim != 2 or ant_local.shape[1] != 3:
        raise ValueError(f"antenna_positions_m must have shape (num_ant, 3), got {ant_local.shape}")

    tx_center = _position3(csi_data.get("tx_pos"), "tx_pos")
    uav_pos = _position3(csi_data.get("uav_pos"), "uav_pos")
    if radar_world_to_local is not None:
        rotation = np.asarray(radar_world_to_local, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("radar_world_to_local must have shape (3, 3)")
        ant_world = ant_local @ rotation
    else:
        ant_world = ant_local
    ant_abs = tx_center[np.newaxis, :] + ant_world

    interactions = _representative_path_steps(csi_data.get("interactions"), num_paths)
    vertices = _representative_path_steps(csi_data.get("vertices"), num_paths, vector_dim=3)

    out = np.zeros((ant_abs.shape[0], num_paths), dtype=np.float64)
    for path_idx in range(num_paths):
        path_vertices = _vertices_for_path(vertices, interactions, path_idx)
        if path_vertices.shape[0] == 0:
            center_length = float(np.linalg.norm(uav_pos - tx_center))
            element_lengths = np.linalg.norm(uav_pos[np.newaxis, :] - ant_abs, axis=1)
        else:
            first_vertex = path_vertices[0]
            last_vertex = path_vertices[-1]
            middle_length = (
                float(np.sum(np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)))
                if path_vertices.shape[0] > 1
                else 0.0
            )
            center_length = (
                float(np.linalg.norm(first_vertex - tx_center))
                + middle_length
                + float(np.linalg.norm(uav_pos - last_vertex))
            )
            element_lengths = np.linalg.norm(first_vertex[np.newaxis, :] - ant_abs, axis=1) + middle_length + float(
                np.linalg.norm(uav_pos - last_vertex)
            )
        out[:, path_idx] = tau[path_idx] + (element_lengths - center_length) / C_M_S
    return out


def synthesize_radar_cube(
    csi_data: dict[str, Any],
    uav_rotation_l2w: np.ndarray,
    rcs_model: ConstantRCSModel | H5RCSModel,
    radar_system: RadarSystem,
    antenna_positions_m: np.ndarray,
    radar_world_to_local: np.ndarray | None = None,
    array_model: str = ARRAY_MODEL_FAR_FIELD,
    add_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    model = normalize_array_model(array_model)
    a_path = np.asarray(csi_data["a"], dtype=np.complex128).reshape(-1)
    tau = np.asarray(csi_data["tau"], dtype=np.float64).reshape(-1)
    doppler = np.asarray(csi_data["doppler"], dtype=np.float64).reshape(-1)
    theta_r = np.asarray(csi_data["theta_r"], dtype=np.float64).reshape(-1)
    phi_r = np.asarray(csi_data["phi_r"], dtype=np.float64).reshape(-1)
    theta_t = np.asarray(csi_data["theta_t"], dtype=np.float64).reshape(-1)
    phi_t = np.asarray(csi_data["phi_t"], dtype=np.float64).reshape(-1)

    num_paths = a_path.size
    if not all(values.size == num_paths for values in (tau, doppler, theta_r, phi_r, theta_t, phi_t)):
        raise ValueError("all CSI path fields must have the same path count")

    ant_pos = np.asarray(antenna_positions_m, dtype=np.float64)
    if ant_pos.ndim != 2 or ant_pos.shape[1] != 3:
        raise ValueError(f"antenna_positions_m must have shape (num_ant, 3), got {ant_pos.shape}")

    uav_rotation_l2w = np.asarray(uav_rotation_l2w, dtype=np.float64)
    if uav_rotation_l2w.shape != (3, 3):
        raise ValueError("uav_rotation_l2w must have shape (3, 3)")

    rx_dirs_world = _direction_vectors(theta_r, phi_r)
    body_dirs = rx_dirs_world @ uav_rotation_l2w
    z_body = np.clip(body_dirs[:, 2], -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(z_body))
    phi_deg = np.degrees(np.arctan2(body_dirs[:, 1], body_dirs[:, 0]))
    rcs_values = np.asarray([rcs_model.get_rcs(t, p) for t, p in zip(theta_deg, phi_deg)], dtype=np.complex128)

    if radar_world_to_local is not None:
        radar_world_to_local = np.asarray(radar_world_to_local, dtype=np.float64)
        if radar_world_to_local.shape != (3, 3):
            raise ValueError("radar_world_to_local must have shape (3, 3)")

    coeff = (a_path ** 2) * np.sqrt(rcs_values)
    if model == ARRAY_MODEL_SPHERICAL:
        two_way_tau = 2.0 * radar_one_way_tau_by_antenna(
            csi_data=csi_data,
            antenna_positions_m=ant_pos,
            radar_world_to_local=radar_world_to_local,
        )
        array_sv = None
    else:
        tx_dirs_world = _direction_vectors(theta_t, phi_t)
        radar_dirs = tx_dirs_world @ radar_world_to_local.T if radar_world_to_local is not None else tx_dirs_world
        two_way_tau = 2.0 * tau
        k_wave = 2.0 * np.pi / radar_system.lambda_c
        array_sv = np.exp(1j * k_wave * (ant_pos @ radar_dirs.T))
    two_way_doppler = 2.0 * doppler

    t_fast = np.arange(radar_system.num_samples, dtype=np.float64) / radar_system.sample_rate
    t_slow = np.arange(radar_system.num_chirps, dtype=np.float64) * radar_system.effective_chirp_interval
    fast_grid, slow_grid = np.meshgrid(t_fast, t_slow)

    cube = np.zeros((ant_pos.shape[0], radar_system.num_chirps, radar_system.num_samples), dtype=np.complex128)
    for path_idx in range(num_paths):
        if model == ARRAY_MODEL_SPHERICAL:
            tau_path = two_way_tau[:, path_idx]
            phase = (
                -2.0 * np.pi * radar_system.slope * tau_path[:, np.newaxis, np.newaxis] * fast_grid
                + 2.0 * np.pi * two_way_doppler[path_idx] * slow_grid
                - 2.0 * np.pi * radar_system.f_c * tau_path[:, np.newaxis, np.newaxis]
            )
            base = coeff[path_idx] * np.exp(1j * phase)
            cube += base
        else:
            tau_path = two_way_tau[path_idx]
            phase = (
                -2.0 * np.pi * radar_system.slope * tau_path * fast_grid
                + 2.0 * np.pi * two_way_doppler[path_idx] * slow_grid
                - 2.0 * np.pi * radar_system.f_c * tau_path
            )
            base = coeff[path_idx] * np.exp(1j * phase)
            cube += array_sv[:, path_idx][:, None, None] * base[None, :, :]

    if add_noise:
        rng = rng or np.random.default_rng()
        noise = radar_system.noise_std * (
            rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)
        )
        cube += noise

    return np.fft.fft(cube, axis=-1).astype(np.complex64)


def load_radar_npz(npz_path: str | Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "radar_data" not in data:
            raise KeyError(f"{path} is missing radar_data")
        if "radar_params" not in data:
            raise KeyError(f"{path} is missing radar_params")
        cube = data["radar_data"]
        radar_params = data["radar_params"].astype(np.float64)
        info = {
            "timestamp": float(np.asarray(data.get("timestamp", 0.0)).reshape(-1)[0]),
            "gt_pos": data["gt_pos"] if "gt_pos" in data else None,
            "gt_vel": data["gt_vel"] if "gt_vel" in data else None,
            "source_csi_path": str(data["source_csi_path"]) if "source_csi_path" in data else "",
        }
    return cube, info, radar_params

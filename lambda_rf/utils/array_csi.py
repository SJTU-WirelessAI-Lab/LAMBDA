from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


C_M_S = 299792458.0


def validate_array_shape(shape: tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must contain exactly two integers: rows, cols")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name} rows and cols must be > 0, got {shape}")
    return rows, cols


def parse_array_shape(
    value: str | tuple[int, int] | list[int] | None,
    default: tuple[int, int] | list[int],
    name: str = "array_shape",
) -> tuple[int, int]:
    if value is None:
        return validate_array_shape(default, name)
    if isinstance(value, str):
        parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid {name} {value!r}; expected ROWS,COLS or ROWSxCOLS")
        return validate_array_shape((int(parts[0]), int(parts[1])), name)
    return validate_array_shape(value, name)


def planar_array_positions(
    shape: tuple[int, int] | list[int],
    wavelength_m: float,
    spacing_wavelengths: float = 0.5,
) -> np.ndarray:
    """Return centered planar array positions on the y-z plane, shape (N, 3)."""
    rows, cols = validate_array_shape(shape, "array_shape")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0")
    if spacing_wavelengths <= 0.0:
        raise ValueError("spacing_wavelengths must be > 0")

    spacing_m = wavelength_m * spacing_wavelengths
    y_range = np.arange(cols, dtype=np.float64) * spacing_m
    z_range = np.arange(rows, dtype=np.float64) * spacing_m
    y_range -= np.mean(y_range)
    z_range -= np.mean(z_range)
    y_grid, z_grid = np.meshgrid(y_range, z_range)
    x_grid = np.zeros_like(y_grid)
    return np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)


def validate_rotation_matrix(matrix: np.ndarray | None, name: str) -> np.ndarray:
    if matrix is None:
        return np.eye(3, dtype=np.float64)
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} contains non-finite values")
    should_be_identity = rotation.T @ rotation
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} must be orthonormal")
    det = float(np.linalg.det(rotation))
    if not math.isclose(det, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{name} determinant must be +1, got {det}")
    return rotation


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must contain 4 values [x, y, z, w], got shape {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("quaternion contains non-finite values")

    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be > 0")
    x, y, z, w = quat / norm

    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _orientation_dict_to_quaternion_xyzw(orientation: dict[str, Any]) -> np.ndarray:
    for key in ("x", "y", "z", "w"):
        if key not in orientation:
            raise KeyError(f"orientation is missing {key!r}")
    return np.asarray(
        [
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ],
        dtype=np.float64,
    )


def load_rotation_matrix_from_pose_json(pose_path: str | Path) -> np.ndarray:
    """Load a local-to-world rotation matrix from a camera/pose JSON quaternion."""
    path = Path(pose_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orientation = None
    if isinstance(data.get("world_transform"), dict):
        orientation = data["world_transform"].get("orientation")
    if orientation is None:
        orientation = data.get("orientation")
    if orientation is None and isinstance(data.get("rotation"), dict):
        orientation = data["rotation"].get("orientation") or data["rotation"].get("quaternion")
    if orientation is None:
        raise KeyError(f"No supported orientation quaternion found in {path}")

    if isinstance(orientation, dict):
        quat = _orientation_dict_to_quaternion_xyzw(orientation)
    else:
        quat_values = np.asarray(orientation, dtype=np.float64).reshape(-1)
        if quat_values.shape != (4,):
            raise ValueError(f"orientation list in {path} must contain 4 values")
        # List values are interpreted as scipy-style [x, y, z, w].
        quat = quat_values

    return quaternion_xyzw_to_rotation_matrix(quat)


def rotate_array_positions(positions_m: np.ndarray, rotation_matrix: np.ndarray | None) -> np.ndarray:
    """Rotate local array offsets into the world frame."""
    rotation = validate_rotation_matrix(rotation_matrix, "rotation_matrix")
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_m must have shape (num_ant, 3), got {positions.shape}")
    return positions @ rotation.T


def direction_unit_vectors(theta_rad: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_rad, dtype=np.float64).reshape(-1)
    if theta.shape != phi.shape:
        raise ValueError(f"theta shape {theta.shape} != phi shape {phi.shape}")

    sin_theta = np.sin(theta)
    return np.stack(
        [
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            np.cos(theta),
        ],
        axis=1,
    )


def steering_vectors(
    positions_m: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """Return steering matrix with shape (num_ant, num_paths)."""
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_m must have shape (num_ant, 3), got {positions.shape}")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0")

    directions = direction_unit_vectors(theta_rad, phi_rad)
    phase = (2.0 * np.pi / wavelength_m) * (positions @ directions.T)
    return np.exp(1j * phase)


def representative_by_path(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Reduce all non-path dimensions and return one representative value per path."""
    values = np.asarray(values)
    if values.ndim == 0:
        return np.asarray([values.item()])

    num_paths = values.shape[-1]
    if num_paths == 0:
        return np.asarray([], dtype=values.dtype)

    flat_values = values.reshape(-1, num_paths)
    if valid is None or np.asarray(valid).shape != values.shape:
        return flat_values[0]

    flat_valid = np.asarray(valid).reshape(-1, num_paths)
    reduced = np.empty(num_paths, dtype=values.dtype)
    for path_idx in range(num_paths):
        valid_rows = np.flatnonzero(flat_valid[:, path_idx])
        reduced[path_idx] = flat_values[valid_rows[0], path_idx] if valid_rows.size else flat_values[0, path_idx]
    return reduced


def _required_array(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"Missing required CSI field: {key}")
    return arrays[key]


def build_array_csi_fields(
    arrays: dict[str, np.ndarray],
    carrier_frequency_hz: float,
    tx_shape: tuple[int, int] | list[int],
    rx_shape: tuple[int, int] | list[int],
    spacing_wavelengths: float = 0.5,
    tx_rotation_matrix: np.ndarray | None = None,
    rx_rotation_matrix: np.ndarray | None = None,
    tx_orientation_source: str | None = None,
    rx_orientation_source: str | None = None,
) -> dict[str, np.ndarray]:
    """Expand single-link per-path CSI into far-field MIMO path coefficients."""
    tx_shape = validate_array_shape(tx_shape, "tx_shape")
    rx_shape = validate_array_shape(rx_shape, "rx_shape")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be > 0")

    a_real = _required_array(arrays, "a_real")
    a_imag = _required_array(arrays, "a_imag")
    if a_real.shape != a_imag.shape:
        raise ValueError(f"a_real shape {a_real.shape} != a_imag shape {a_imag.shape}")

    valid = arrays.get("valid")
    a_path = representative_by_path(a_real, valid).astype(np.float64) + 1j * representative_by_path(a_imag, valid).astype(np.float64)
    num_paths = a_path.shape[0]

    theta_t = representative_by_path(_required_array(arrays, "theta_t"), valid)
    phi_t = representative_by_path(_required_array(arrays, "phi_t"), valid)
    theta_r = representative_by_path(_required_array(arrays, "theta_r"), valid)
    phi_r = representative_by_path(_required_array(arrays, "phi_r"), valid)
    for key, values in {
        "theta_t": theta_t,
        "phi_t": phi_t,
        "theta_r": theta_r,
        "phi_r": phi_r,
    }.items():
        if values.shape[0] != num_paths:
            raise ValueError(f"{key} reduced path count {values.shape[0]} != CSI path count {num_paths}")

    wavelength_m = C_M_S / float(carrier_frequency_hz)
    tx_rotation = validate_rotation_matrix(tx_rotation_matrix, "tx_rotation_matrix")
    rx_rotation = validate_rotation_matrix(rx_rotation_matrix, "rx_rotation_matrix")
    tx_positions_local = planar_array_positions(tx_shape, wavelength_m, spacing_wavelengths=spacing_wavelengths)
    rx_positions_local = planar_array_positions(rx_shape, wavelength_m, spacing_wavelengths=spacing_wavelengths)
    tx_positions = rotate_array_positions(tx_positions_local, tx_rotation)
    rx_positions = rotate_array_positions(rx_positions_local, rx_rotation)

    if num_paths == 0:
        a_mimo = np.zeros((rx_positions.shape[0], tx_positions.shape[0], 0), dtype=np.complex128)
    else:
        tx_sv = steering_vectors(tx_positions, theta_t, phi_t, wavelength_m)
        rx_sv = steering_vectors(rx_positions, theta_r, phi_r, wavelength_m)
        a_mimo = rx_sv[:, np.newaxis, :] * np.conjugate(tx_sv[np.newaxis, :, :]) * a_path[np.newaxis, np.newaxis, :]

    return {
        "a_mimo_real": a_mimo.real.astype(np.float32),
        "a_mimo_imag": a_mimo.imag.astype(np.float32),
        "tx_array_pos": tx_positions.astype(np.float32),
        "rx_array_pos": rx_positions.astype(np.float32),
        "tx_array_pos_local": tx_positions_local.astype(np.float32),
        "rx_array_pos_local": rx_positions_local.astype(np.float32),
        "tx_array_rotation": tx_rotation.astype(np.float32),
        "rx_array_rotation": rx_rotation.astype(np.float32),
        "tx_array_shape": np.asarray(tx_shape, dtype=np.int32),
        "rx_array_shape": np.asarray(rx_shape, dtype=np.int32),
        "array_spacing_wavelengths": np.asarray(spacing_wavelengths, dtype=np.float32),
        "array_model": np.asarray("far_field_steering_from_single_link_with_optional_orientation"),
        "array_orientation_model": np.asarray("local_yz_panel_local_x_boresight"),
        "tx_array_orientation_source": np.asarray(tx_orientation_source or "identity"),
        "rx_array_orientation_source": np.asarray(rx_orientation_source or "identity"),
    }


def expand_csi_npz(
    input_path: str | Path,
    output_path: str | Path,
    tx_shape: tuple[int, int] | list[int],
    rx_shape: tuple[int, int] | list[int],
    spacing_wavelengths: float = 0.5,
    tx_rotation_matrix: np.ndarray | None = None,
    rx_rotation_matrix: np.ndarray | None = None,
    tx_orientation_source: str | None = None,
    rx_orientation_source: str | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with np.load(input_path, allow_pickle=False) as data:
        arrays: dict[str, Any] = {key: data[key] for key in data.files}

    carrier_frequency = float(arrays.get("carrier_frequency", 0.0))
    if carrier_frequency <= 0.0:
        raise ValueError(f"{input_path} is missing a positive carrier_frequency field")

    array_fields = build_array_csi_fields(
        arrays=arrays,
        carrier_frequency_hz=carrier_frequency,
        tx_shape=tx_shape,
        rx_shape=rx_shape,
        spacing_wavelengths=spacing_wavelengths,
        tx_rotation_matrix=tx_rotation_matrix,
        rx_rotation_matrix=rx_rotation_matrix,
        tx_orientation_source=tx_orientation_source,
        rx_orientation_source=rx_orientation_source,
    )
    arrays.update(array_fields)
    arrays["source_csi_path"] = np.asarray(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(output_path)

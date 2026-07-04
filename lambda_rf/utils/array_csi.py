from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


C_M_S = 299792458.0
ARRAY_MODEL_FAR_FIELD = "far-field"
ARRAY_MODEL_SPHERICAL = "spherical-wave"
ARRAY_MODEL_ALIASES = {
    "far-field": ARRAY_MODEL_FAR_FIELD,
    "far_field": ARRAY_MODEL_FAR_FIELD,
    "farfield": ARRAY_MODEL_FAR_FIELD,
    "far_field_steering_from_single_link_with_optional_orientation": ARRAY_MODEL_FAR_FIELD,
    "spherical-wave": ARRAY_MODEL_SPHERICAL,
    "spherical_wave": ARRAY_MODEL_SPHERICAL,
    "spherical": ARRAY_MODEL_SPHERICAL,
    "near-field": ARRAY_MODEL_SPHERICAL,
    "near_field": ARRAY_MODEL_SPHERICAL,
    "nearfield": ARRAY_MODEL_SPHERICAL,
    "spherical_wavefront_from_path_vertices": ARRAY_MODEL_SPHERICAL,
}


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


def normalize_array_model(value: str | None) -> str:
    key = "far-field" if value is None else str(value).strip().lower()
    try:
        return ARRAY_MODEL_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted({ARRAY_MODEL_FAR_FIELD, ARRAY_MODEL_SPHERICAL}))
        raise ValueError(f"array_model must be one of: {choices}; got {value!r}") from exc


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


def _position3(arrays: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in arrays:
            value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            if value.shape != (3,):
                raise ValueError(f"{key} must contain exactly 3 coordinates, got shape {np.asarray(arrays[key]).shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key} contains non-finite coordinates")
            return value
    raise KeyError(f"Missing required position field; expected one of {keys}")


def _representative_interactions(interactions: np.ndarray | None, num_paths: int) -> np.ndarray | None:
    if interactions is None:
        return None
    values = np.asarray(interactions, dtype=np.int32)
    if values.size == 0:
        return np.zeros((0, num_paths), dtype=np.int32)
    if values.shape[-1] != num_paths:
        raise ValueError(f"interactions must end with path dimension {num_paths}, got {values.shape}")
    if values.ndim == 1:
        return values.reshape(1, num_paths)
    if values.ndim == 2:
        return values

    steps = values.shape[0]
    middle = int(np.prod(values.shape[1:-1]))
    flat = values.reshape(steps, middle, num_paths)
    reduced = flat[:, 0, :].copy()
    for step_idx in range(steps):
        for path_idx in range(num_paths):
            nonzero = np.flatnonzero(flat[step_idx, :, path_idx] != 0)
            if nonzero.size:
                reduced[step_idx, path_idx] = flat[step_idx, nonzero[0], path_idx]
    return reduced


def _representative_vertices(vertices: np.ndarray, num_paths: int) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim < 3 or values.shape[-1] != 3 or values.shape[-2] != num_paths:
        raise ValueError(f"vertices must have shape (..., {num_paths}, 3), got {values.shape}")
    if values.ndim == 3:
        return values

    steps = values.shape[0]
    middle = int(np.prod(values.shape[1:-2]))
    flat = values.reshape(steps, middle, num_paths, 3)
    reduced = np.full((steps, num_paths, 3), np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(flat), axis=-1)
    for step_idx in range(steps):
        for path_idx in range(num_paths):
            rows = np.flatnonzero(finite[step_idx, :, path_idx])
            if rows.size:
                reduced[step_idx, path_idx, :] = flat[step_idx, rows[0], path_idx, :]
    return reduced


def _path_vertices_for_index(
    vertices: np.ndarray | None,
    interactions: np.ndarray | None,
    path_idx: int,
) -> tuple[np.ndarray, bool]:
    if interactions is not None:
        path_interactions = interactions[:, path_idx]
        has_interactions = bool(np.any(path_interactions != 0))
    else:
        path_interactions = None
        has_interactions = False

    if vertices is None:
        if has_interactions:
            raise ValueError("spherical-wave array model requires vertices for NLOS paths")
        return np.zeros((0, 3), dtype=np.float64), False

    path_vertices = vertices[:, path_idx, :]
    finite = np.all(np.isfinite(path_vertices), axis=1)
    if path_interactions is not None and path_interactions.shape[0] == path_vertices.shape[0]:
        mask = finite & (path_interactions != 0)
    else:
        mask = finite

    selected = path_vertices[mask]
    if has_interactions and selected.size == 0:
        raise ValueError("spherical-wave array model found an NLOS path without finite vertices")
    return selected, has_interactions


def spherical_path_lengths_from_vertices(
    arrays: dict[str, np.ndarray],
    tx_positions_m: np.ndarray,
    rx_positions_m: np.ndarray,
    num_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return element and center geometric path lengths for a spherical-wave model."""
    tx_center = _position3(arrays, ("tx_pos",))
    rx_center = _position3(arrays, ("uav_pos", "rx_pos"))
    tx_abs = tx_center[np.newaxis, :] + np.asarray(tx_positions_m, dtype=np.float64)
    rx_abs = rx_center[np.newaxis, :] + np.asarray(rx_positions_m, dtype=np.float64)

    raw_vertices = arrays.get("vertices")
    vertices = _representative_vertices(raw_vertices, num_paths) if raw_vertices is not None else None
    interactions = _representative_interactions(arrays.get("interactions"), num_paths)

    element_lengths = np.zeros((rx_abs.shape[0], tx_abs.shape[0], num_paths), dtype=np.float64)
    center_lengths = np.zeros(num_paths, dtype=np.float64)

    for path_idx in range(num_paths):
        path_vertices, _ = _path_vertices_for_index(vertices, interactions, path_idx)
        if path_vertices.shape[0] == 0:
            element_delta = rx_abs[:, np.newaxis, :] - tx_abs[np.newaxis, :, :]
            element_lengths[:, :, path_idx] = np.linalg.norm(element_delta, axis=-1)
            center_lengths[path_idx] = float(np.linalg.norm(rx_center - tx_center))
            continue

        first_vertex = path_vertices[0]
        last_vertex = path_vertices[-1]
        if path_vertices.shape[0] > 1:
            middle_length = float(np.sum(np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)))
        else:
            middle_length = 0.0

        tx_lengths = np.linalg.norm(first_vertex[np.newaxis, :] - tx_abs, axis=1)
        rx_lengths = np.linalg.norm(rx_abs - last_vertex[np.newaxis, :], axis=1)
        element_lengths[:, :, path_idx] = rx_lengths[:, np.newaxis] + tx_lengths[np.newaxis, :] + middle_length
        center_lengths[path_idx] = (
            float(np.linalg.norm(first_vertex - tx_center))
            + middle_length
            + float(np.linalg.norm(rx_center - last_vertex))
        )

    return element_lengths, center_lengths


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
    array_model: str = ARRAY_MODEL_FAR_FIELD,
) -> dict[str, np.ndarray]:
    """Expand single-link per-path CSI into MIMO path coefficients."""
    model = normalize_array_model(array_model)
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

    tau_mimo = None
    path_length_mimo = None
    if num_paths == 0:
        a_mimo = np.zeros((rx_positions.shape[0], tx_positions.shape[0], 0), dtype=np.complex128)
    elif model == ARRAY_MODEL_FAR_FIELD:
        tx_sv = steering_vectors(tx_positions, theta_t, phi_t, wavelength_m)
        rx_sv = steering_vectors(rx_positions, theta_r, phi_r, wavelength_m)
        a_mimo = rx_sv[:, np.newaxis, :] * np.conjugate(tx_sv[np.newaxis, :, :]) * a_path[np.newaxis, np.newaxis, :]
    else:
        tau_path = representative_by_path(_required_array(arrays, "tau"), valid).astype(np.float64)
        if tau_path.shape[0] != num_paths:
            raise ValueError(f"tau reduced path count {tau_path.shape[0]} != CSI path count {num_paths}")
        if not np.all(np.isfinite(tau_path)):
            raise ValueError("tau contains non-finite values")

        element_lengths, center_lengths = spherical_path_lengths_from_vertices(
            arrays=arrays,
            tx_positions_m=tx_positions,
            rx_positions_m=rx_positions,
            num_paths=num_paths,
        )
        delta_length = element_lengths - center_lengths[np.newaxis, np.newaxis, :]
        phase = np.exp(-1j * (2.0 * np.pi / wavelength_m) * delta_length)
        a_mimo = a_path[np.newaxis, np.newaxis, :] * phase
        tau_mimo = tau_path[np.newaxis, np.newaxis, :] + delta_length / C_M_S
        path_length_mimo = tau_mimo * C_M_S

    fields = {
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
        "array_model": np.asarray(
            "far_field_steering_from_single_link_with_optional_orientation"
            if model == ARRAY_MODEL_FAR_FIELD
            else "spherical_wavefront_from_path_vertices"
        ),
        "array_orientation_model": np.asarray("local_yz_panel_local_x_boresight"),
        "tx_array_orientation_source": np.asarray(tx_orientation_source or "identity"),
        "rx_array_orientation_source": np.asarray(rx_orientation_source or "identity"),
    }
    if tau_mimo is not None and path_length_mimo is not None:
        fields.update(
            {
                "tau_mimo": tau_mimo.astype(np.float64),
                "path_length_mimo": path_length_mimo.astype(np.float64),
                "near_field_reference": np.asarray("sionna_center_tau_with_geometric_delta"),
                "near_field_spreading": np.asarray("phase-only"),
            }
        )
    return fields


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
    array_model: str = ARRAY_MODEL_FAR_FIELD,
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
        array_model=array_model,
    )
    arrays.update(array_fields)
    arrays["source_csi_path"] = np.asarray(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(output_path)

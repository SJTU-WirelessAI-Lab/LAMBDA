from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lambda_rf.utils.array_csi import representative_by_path


INPUT_MODES = {"auto", "single", "array"}


def validate_subcarrier_config(num_subcarriers: int, subcarrier_spacing_hz: float) -> tuple[int, float]:
    num = int(num_subcarriers)
    spacing = float(subcarrier_spacing_hz)
    if num <= 0:
        raise ValueError(f"num_subcarriers must be > 0, got {num_subcarriers}")
    if spacing <= 0.0:
        raise ValueError(f"subcarrier_spacing_hz must be > 0, got {subcarrier_spacing_hz}")
    return num, spacing


def centered_subcarrier_offsets(num_subcarriers: int, subcarrier_spacing_hz: float) -> np.ndarray:
    """Return centered OFDM subcarrier offsets in Hz, with DC at index N//2."""
    num, spacing = validate_subcarrier_config(num_subcarriers, subcarrier_spacing_hz)
    bins = np.arange(num, dtype=np.float64) - (num // 2)
    return bins * spacing


def infer_input_mode(arrays: dict[str, np.ndarray], input_mode: str = "auto") -> str:
    mode = str(input_mode).lower()
    if mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}, got {input_mode!r}")

    has_real = "a_mimo_real" in arrays
    has_imag = "a_mimo_imag" in arrays
    if has_real != has_imag:
        missing = "a_mimo_imag" if has_real else "a_mimo_real"
        raise KeyError(f"Missing MIMO CSI field: {missing}")

    if mode == "auto":
        return "array" if has_real else "single"
    if mode == "array" and not has_real:
        raise KeyError("input_mode='array' requires a_mimo_real and a_mimo_imag fields")
    return mode


def path_valid_mask(valid: np.ndarray | None, num_paths: int) -> np.ndarray:
    if valid is None:
        return np.ones(num_paths, dtype=bool)

    valid_array = np.asarray(valid, dtype=bool)
    if valid_array.ndim == 0 or valid_array.shape[-1] != num_paths:
        raise ValueError(f"valid field must end with path dimension {num_paths}, got {valid_array.shape}")
    return valid_array.reshape(-1, num_paths).any(axis=0)


def _required_array(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"Missing required CSI field: {key}")
    return arrays[key]


def _path_delays(arrays: dict[str, np.ndarray], valid: np.ndarray | None, num_paths: int) -> np.ndarray:
    tau = representative_by_path(_required_array(arrays, "tau"), valid).astype(np.float64)
    if tau.shape[0] != num_paths:
        raise ValueError(f"tau path count {tau.shape[0]} != CSI path count {num_paths}")
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau contains non-finite values")
    return tau


def _single_path_coefficients(arrays: dict[str, np.ndarray], valid: np.ndarray | None) -> np.ndarray:
    a_real = _required_array(arrays, "a_real")
    a_imag = _required_array(arrays, "a_imag")
    if a_real.shape != a_imag.shape:
        raise ValueError(f"a_real shape {a_real.shape} != a_imag shape {a_imag.shape}")
    return representative_by_path(a_real, valid).astype(np.float64) + 1j * representative_by_path(a_imag, valid).astype(np.float64)


def _array_path_coefficients(arrays: dict[str, np.ndarray]) -> np.ndarray:
    a_mimo_real = _required_array(arrays, "a_mimo_real")
    a_mimo_imag = _required_array(arrays, "a_mimo_imag")
    if a_mimo_real.shape != a_mimo_imag.shape:
        raise ValueError(f"a_mimo_real shape {a_mimo_real.shape} != a_mimo_imag shape {a_mimo_imag.shape}")
    if a_mimo_real.ndim != 3:
        raise ValueError(f"a_mimo_* must have shape (rx_ant, tx_ant, path), got {a_mimo_real.shape}")
    return a_mimo_real.astype(np.float64) + 1j * a_mimo_imag.astype(np.float64)


def frequency_response_from_paths(
    path_coefficients: np.ndarray,
    tau_s: np.ndarray,
    subcarrier_offsets_hz: np.ndarray,
) -> np.ndarray:
    """Sum path coefficients into frequency-domain CSI over subcarrier offsets."""
    coeffs = np.asarray(path_coefficients, dtype=np.complex128)
    tau = np.asarray(tau_s, dtype=np.float64).reshape(-1)
    offsets = np.asarray(subcarrier_offsets_hz, dtype=np.float64).reshape(-1)
    if coeffs.ndim == 0:
        coeffs = coeffs.reshape(1)
    if coeffs.shape[-1] != tau.shape[0]:
        raise ValueError(f"path coefficient count {coeffs.shape[-1]} != tau count {tau.shape[0]}")

    phase = np.exp(-1j * 2.0 * np.pi * tau[:, np.newaxis] * offsets[np.newaxis, :])
    return np.sum(coeffs[..., :, np.newaxis] * phase, axis=-2)


def frequency_response_from_array_paths(
    path_coefficients: np.ndarray,
    tau_s: np.ndarray,
    subcarrier_offsets_hz: np.ndarray,
) -> np.ndarray:
    """Sum array path coefficients using one delay per RX/TX/path coefficient."""
    coeffs = np.asarray(path_coefficients, dtype=np.complex128)
    tau = np.asarray(tau_s, dtype=np.float64)
    offsets = np.asarray(subcarrier_offsets_hz, dtype=np.float64).reshape(-1)
    if coeffs.shape != tau.shape:
        raise ValueError(f"array path coefficient shape {coeffs.shape} != tau_mimo shape {tau.shape}")
    if coeffs.ndim < 1:
        raise ValueError("array path coefficients must include a path dimension")

    phase = np.exp(-1j * 2.0 * np.pi * tau[..., :, np.newaxis] * offsets)
    return np.sum(coeffs[..., :, np.newaxis] * phase, axis=-2)


def build_subcarrier_csi_fields(
    arrays: dict[str, np.ndarray],
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    input_mode: str = "auto",
    profile_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Build OFDM-like subcarrier CSI from path-level delay/complex coefficient snapshots."""
    num, spacing = validate_subcarrier_config(num_subcarriers, subcarrier_spacing_hz)
    mode = infer_input_mode(arrays, input_mode=input_mode)

    if mode == "array":
        path_coefficients = _array_path_coefficients(arrays)
    else:
        path_coefficients = _single_path_coefficients(arrays, arrays.get("valid"))

    num_paths = int(path_coefficients.shape[-1])
    valid = arrays.get("valid")
    valid_mask = path_valid_mask(valid, num_paths)

    if num_paths:
        path_coefficients = path_coefficients * valid_mask.reshape((1,) * (path_coefficients.ndim - 1) + (num_paths,))

    offsets = centered_subcarrier_offsets(num, spacing)
    if mode == "array" and "tau_mimo" in arrays:
        tau_mimo = np.asarray(arrays["tau_mimo"], dtype=np.float64)
        if tau_mimo.shape != path_coefficients.shape:
            raise ValueError(f"tau_mimo shape {tau_mimo.shape} != a_mimo shape {path_coefficients.shape}")
        valid_broadcast = valid_mask.reshape((1,) * (tau_mimo.ndim - 1) + (num_paths,))
        if np.any(valid_broadcast & (tau_mimo < 0.0)):
            raise ValueError("tau_mimo must be >= 0 for valid paths")
        h_freq = frequency_response_from_array_paths(path_coefficients, tau_mimo, offsets)
        response_model = "array_pair_path_delay_phase_from_snapshot"
    else:
        tau = _path_delays(arrays, valid, num_paths)
        if np.any(valid_mask & (tau < 0.0)):
            raise ValueError("tau must be >= 0 for valid paths")
        h_freq = frequency_response_from_paths(path_coefficients, tau, offsets)
        response_model = "path_delay_phase_from_snapshot"

    carrier_values = np.asarray(_required_array(arrays, "carrier_frequency"), dtype=np.float64).reshape(-1)
    if carrier_values.size != 1:
        raise ValueError(f"carrier_frequency must be scalar, got shape {np.asarray(arrays['carrier_frequency']).shape}")
    carrier_frequency = float(carrier_values[0])
    if carrier_frequency <= 0.0:
        raise ValueError("carrier_frequency must be > 0")

    profile = profile_name or "custom"
    return {
        "h_freq_real": h_freq.real.astype(np.float32),
        "h_freq_imag": h_freq.imag.astype(np.float32),
        "subcarrier_offsets_hz": offsets.astype(np.float64),
        "subcarrier_frequencies_hz": (carrier_frequency + offsets).astype(np.float64),
        "subcarrier_spacing_hz": np.asarray(spacing, dtype=np.float64),
        "num_subcarriers": np.asarray(num, dtype=np.int32),
        "subcarrier_nominal_bandwidth_hz": np.asarray(num * spacing, dtype=np.float64),
        "subcarrier_span_hz": np.asarray(max(0, num - 1) * spacing, dtype=np.float64),
        "subcarrier_input_mode": np.asarray(mode),
        "subcarrier_index_order": np.asarray("centered_dc_at_floor_n_over_2"),
        "subcarrier_profile": np.asarray(profile),
        "frequency_response_model": np.asarray(response_model),
    }


def expand_subcarrier_npz(
    input_path: str | Path,
    output_path: str | Path,
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    input_mode: str = "auto",
    profile_name: str | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with np.load(input_path, allow_pickle=False) as data:
        arrays: dict[str, Any] = {key: data[key] for key in data.files}

    subcarrier_fields = build_subcarrier_csi_fields(
        arrays=arrays,
        num_subcarriers=num_subcarriers,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        input_mode=input_mode,
        profile_name=profile_name,
    )
    arrays.update(subcarrier_fields)
    arrays["source_subcarrier_input_path"] = np.asarray(str(input_path))
    if "source_csi_path" not in arrays:
        arrays["source_csi_path"] = np.asarray(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(output_path)

#!/usr/bin/env python
"""Read and summarize one released CSI NPZ file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def scalar_value(value: np.ndarray) -> Any:
    """Return a Python scalar for zero-dimensional arrays."""
    if value.shape == ():
        return value.item()
    return value


def reduce_by_path(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Reduce all non-path dimensions and keep one representative value per path."""
    values = np.asarray(values)
    if values.ndim == 0:
        return np.asarray([values.item()])

    num_paths = values.shape[-1]
    flat_values = values.reshape(-1, num_paths)

    if valid is None or valid.shape != values.shape:
        return flat_values[0]

    flat_valid = valid.reshape(-1, num_paths)
    reduced = np.empty(num_paths, dtype=values.dtype)
    for path_idx in range(num_paths):
        valid_rows = np.flatnonzero(flat_valid[:, path_idx])
        reduced[path_idx] = (
            flat_values[valid_rows[0], path_idx]
            if valid_rows.size
            else flat_values[0, path_idx]
        )
    return reduced


def unique_interactions_by_path(interactions: np.ndarray) -> list[str]:
    """Return unique interaction-type IDs for each path."""
    interactions = np.asarray(interactions)
    if interactions.ndim == 0:
        return [str(interactions.item())]

    num_paths = interactions.shape[-1]
    flat = interactions.reshape(-1, num_paths)
    result: list[str] = []
    for path_idx in range(num_paths):
        unique_ids = sorted(int(x) for x in np.unique(flat[:, path_idx]))
        result.append(",".join(str(x) for x in unique_ids))
    return result


def load_csi_npz(npz_path: str | Path) -> dict[str, Any]:
    """Load an NPZ file and derive complex CSI/path-level summaries."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    required = {"a_real", "a_imag"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"Missing required field(s): {', '.join(missing)}")

    a_real = arrays["a_real"]
    a_imag = arrays["a_imag"]
    if a_real.shape != a_imag.shape:
        raise ValueError(f"a_real shape {a_real.shape} != a_imag shape {a_imag.shape}")

    csi = a_real.astype(np.float64) + 1j * a_imag.astype(np.float64)
    valid = arrays.get("valid")
    if valid is not None and valid.shape != csi.shape:
        valid = None

    power = np.abs(csi) ** 2
    if valid is not None:
        power = np.where(valid, power, 0.0)

    num_paths = csi.shape[-1] if csi.ndim else 1
    path_power = power.reshape(-1, num_paths).sum(axis=0)
    with np.errstate(divide="ignore"):
        path_power_db = 10.0 * np.log10(path_power)

    path_valid = (
        valid.reshape(-1, num_paths).any(axis=0)
        if valid is not None
        else np.ones(num_paths, dtype=bool)
    )

    summary = {
        "path": path,
        "arrays": arrays,
        "csi": csi,
        "num_paths": int(num_paths),
        "path_valid": path_valid,
        "path_power": path_power,
        "path_power_db": path_power_db,
        "path_abs": np.sqrt(path_power),
        "delay_ns": reduce_by_path(arrays["tau"], valid) * 1e9
        if "tau" in arrays
        else np.full(num_paths, np.nan),
        "doppler_hz": reduce_by_path(arrays["doppler"], valid)
        if "doppler" in arrays
        else np.full(num_paths, np.nan),
        "theta_t_rad": reduce_by_path(arrays["theta_t"], valid)
        if "theta_t" in arrays
        else np.full(num_paths, np.nan),
        "phi_t_rad": reduce_by_path(arrays["phi_t"], valid)
        if "phi_t" in arrays
        else np.full(num_paths, np.nan),
        "theta_r_rad": reduce_by_path(arrays["theta_r"], valid)
        if "theta_r" in arrays
        else np.full(num_paths, np.nan),
        "phi_r_rad": reduce_by_path(arrays["phi_r"], valid)
        if "phi_r" in arrays
        else np.full(num_paths, np.nan),
        "interactions": unique_interactions_by_path(arrays["interactions"])
        if "interactions" in arrays
        else [""] * num_paths,
    }

    if "a_mimo_real" in arrays or "a_mimo_imag" in arrays:
        missing_mimo = [key for key in ("a_mimo_real", "a_mimo_imag") if key not in arrays]
        if missing_mimo:
            raise KeyError(f"Missing MIMO CSI field(s): {', '.join(missing_mimo)}")
        if arrays["a_mimo_real"].shape != arrays["a_mimo_imag"].shape:
            raise ValueError(
                f"a_mimo_real shape {arrays['a_mimo_real'].shape} "
                f"!= a_mimo_imag shape {arrays['a_mimo_imag'].shape}"
            )
        mimo_csi = arrays["a_mimo_real"].astype(np.float64) + 1j * arrays["a_mimo_imag"].astype(np.float64)
        if mimo_csi.ndim != 3:
            raise ValueError(f"a_mimo_* must have shape (rx_ant, tx_ant, path), got {mimo_csi.shape}")
        if mimo_csi.shape[-1] != num_paths:
            raise ValueError(f"MIMO CSI path count {mimo_csi.shape[-1]} != base CSI path count {num_paths}")
        mimo_power = np.abs(mimo_csi) ** 2
        summary.update(
            {
                "mimo_csi": mimo_csi,
                "mimo_shape": mimo_csi.shape,
                "mimo_path_power": mimo_power.reshape(-1, num_paths).sum(axis=0),
            }
        )

    if "h_freq_real" in arrays or "h_freq_imag" in arrays:
        missing_freq = [key for key in ("h_freq_real", "h_freq_imag") if key not in arrays]
        if missing_freq:
            raise KeyError(f"Missing subcarrier CSI field(s): {', '.join(missing_freq)}")
        if arrays["h_freq_real"].shape != arrays["h_freq_imag"].shape:
            raise ValueError(
                f"h_freq_real shape {arrays['h_freq_real'].shape} "
                f"!= h_freq_imag shape {arrays['h_freq_imag'].shape}"
            )
        freq_csi = arrays["h_freq_real"].astype(np.float64) + 1j * arrays["h_freq_imag"].astype(np.float64)
        if freq_csi.ndim not in (1, 3):
            raise ValueError(f"h_freq_* must have shape (subcarrier,) or (rx_ant, tx_ant, subcarrier), got {freq_csi.shape}")
        if "num_subcarriers" in arrays:
            expected_num_subcarriers = int(scalar_value(arrays["num_subcarriers"]))
            if freq_csi.shape[-1] != expected_num_subcarriers:
                raise ValueError(
                    f"frequency CSI subcarrier count {freq_csi.shape[-1]} "
                    f"!= num_subcarriers {expected_num_subcarriers}"
                )
        summary.update(
            {
                "freq_csi": freq_csi,
                "freq_shape": freq_csi.shape,
                "freq_power": np.abs(freq_csi) ** 2,
            }
        )
    return summary


def format_float(value: float, digits: int = 6) -> str:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return str(value)
    return f"{float(value):.{digits}g}"


def print_file_summary(summary: dict[str, Any], top: int) -> None:
    arrays = summary["arrays"]
    print(f"File: {summary['path']}")
    print("Fields:")
    for key, value in arrays.items():
        scalar = scalar_value(value)
        if isinstance(scalar, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, value={scalar}")

    print()
    print(f"Complex CSI shape: {summary['csi'].shape}")
    if "mimo_csi" in summary:
        rx_ant, tx_ant, _ = summary["mimo_shape"]
        print(f"MIMO CSI shape: {summary['mimo_shape']} (rx_ant={rx_ant}, tx_ant={tx_ant}, paths={summary['num_paths']})")
        if "rx_array_shape" in arrays:
            print(f"RX array shape: {arrays['rx_array_shape']}")
        if "tx_array_shape" in arrays:
            print(f"TX array shape: {arrays['tx_array_shape']}")
    if "freq_csi" in summary:
        shape = summary["freq_shape"]
        if len(shape) == 3:
            rx_ant, tx_ant, num_subcarriers = shape
            print(
                f"Subcarrier CSI shape: {shape} "
                f"(rx_ant={rx_ant}, tx_ant={tx_ant}, subcarriers={num_subcarriers})"
            )
        else:
            print(f"Subcarrier CSI shape: {shape} (subcarriers={shape[-1]})")
        if "subcarrier_profile" in arrays:
            print(f"Subcarrier profile: {scalar_value(arrays['subcarrier_profile'])}")
        if "subcarrier_spacing_hz" in arrays:
            print(f"Subcarrier spacing: {format_float(scalar_value(arrays['subcarrier_spacing_hz']))} Hz")
        if "subcarrier_nominal_bandwidth_hz" in arrays:
            print(f"Nominal bandwidth: {format_float(scalar_value(arrays['subcarrier_nominal_bandwidth_hz']))} Hz")
    print(f"Number of paths: {summary['num_paths']}")
    if "uav_pos" in arrays:
        print(f"UAV position: {arrays['uav_pos']}")
    if "uav_vel" in arrays:
        print(f"UAV velocity: {arrays['uav_vel']}")
    print()

    order = np.argsort(summary["path_power"])[::-1]
    if top > 0:
        order = order[:top]

    print("Path summary sorted by power:")
    header = (
        "idx valid delay_ns power_db abs_gain doppler_hz "
        "theta_t phi_t theta_r phi_r interactions"
    )
    print(header)
    for idx in order:
        power_db = summary["path_power_db"][idx]
        print(
            f"{idx:3d} "
            f"{str(bool(summary['path_valid'][idx])):5s} "
            f"{format_float(summary['delay_ns'][idx]):>10s} "
            f"{format_float(power_db):>9s} "
            f"{format_float(summary['path_abs'][idx]):>9s} "
            f"{format_float(summary['doppler_hz'][idx]):>10s} "
            f"{format_float(summary['theta_t_rad'][idx]):>7s} "
            f"{format_float(summary['phi_t_rad'][idx]):>7s} "
            f"{format_float(summary['theta_r_rad'][idx]):>7s} "
            f"{format_float(summary['phi_r_rad'][idx]):>7s} "
            f"{summary['interactions'][idx]}"
        )


def write_csv(summary: dict[str, Any], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    fieldnames = [
        "path_idx",
        "valid",
        "delay_ns",
        "power_db",
        "abs_gain",
        "doppler_hz",
        "theta_t_rad",
        "phi_t_rad",
        "theta_r_rad",
        "phi_r_rad",
        "interactions",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(summary["num_paths"]):
            writer.writerow(
                {
                    "path_idx": idx,
                    "valid": bool(summary["path_valid"][idx]),
                    "delay_ns": summary["delay_ns"][idx],
                    "power_db": summary["path_power_db"][idx],
                    "abs_gain": summary["path_abs"][idx],
                    "doppler_hz": summary["doppler_hz"][idx],
                    "theta_t_rad": summary["theta_t_rad"][idx],
                    "phi_t_rad": summary["phi_t_rad"][idx],
                    "theta_r_rad": summary["theta_r_rad"][idx],
                    "phi_r_rad": summary["phi_r_rad"][idx],
                    "interactions": summary["interactions"][idx],
                }
            )
    print(f"\nCSV saved to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one generated multi-path CSI NPZ file."
    )
    parser.add_argument(
        "npz_path",
        help="Path to csi_*.npz.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of strongest paths to print. Use 0 to print all paths.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path for the per-path summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_csi_npz(args.npz_path)
    top = summary["num_paths"] if args.top == 0 else args.top
    print_file_summary(summary, top=top)
    if args.csv:
        write_csv(summary, args.csv)


if __name__ == "__main__":
    main()

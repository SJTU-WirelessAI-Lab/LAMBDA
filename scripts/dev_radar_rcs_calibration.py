#!/usr/bin/env python3
"""Development-only calibration for FEKO RCS patterns and radar amplitude.

This script is intentionally independent from the production radar generator.
It checks the FEKO relation

    sigma = 4*pi*|F|^2

and compares candidate target-scattering formulas against the free-space
monostatic radar equation. It can also stream an original FEKO .ffe file and
verify every sample against the H5 export.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


C_M_S = 299_792_458.0


@dataclass(frozen=True)
class ScatteringGrid:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray
    attributes: dict[str, object]


def _as_theta_phi(values: np.ndarray, num_theta: int, num_phi: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (num_phi, num_theta):
        array = array.T
    if array.shape != (num_theta, num_phi):
        raise ValueError(
            f"{name} must have shape ({num_theta}, {num_phi}) or "
            f"({num_phi}, {num_theta}), got {array.shape}"
        )
    return array


def load_scattering_grid(path: Path) -> ScatteringGrid:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("This calibration script requires h5py.") from exc

    with h5py.File(path, "r") as handle:
        theta = np.asarray(handle["axes/theta"][:], dtype=np.float64).reshape(-1)
        phi = np.asarray(handle["axes/phi"][:], dtype=np.float64).reshape(-1)
        et_real = _as_theta_phi(handle["E_theta/real"][:], theta.size, phi.size, "E_theta/real")
        et_imag = _as_theta_phi(handle["E_theta/imag"][:], theta.size, phi.size, "E_theta/imag")
        ep_real = _as_theta_phi(handle["E_phi/real"][:], theta.size, phi.size, "E_phi/real")
        ep_imag = _as_theta_phi(handle["E_phi/imag"][:], theta.size, phi.size, "E_phi/imag")
        attributes = {str(key): handle.attrs[key] for key in handle.attrs}

    if theta.size < 2 or phi.size < 2:
        raise ValueError("theta and phi axes must each contain at least two samples")
    if not np.all(np.diff(theta) > 0.0) or not np.all(np.diff(phi) > 0.0):
        raise ValueError("theta and phi axes must be strictly increasing")

    e_theta = et_real + 1j * et_imag
    e_phi = ep_real + 1j * ep_imag
    if not np.all(np.isfinite(e_theta)) or not np.all(np.isfinite(e_phi)):
        raise ValueError("H5 scattering arrays contain non-finite values")

    return ScatteringGrid(theta, phi, e_theta, e_phi, attributes)


def _interp2(grid: ScatteringGrid, values: np.ndarray, theta_deg: float, phi_deg: float) -> complex:
    theta = float(np.clip(theta_deg, grid.theta_deg[0], grid.theta_deg[-1]))
    phi_period = 360.0
    phi = ((float(phi_deg) - grid.phi_deg[0]) % phi_period) + grid.phi_deg[0]
    if phi > grid.phi_deg[-1]:
        phi = grid.phi_deg[-1]

    ti = int(np.searchsorted(grid.theta_deg, theta, side="right") - 1)
    pi = int(np.searchsorted(grid.phi_deg, phi, side="right") - 1)
    ti = int(np.clip(ti, 0, grid.theta_deg.size - 2))
    pi = int(np.clip(pi, 0, grid.phi_deg.size - 2))

    t0, t1 = grid.theta_deg[ti], grid.theta_deg[ti + 1]
    p0, p1 = grid.phi_deg[pi], grid.phi_deg[pi + 1]
    wt = 0.0 if t1 == t0 else (theta - t0) / (t1 - t0)
    wp = 0.0 if p1 == p0 else (phi - p0) / (p1 - p0)

    return complex(
        (1.0 - wt) * (1.0 - wp) * values[ti, pi]
        + wt * (1.0 - wp) * values[ti + 1, pi]
        + (1.0 - wt) * wp * values[ti, pi + 1]
        + wt * wp * values[ti + 1, pi + 1]
    )


def scattering_response(
    grid: ScatteringGrid,
    theta_deg: float,
    phi_deg: float,
    component: str,
) -> tuple[complex, complex, complex]:
    e_theta = _interp2(grid, grid.e_theta, theta_deg, phi_deg)
    e_phi = _interp2(grid, grid.e_phi, theta_deg, phi_deg)
    if component == "theta":
        response = e_theta
    elif component == "phi":
        response = e_phi
    else:
        # Orthogonal polarizations add in power. Total RCS has no unique phase.
        response = complex(math.hypot(abs(e_theta), abs(e_phi)), 0.0)
    return response, e_theta, e_phi


def infer_frequency_hz(path: Path, grid: ScatteringGrid, override_hz: float | None) -> float:
    attr_value = grid.attributes.get("carrier_frequency_hz")
    attr_frequency = float(attr_value) if attr_value is not None else None
    if override_hz is not None:
        frequency = float(override_hz)
        if attr_frequency is not None and not math.isclose(frequency, attr_frequency, rel_tol=1e-9):
            raise ValueError(
                f"--frequency-hz={frequency:g} disagrees with H5 carrier_frequency_hz={attr_frequency:g}"
            )
        return frequency
    if attr_frequency is not None:
        return attr_frequency

    match = re.search(r"(\d+(?:[p.]\d+)?)\s*ghz", path.name, flags=re.IGNORECASE)
    if match:
        return float(match.group(1).replace("p", ".")) * 1e9
    raise ValueError("Frequency is absent from H5 metadata and filename; pass --frequency-hz")


def _db10(value: float) -> float:
    return 10.0 * math.log10(max(float(value), np.finfo(np.float64).tiny))


def _db20(value: float) -> float:
    return 20.0 * math.log10(max(float(value), np.finfo(np.float64).tiny))


def _format_complex(value: complex) -> str:
    return f"{value.real:+.9e}{value.imag:+.9e}j"


def free_space_calibration(
    response: complex,
    frequency_hz: float,
    range_m: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    tx_power_dbm: float,
) -> dict[str, float | complex]:
    wavelength = C_M_S / frequency_hz
    gt = 10.0 ** (tx_gain_dbi / 10.0)
    gr = 10.0 ** (rx_gain_dbi / 10.0)
    sigma_m2 = 4.0 * math.pi * abs(response) ** 2

    forward = math.sqrt(gt) * wavelength / (4.0 * math.pi * range_m)
    reverse = math.sqrt(gr) * wavelength / (4.0 * math.pi * range_m)
    expected_power_ratio = (
        gt
        * gr
        * wavelength**2
        * sigma_m2
        / ((4.0 * math.pi) ** 3 * range_m**4)
    )
    expected_amplitude = math.sqrt(expected_power_ratio)

    legacy = forward * reverse * np.sqrt(response)
    direct = forward * reverse * response
    scale = 4.0 * math.pi / wavelength
    calibrated = forward * reverse * scale * response
    tx_power_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)

    return {
        "wavelength_m": wavelength,
        "sigma_m2": sigma_m2,
        "forward": forward,
        "reverse": reverse,
        "scale_per_m": scale,
        "expected_power_ratio": expected_power_ratio,
        "expected_amplitude": expected_amplitude,
        "expected_rx_dbm": 10.0 * math.log10(tx_power_w * expected_power_ratio) + 30.0,
        "legacy": complex(legacy),
        "direct": complex(direct),
        "calibrated": complex(calibrated),
    }


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), np.finfo(np.float64).tiny)


def validate_ffe(
    ffe_path: Path,
    grid: ScatteringGrid,
    expected_frequency_hz: float,
    limit: int | None,
) -> dict[str, float | int]:
    incident: tuple[float, float] | None = None
    num_theta: int | None = None
    num_phi: int | None = None
    frequency_hz: float | None = None
    count = 0
    max_h5_error = 0.0
    max_rcs_relative_error = 0.0
    max_angle_error_deg = 0.0

    incident_pattern = re.compile(
        r"\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)"
    )
    with ffe_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("#Frequency:"):
                frequency_hz = float(line.split(":", 1)[1])
            elif line.startswith("#No. of Theta Samples:"):
                num_theta = int(line.split(":", 1)[1])
            elif line.startswith("#No. of Phi Samples:"):
                num_phi = int(line.split(":", 1)[1])
            elif line.startswith("#Incident Wave Direction:"):
                match = incident_pattern.search(line)
                if match is None:
                    raise ValueError(f"Could not parse incident direction: {line}")
                incident = float(match.group(1)), float(match.group(2))
            elif line and line[0] not in "#*" and incident is not None:
                values = np.fromstring(line, sep=" ")
                if values.size < 9:
                    continue
                if num_theta != 1 or num_phi != 1:
                    raise ValueError(
                        "FFE validation expects one observation sample per incident direction; "
                        f"got theta={num_theta}, phi={num_phi}"
                    )
                if frequency_hz is None or not math.isclose(
                    frequency_hz, expected_frequency_hz, rel_tol=1e-9
                ):
                    raise ValueError(
                        f"FFE frequency {frequency_hz} does not match {expected_frequency_hz}"
                    )

                inc_theta, inc_phi = incident
                obs_theta, obs_phi = float(values[0]), float(values[1])
                max_angle_error_deg = max(
                    max_angle_error_deg,
                    abs(obs_theta - inc_theta),
                    abs(obs_phi - inc_phi),
                )
                e_theta = complex(values[2], values[3])
                e_phi = complex(values[4], values[5])
                h5_theta = _interp2(grid, grid.e_theta, inc_theta, inc_phi)
                h5_phi = _interp2(grid, grid.e_phi, inc_theta, inc_phi)
                max_h5_error = max(max_h5_error, abs(h5_theta - e_theta), abs(h5_phi - e_phi))

                calculated = (
                    4.0 * math.pi * abs(e_theta) ** 2,
                    4.0 * math.pi * abs(e_phi) ** 2,
                    4.0 * math.pi * (abs(e_theta) ** 2 + abs(e_phi) ** 2),
                )
                for actual, expected in zip(calculated, values[6:9]):
                    max_rcs_relative_error = max(
                        max_rcs_relative_error,
                        _relative_error(actual, float(expected)),
                    )
                count += 1
                incident = None
                if limit is not None and count >= limit:
                    break

    if count == 0:
        raise ValueError(f"No RCS samples were parsed from {ffe_path}")
    return {
        "samples": count,
        "max_h5_abs_error": max_h5_error,
        "max_rcs_relative_error": max_rcs_relative_error,
        "max_angle_error_deg": max_angle_error_deg,
    }


def validate_los_path_coefficients(
    csi_dir: Path,
    expected_frequency_hz: float,
    limit: int,
) -> dict[str, float | int]:
    files = sorted(csi_dir.glob("csi_*.npz"))[:limit]
    if not files:
        raise FileNotFoundError(f"No csi_*.npz files found in {csi_dir}")

    amplitude_ratios: list[float] = []
    delay_errors_m: list[float] = []
    frames_with_los = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            frequency_hz = float(np.asarray(data["carrier_frequency"]).reshape(-1)[0])
            if not math.isclose(frequency_hz, expected_frequency_hz, rel_tol=0.0, abs_tol=1.0):
                raise ValueError(
                    f"{path}: CSI frequency {frequency_hz:g} does not match RCS frequency {expected_frequency_hz:g}"
                )
            a = (np.asarray(data["a_real"]) + 1j * np.asarray(data["a_imag"])).reshape(-1)
            tau = np.asarray(data["tau"], dtype=np.float64).reshape(-1)
            valid = np.asarray(data.get("valid", np.ones(a.size)), dtype=bool).reshape(-1)
            interactions = np.asarray(data["interactions"])
            interactions = interactions.reshape(interactions.shape[0], -1, interactions.shape[-1])
            is_los = np.all(interactions == 0, axis=(0, 1)) & valid
            los_indices = np.flatnonzero(is_los)
            if los_indices.size == 0:
                continue

            tx_pos = np.asarray(data["tx_pos"], dtype=np.float64).reshape(3)
            uav_pos = np.asarray(data["uav_pos"], dtype=np.float64).reshape(3)
            distance_m = float(np.linalg.norm(uav_pos - tx_pos))
            expected_amplitude = C_M_S / expected_frequency_hz / (4.0 * math.pi * distance_m)
            for path_idx in los_indices:
                amplitude_ratios.append(float(abs(a[path_idx]) / expected_amplitude))
                delay_errors_m.append(float(abs(tau[path_idx] * C_M_S - distance_m)))
            frames_with_los += 1

    if not amplitude_ratios:
        raise ValueError(f"No valid zero-interaction LoS paths found in {csi_dir}")
    ratios = np.asarray(amplitude_ratios, dtype=np.float64)
    delay_errors = np.asarray(delay_errors_m, dtype=np.float64)
    return {
        "files_checked": len(files),
        "frames_with_los": frames_with_los,
        "los_paths": ratios.size,
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios)),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "max_delay_error_m": float(np.max(delay_errors)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a FEKO scattering H5 and calibrate radar target amplitude."
    )
    parser.add_argument("h5_path", type=Path, help="H5 containing E_theta, E_phi, and angle axes.")
    parser.add_argument("--ffe", type=Path, help="Optional original POSTFEKO .ffe for sample-by-sample validation.")
    parser.add_argument("--ffe-limit", type=int, help="Only validate the first N FFE samples.")
    parser.add_argument("--frequency-hz", type=float, help="Carrier frequency when absent from H5 metadata/filename.")
    parser.add_argument("--theta", type=float, default=0.0, help="Incident theta angle in degrees.")
    parser.add_argument("--phi", type=float, default=0.0, help="Incident phi angle in degrees.")
    parser.add_argument(
        "--component",
        choices=("theta", "phi", "total"),
        default="theta",
        help="Scattering polarization component used by the free-space calibration.",
    )
    parser.add_argument("--range-m", type=float, default=100.0, help="Free-space target range in meters.")
    parser.add_argument("--tx-gain-dbi", type=float, default=0.0, help="Radar transmit gain in dBi.")
    parser.add_argument("--rx-gain-dbi", type=float, default=0.0, help="Radar receive gain in dBi.")
    parser.add_argument("--tx-power-dbm", type=float, default=0.0, help="Transmit power used for the receive-power report.")
    parser.add_argument("--csi-dir", type=Path, help="Optional real path-level CSI directory for LoS Friis validation.")
    parser.add_argument("--csi-limit", type=int, default=100, help="Maximum CSI frames checked by --csi-dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if args.range_m <= 0.0:
        raise ValueError("--range-m must be > 0")
    if args.ffe_limit is not None and args.ffe_limit <= 0:
        raise ValueError("--ffe-limit must be > 0")
    if args.csi_limit <= 0:
        raise ValueError("--csi-limit must be > 0")

    grid = load_scattering_grid(h5_path)
    frequency_hz = infer_frequency_hz(h5_path, grid, args.frequency_hz)
    response, e_theta, e_phi = scattering_response(
        grid,
        theta_deg=args.theta,
        phi_deg=args.phi,
        component=args.component,
    )
    result = free_space_calibration(
        response=response,
        frequency_hz=frequency_hz,
        range_m=float(args.range_m),
        tx_gain_dbi=float(args.tx_gain_dbi),
        rx_gain_dbi=float(args.rx_gain_dbi),
        tx_power_dbm=float(args.tx_power_dbm),
    )

    print("=" * 84)
    print("FEKO RCS / RADAR AMPLITUDE DEVELOPMENT CALIBRATION")
    print("=" * 84)
    print(f"H5 model:          {h5_path}")
    print(f"Grid:              theta={grid.theta_deg.size}, phi={grid.phi_deg.size}")
    print(f"Carrier:           {frequency_hz / 1e9:.9g} GHz")
    print(f"Wavelength:        {float(result['wavelength_m']):.9e} m")
    print(f"Selected angle:    theta={args.theta:g} deg, phi={args.phi:g} deg")
    print(f"E_theta:           {_format_complex(e_theta)} m")
    print(f"E_phi:             {_format_complex(e_phi)} m")
    print(f"Component:         {args.component}")
    print(f"Target response F: {_format_complex(response)} m")
    print(f"Derived RCS:       {float(result['sigma_m2']):.9e} m^2 ({_db10(float(result['sigma_m2'])):.6f} dBsm)")
    if args.component == "total":
        print("Phase note:         total RCS combines orthogonal components in power; phase is undefined.")

    print("-" * 84)
    print(f"Reference range:   {args.range_m:g} m")
    print(f"TX/RX gain:        {args.tx_gain_dbi:g} / {args.rx_gain_dbi:g} dBi")
    print(f"One-way amplitudes:{float(result['forward']):.9e} / {float(result['reverse']):.9e}")
    print(
        f"Required target scale: 4*pi/lambda = {float(result['scale_per_m']):.9e} 1/m "
        f"({_db20(float(result['scale_per_m'])):.6f} dB amplitude)"
    )
    print(f"Radar-equation amplitude: {float(result['expected_amplitude']):.9e}")
    print(f"Radar-equation power ratio: {float(result['expected_power_ratio']):.9e}")
    print(f"Expected receive power: {float(result['expected_rx_dbm']):.6f} dBm")

    print("-" * 84)
    print(f"{'Candidate':<34}{'|h|':>16}{'power error':>18}")
    print("-" * 84)
    expected_amplitude = float(result["expected_amplitude"])
    for label, key in (
        ("legacy: A_fwd*A_rev*sqrt(F)", "legacy"),
        ("direct: A_fwd*A_rev*F", "direct"),
        ("radar eq: A_fwd*A_rev*(4pi/lambda)*F", "calibrated"),
    ):
        amplitude = abs(complex(result[key]))
        error_db = _db20(amplitude / expected_amplitude)
        print(f"{label:<34}{amplitude:>16.9e}{error_db:>+17.6f} dB")

    calibrated_error = _relative_error(abs(complex(result["calibrated"])), expected_amplitude)
    print("-" * 84)
    print(f"Calibrated relative error: {calibrated_error:.3e}")
    if calibrated_error > 1e-12:
        raise AssertionError("Free-space calibrated amplitude does not match the radar equation")
    print("[PASS] The 4*pi/lambda scattering normalization matches the monostatic radar equation.")

    if args.ffe is not None:
        ffe_path = args.ffe.expanduser().resolve()
        if not ffe_path.is_file():
            raise FileNotFoundError(ffe_path)
        report = validate_ffe(
            ffe_path=ffe_path,
            grid=grid,
            expected_frequency_hz=frequency_hz,
            limit=args.ffe_limit,
        )
        print("-" * 84)
        print(f"FFE samples checked:       {int(report['samples'])}")
        print(f"FFE/H5 max field error:    {float(report['max_h5_abs_error']):.3e}")
        print(f"FFE RCS max relative error:{float(report['max_rcs_relative_error']):.3e}")
        print(f"Observation/incident error:{float(report['max_angle_error_deg']):.3e} deg")
        if float(report["max_angle_error_deg"]) > 1e-9:
            raise AssertionError("FFE observation direction is not monostatic with the incident direction")
        if float(report["max_rcs_relative_error"]) > 1e-6:
            raise AssertionError("FFE RCS columns do not match 4*pi*|E|^2")
        print("[PASS] FFE samples, H5 fields, and FEKO RCS columns are consistent.")

    if args.csi_dir is not None:
        csi_report = validate_los_path_coefficients(
            csi_dir=args.csi_dir.expanduser().resolve(),
            expected_frequency_hz=frequency_hz,
            limit=args.csi_limit,
        )
        print("-" * 84)
        print(f"CSI files checked:         {int(csi_report['files_checked'])}")
        print(f"Frames with LoS:           {int(csi_report['frames_with_los'])}")
        print(f"LoS paths checked:         {int(csi_report['los_paths'])}")
        print(
            "|a| / [lambda/(4*pi*R)]: "
            f"mean={float(csi_report['ratio_mean']):.9f}, "
            f"std={float(csi_report['ratio_std']):.3e}, "
            f"range=[{float(csi_report['ratio_min']):.9f}, {float(csi_report['ratio_max']):.9f}]"
        )
        print(f"Maximum LoS delay error:   {float(csi_report['max_delay_error_m']):.3e} m")
        if abs(float(csi_report["ratio_mean"]) - 1.0) > 5e-3:
            raise AssertionError("Actual LoS path amplitudes do not follow the Friis voltage coefficient")
        print("[PASS] Actual path coefficients use the one-way Friis voltage normalization.")


if __name__ == "__main__":
    main()

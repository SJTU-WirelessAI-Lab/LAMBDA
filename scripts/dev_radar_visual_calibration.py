#!/usr/bin/env python3
"""Run a synthetic 28 GHz point-target calibration through production radar code."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lambda_rf.utils.radar import (
    C_M_S,
    H5RCSModel,
    RadarSystem,
    synthesize_radar_cube,
    virtual_array_positions,
)
from lambda_rf.visualize_radar import compute_radar_maps, visualize_file


class FixedScatteringModel:
    def __init__(self, value: complex):
        self.value = complex(value)

    def get_scattering_amplitude(self, theta_deg: float, phi_deg: float) -> complex:
        return self.value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "h5_path",
        nargs="?",
        type=Path,
        default=Path("assets/default_drone_rcs_28ghz.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_export/dev_radar_visual_calibration"),
    )
    parser.add_argument("--range-m", type=float, default=60.0)
    parser.add_argument("--velocity-mps", type=float, default=10.0)
    parser.add_argument("--azimuth-deg", type=float, default=15.0)
    parser.add_argument("--elevation-deg", type=float, default=10.0)
    parser.add_argument("--rcs-theta-deg", type=float, default=0.0)
    parser.add_argument("--rcs-phi-deg", type=float, default=0.0)
    parser.add_argument("--bandwidth-hz", type=float, default=1.0e9)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0e6)
    parser.add_argument("--chirp-duration-s", type=float, default=40.0e-6)
    parser.add_argument("--chirp-interval-s", type=float, default=50.0e-6)
    parser.add_argument("--num-chirps", type=int, default=128)
    parser.add_argument("--array-rows", type=int, default=4)
    parser.add_argument("--array-cols", type=int, default=4)
    parser.add_argument("--spacing-wavelengths", type=float, default=0.49)
    parser.add_argument("--angle-fft-size", type=int, default=64)
    parser.add_argument("--tx-power-dbm", type=float, default=12.0)
    parser.add_argument("--tx-gain-db", type=float, default=25.0)
    parser.add_argument("--rx-gain-db", type=float, default=25.0)
    parser.add_argument("--noise-figure-db", type=float, default=6.0)
    parser.add_argument("--noise-bandwidth-hz", type=float, default=100.0e6)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--no-noise", action="store_true")
    return parser.parse_args()


def direction_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def angles_from_direction(direction: np.ndarray) -> tuple[float, float]:
    theta = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
    phi = math.atan2(float(direction[1]), float(direction[0]))
    return theta, phi


def peak_2d(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[float, float]:
    y_idx, x_idx = np.unravel_index(int(np.argmax(image)), image.shape)
    return float(x_axis[x_idx]), float(y_axis[y_idx])


def main() -> None:
    args = parse_args()
    if args.range_m <= 0.0:
        raise ValueError("--range-m must be > 0")
    if args.chirp_interval_s < args.chirp_duration_s:
        raise ValueError("--chirp-interval-s must be >= --chirp-duration-s")

    h5_path = args.h5_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rcs_model = H5RCSModel(h5_path, expected_frequency_hz=28.0e9, component="theta")
    scattering = rcs_model.get_scattering_amplitude(args.rcs_theta_deg, args.rcs_phi_deg)
    sigma_m2 = rcs_model.get_rcs(args.rcs_theta_deg, args.rcs_phi_deg)

    radar = RadarSystem(
        f_c=28.0e9,
        bandwidth=args.bandwidth_hz,
        sample_rate=args.sample_rate_hz,
        chirp_duration=args.chirp_duration_s,
        chirp_interval=args.chirp_interval_s,
        num_chirps=args.num_chirps,
        noise_floor_dbm=None,
        noise_figure_db=args.noise_figure_db,
        noise_bandwidth_hz=args.noise_bandwidth_hz,
        tx_power_dbm=args.tx_power_dbm,
        tx_gain_db=args.tx_gain_db,
        rx_gain_db=args.rx_gain_db,
    )
    direction = direction_vector(args.azimuth_deg, args.elevation_deg)
    theta_t, phi_t = angles_from_direction(direction)
    theta_r, phi_r = angles_from_direction(-direction)
    one_way_a = radar.lambda_c / (4.0 * math.pi * args.range_m)
    csi_data = {
        "a": np.asarray([one_way_a + 0.0j]),
        "tau": np.asarray([args.range_m / C_M_S]),
        "doppler": np.asarray([args.velocity_mps / radar.lambda_c]),
        "theta_r": np.asarray([theta_r]),
        "phi_r": np.asarray([phi_r]),
        "theta_t": np.asarray([theta_t]),
        "phi_t": np.asarray([phi_t]),
    }
    shape = (args.array_rows, args.array_cols)
    antenna_positions = virtual_array_positions(
        radar.f_c,
        shape=shape,
        spacing_wavelengths=args.spacing_wavelengths,
    )
    cube = synthesize_radar_cube(
        csi_data=csi_data,
        uav_rotation_l2w=np.eye(3),
        rcs_model=FixedScatteringModel(scattering),
        radar_system=radar,
        antenna_positions_m=antenna_positions,
        add_noise=not args.no_noise,
        rng=np.random.default_rng(args.seed),
    )
    expected_shape = (shape[0] * shape[1], radar.num_chirps, radar.num_samples)
    if cube.shape != expected_shape or not np.all(np.isfinite(cube)):
        raise AssertionError(f"Invalid cube: shape={cube.shape}, expected={expected_shape}")

    maps = compute_radar_maps(
        cube,
        radar.params_array(),
        array_shape=shape,
        angle_fft_size=args.angle_fft_size,
        remove_clutter=True,
        spacing_wavelengths=args.spacing_wavelengths,
    )
    rd_peak = peak_2d(maps["rd_db"], maps["range_axis_m"], maps["velocity_axis_m_s"])
    ra_peak = peak_2d(maps["ra_db"], maps["range_axis_m"], maps["azimuth_axis_deg"])
    re_peak = peak_2d(maps["re_db"], maps["range_axis_m"], maps["elevation_axis_deg"])
    expected_azimuth_direction = math.degrees(math.asin(float(direction[1])))
    expected_elevation_direction = math.degrees(math.asin(float(direction[2])))

    npz_path = output_dir / "radar_000000.npz"
    np.savez(
        npz_path,
        radar_data=cube,
        timestamp=np.asarray(0.0),
        gt_pos=args.range_m * direction,
        gt_vel=-args.velocity_mps * direction,
        radar_params=radar.params_array(),
        radar_bandwidth_hz=np.asarray(radar.bandwidth),
        radar_array_shape=np.asarray(shape, dtype=np.int32),
        radar_array_pos=antenna_positions.astype(np.float32),
        radar_spacing_wavelengths=np.asarray(args.spacing_wavelengths),
        radar_mount_yaw_pitch_roll_deg=np.zeros(3, dtype=np.float64),
        radar_idle_time_s=np.asarray(radar.idle_time),
        radar_add_noise=np.asarray(not args.no_noise),
        radar_noise_floor_dbm=np.asarray(radar.noise_floor_effective_dbm),
        radar_noise_power_w=np.asarray(radar.noise_power_w),
        radar_noise_std=np.asarray(radar.noise_std),
        radar_tx_power_dbm=np.asarray(radar.tx_power_dbm),
        radar_tx_gain_db=np.asarray(radar.tx_gain_db),
        radar_rx_gain_db=np.asarray(radar.rx_gain_db),
        rcs_model=np.asarray(str(h5_path)),
        rcs_frequency_hz=np.asarray(rcs_model.frequency_hz),
        rcs_incident_polarization=np.asarray("theta"),
        rcs_scattering_component=np.asarray("theta"),
        radar_scattering_amplitude=np.asarray(scattering),
        radar_signal_unit=np.asarray("sqrt_w_after_range_fft"),
        radar_scattering_normalization=np.asarray(
            "a_squared_times_4pi_over_lambda_times_complex_F"
        ),
    )
    visualize_file(
        npz_path,
        output_dir,
        show_gt=True,
        bs_position=(0.0, 0.0, 0.0),
    )

    range_resolution = C_M_S / (2.0 * radar.bandwidth)
    velocity_resolution = radar.lambda_c / (
        2.0 * radar.num_chirps * radar.effective_chirp_interval
    )
    angle_step = float(np.max(np.diff(maps["azimuth_axis_deg"])[1:-1]))
    checks = {
        "shape": cube.shape == expected_shape,
        "range": abs(rd_peak[0] - args.range_m) <= range_resolution,
        "velocity": abs(rd_peak[1] - args.velocity_mps) <= velocity_resolution,
        "azimuth": abs(ra_peak[1] - expected_azimuth_direction) <= angle_step,
        "elevation": abs(re_peak[1] - expected_elevation_direction) <= angle_step,
    }
    report = {
        "cube_shape": list(cube.shape),
        "scattering_amplitude": [scattering.real, scattering.imag],
        "rcs_m2": sigma_m2,
        "noise_floor_dbm": radar.noise_floor_effective_dbm,
        "expected": {
            "range_m": args.range_m,
            "velocity_m_s": args.velocity_mps,
            "azimuth_direction_deg": expected_azimuth_direction,
            "elevation_direction_deg": expected_elevation_direction,
        },
        "peaks": {
            "rd": {"range_m": rd_peak[0], "velocity_m_s": rd_peak[1]},
            "ra": {"range_m": ra_peak[0], "azimuth_direction_deg": ra_peak[1]},
            "re": {"range_m": re_peak[0], "elevation_direction_deg": re_peak[1]},
        },
        "checks": checks,
    }
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=" * 84)
    print("28 GHZ PRODUCTION RADAR VISUAL CALIBRATION")
    print("=" * 84)
    print(f"F_theta:    {scattering.real:+.9e}{scattering.imag:+.9e}j m")
    print(f"RCS theta:  {sigma_m2:.9e} m^2")
    print(f"Cube:       {cube.shape}, dtype={cube.dtype}, finite={np.all(np.isfinite(cube))}")
    print(f"Cube values:{cube.reshape(-1)[:8]}")
    print(f"RD peak:    range={rd_peak[0]:.6f} m, velocity={rd_peak[1]:.6f} m/s")
    print(f"RA peak:    range={ra_peak[0]:.6f} m, direction={ra_peak[1]:.6f} deg")
    print(f"RE peak:    range={re_peak[0]:.6f} m, direction={re_peak[1]:.6f} deg")
    print(f"Noise:      floor={radar.noise_floor_effective_dbm:.3f} dBm, std={radar.noise_std:.9e}")
    print(f"Report:     {report_path}")
    print("Checks:     " + ", ".join(f"{name}={'PASS' if value else 'FAIL'}" for name, value in checks.items()))
    if not all(checks.values()):
        raise AssertionError(f"Calibration failed: {checks}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


SERVER_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = SERVER_ROOT / "configs" / "scenarios.json"


def _set_runtime_env(args: argparse.Namespace) -> None:
    if getattr(args, "config", None):
        os.environ["LAMBDA_SCENARIOS_CONFIG"] = str(Path(args.config).resolve())
    if getattr(args, "scenario", None):
        os.environ["LAMBDA_SCENARIO"] = args.scenario
    if getattr(args, "profile", None):
        os.environ["LAMBDA_SUBCARRIER_PROFILE"] = str(args.profile)


def _load_scenarios(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def list_scenarios(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve() if args.config else CONFIG_PATH
    raw = _load_scenarios(config_path)
    default = raw.get("default_scenario")
    common = raw.get("common", {})
    for name, cfg in sorted(raw.get("scenarios", {}).items()):
        merged = _deep_merge(common, cfg)
        marker = " (default)" if name == default else ""
        description = merged.get("description", "")
        carrier_ghz = float(merged.get("carrier_frequency", 0.0)) / 1e9
        tx_shape = "x".join(str(x) for x in merged.get("tx_array_shape", [])) or "default"
        rx_shape = "x".join(str(x) for x in merged.get("rx_array_shape", [])) or "default"
        profile = merged.get("default_subcarrier_profile", "default")
        print(
            f"{name}{marker}: {description} "
            f"carrier={carrier_ghz:g}GHz, tx={tx_shape}, rx={rx_shape}, subcarrier={profile}"
        )


def run_read_csi(args: argparse.Namespace) -> None:
    from lambda_rf.tools.read_csi import load_csi_npz, print_file_summary, write_csv

    summary = load_csi_npz(args.npz_path)
    top = summary["num_paths"] if args.top == 0 else args.top
    print_file_summary(summary, top=top)
    if args.csv:
        write_csv(summary, args.csv)


def run_mimo_ofdm_csi(args: argparse.Namespace) -> None:
    _set_runtime_env(args)

    from lambda_rf.generate_mimo_ofdm_csi import run

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tx_shape=args.tx_shape,
        rx_shape=args.rx_shape,
        spacing_wavelengths=args.spacing_wavelengths,
        profile=args.profile,
        num_subcarriers=args.num_subcarriers,
        subcarrier_spacing_hz=args.subcarrier_spacing,
        skip_existing=args.skip_existing,
        start_frame=args.start_frame,
        limit=args.limit,
        tx_orientation_pose=args.tx_orientation_pose,
        rx_orientation_pose=args.rx_orientation_pose,
        array_model=args.array_model,
    )


def run_radar(args: argparse.Namespace) -> None:
    _set_runtime_env(args)
    from lambda_rf.generate_radar import run

    run(
        csi_dir=args.input_dir,
        output_dir=args.output_dir,
        imu_dir=args.imu_dir,
        carrier_frequency_hz=args.carrier_frequency,
        bandwidth_hz=args.bandwidth,
        sample_rate_hz=args.sample_rate,
        chirp_duration_s=args.chirp_duration,
        chirp_interval_s=args.chirp_interval,
        idle_time_s=args.idle_time,
        num_chirps=args.num_chirps,
        noise_floor_dbm=args.noise_floor_dbm,
        noise_figure_db=args.noise_figure_db,
        noise_bandwidth_hz=args.noise_bandwidth,
        noise_seed=args.noise_seed,
        tx_power_dbm=args.tx_power_dbm,
        tx_gain_db=args.tx_gain_db,
        rx_gain_db=args.rx_gain_db,
        array_shape=args.array_shape,
        spacing_wavelengths=args.spacing_wavelengths,
        radar_yaw_deg=args.radar_yaw,
        radar_pitch_deg=args.radar_pitch,
        radar_roll_deg=args.radar_roll,
        add_noise=args.add_noise,
        rcs_model_path=args.rcs_model,
        rcs_component=args.rcs_component,
        array_model=args.array_model,
        skip_existing=args.skip_existing,
        start_frame=args.start_frame,
        limit=args.limit,
        frame_step=args.frame_step,
    )


def run_radar_vis(args: argparse.Namespace) -> None:
    _set_runtime_env(args)
    from lambda_rf.visualize_radar import run

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        array_shape=args.array_shape,
        angle_fft_size=args.angle_fft_size,
        remove_clutter=not args.keep_clutter,
        frame_step=args.frame_step,
        start_frame=args.start_frame,
        limit=args.limit,
        show_gt=args.show_gt,
        bs_position=args.bs_position,
        radar_yaw_deg=args.radar_yaw,
        radar_pitch_deg=args.radar_pitch,
        radar_roll_deg=args.radar_roll,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lambda-rf")
    parser.add_argument("--config", help="Path to a scenario-style utility config JSON.")

    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list-scenarios", help="List configured utility scenarios.")
    list_cmd.set_defaults(func=list_scenarios)

    read_csi_cmd = sub.add_parser("read-csi", help="Read and summarize one released CSI NPZ file.")
    read_csi_cmd.add_argument("npz_path", help="Path to csi_*.npz.")
    read_csi_cmd.add_argument("--top", type=int, default=25, help="Number of strongest paths to print. Use 0 for all.")
    read_csi_cmd.add_argument("--csv", default=None, help="Optional CSV output path.")
    read_csi_cmd.set_defaults(func=run_read_csi)

    mimo_ofdm_cmd = sub.add_parser("mimo-ofdm-csi", help="Generate final MIMO OFDM CSI from path-level CSI.")
    mimo_ofdm_cmd.add_argument("--scenario", help="Scenario key in configs/scenarios.json.")
    mimo_ofdm_cmd.add_argument("--input-dir", required=True, help="Input directory containing path-level csi_*.npz.")
    mimo_ofdm_cmd.add_argument("--output-dir", help="Output directory for final MIMO OFDM csi_*.npz files.")
    mimo_ofdm_cmd.add_argument("--tx-shape", help="TX MIMO array shape as ROWS,COLS or ROWSxCOLS. Defaults to config.")
    mimo_ofdm_cmd.add_argument("--rx-shape", help="RX MIMO array shape as ROWS,COLS or ROWSxCOLS. Defaults to config.")
    mimo_ofdm_cmd.add_argument("--spacing-wavelengths", type=float, default=0.5, help="MIMO element spacing in wavelengths.")
    mimo_ofdm_cmd.add_argument("--tx-orientation-pose", help="Pose JSON whose quaternion defines the TX MIMO array rotation.")
    mimo_ofdm_cmd.add_argument("--rx-orientation-pose", help="Pose JSON whose quaternion defines the RX MIMO array rotation.")
    mimo_ofdm_cmd.add_argument(
        "--array-model",
        choices=["far-field", "spherical-wave"],
        default="far-field",
        help="Array wavefront model. spherical-wave uses per-element path lengths from CSI path_vertices.",
    )
    mimo_ofdm_cmd.add_argument("--profile", help="OFDM subcarrier profile name, for example sub6_30k_1024.")
    mimo_ofdm_cmd.add_argument("--num-subcarriers", type=int, help="Override number of OFDM subcarriers.")
    mimo_ofdm_cmd.add_argument("--subcarrier-spacing", type=float, help="Override OFDM subcarrier spacing in Hz.")
    mimo_ofdm_cmd.add_argument("--start-frame", type=int, help="Only process frames at or after this index.")
    mimo_ofdm_cmd.add_argument("--limit", type=int, help="Maximum number of files to process.")
    mimo_ofdm_cmd.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist.")
    mimo_ofdm_cmd.set_defaults(func=run_mimo_ofdm_csi)

    radar_cmd = sub.add_parser("radar", help="Generate FMCW radar cubes from released path-level CSI.")
    radar_cmd.add_argument("--scenario", help="Scenario key in configs/scenarios.json.")
    radar_cmd.add_argument("--input-dir", required=True, help="Input directory containing csi_*.npz.")
    radar_cmd.add_argument("--output-dir", help="Output directory for radar_*.npz files.")
    radar_cmd.add_argument("--imu-dir", help="Optional directory containing IMU or drone pose orientation JSON files.")
    radar_cmd.add_argument("--carrier-frequency", type=float, help="Carrier frequency in Hz. Defaults to CSI/config.")
    radar_cmd.add_argument("--bandwidth", type=float, help="FMCW bandwidth in Hz.")
    radar_cmd.add_argument("--sample-rate", type=float, help="ADC sample rate in Hz.")
    radar_cmd.add_argument("--chirp-duration", type=float, help="Chirp duration in seconds.")
    radar_cmd.add_argument("--chirp-interval", type=float, help="Chirp-to-chirp interval/PRI in seconds.")
    radar_cmd.add_argument("--idle-time", type=float, help="Idle gap after each chirp in seconds. Ignored when --chirp-interval is set.")
    radar_cmd.add_argument("--num-chirps", type=int, help="Number of chirps per frame.")
    radar_cmd.add_argument("--noise-floor-dbm", type=float, help="Noise floor in dBm.")
    radar_cmd.add_argument("--noise-figure-db", type=float, help="Receiver noise figure in dB when thermal noise is derived.")
    radar_cmd.add_argument("--noise-bandwidth", type=float, help="Noise bandwidth in Hz. Defaults to sample rate.")
    radar_cmd.add_argument("--noise-seed", type=int, help="Seed for deterministic receiver noise.")
    radar_cmd.add_argument("--tx-power-dbm", type=float, help="Radar transmit power in dBm.")
    radar_cmd.add_argument("--tx-gain-db", type=float, help="Transmit antenna/front-end gain in dB.")
    radar_cmd.add_argument("--rx-gain-db", type=float, help="Receive antenna/front-end gain in dB.")
    radar_cmd.add_argument("--array-shape", help="Radar virtual array shape as ROWS,COLS or ROWSxCOLS.")
    radar_cmd.add_argument("--spacing-wavelengths", type=float, help="Array spacing in wavelengths.")
    radar_cmd.add_argument("--radar-yaw", type=float, help="Radar mount yaw in degrees.")
    radar_cmd.add_argument("--radar-pitch", type=float, help="Radar mount pitch in degrees.")
    radar_cmd.add_argument("--radar-roll", type=float, help="Radar mount roll in degrees.")
    radar_cmd.add_argument(
        "--array-model",
        choices=["far-field", "spherical-wave"],
        default="far-field",
        help="Radar array wavefront model. spherical-wave uses per-antenna path lengths from CSI path_vertices.",
    )
    noise_group = radar_cmd.add_mutually_exclusive_group()
    noise_group.add_argument("--add-noise", dest="add_noise", action="store_true", default=None, help="Add complex Gaussian receiver noise.")
    noise_group.add_argument("--no-noise", dest="add_noise", action="store_false", help="Disable receiver noise.")
    radar_cmd.add_argument("--rcs-model", help="Explicit frequency-matched FEKO H5 model. Defaults to the bundled 28/77 GHz model.")
    radar_cmd.add_argument(
        "--rcs-component",
        choices=["theta", "phi"],
        default="theta",
        help="Scattered field component for theta-linear incidence; theta is co-polar and phi is cross-polar.",
    )
    radar_cmd.add_argument("--start-frame", type=int, help="Only process frames at or after this index.")
    radar_cmd.add_argument("--limit", type=int, help="Maximum number of files to process.")
    radar_cmd.add_argument("--frame-step", type=int, default=1, help="Process every Nth CSI file.")
    radar_cmd.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist.")
    radar_cmd.set_defaults(func=run_radar)

    radar_vis_cmd = sub.add_parser("radar-vis", help="Render RD/RA/RE images from radar_*.npz files.")
    radar_vis_cmd.add_argument("--scenario", help="Scenario key in configs/scenarios.json.")
    radar_vis_cmd.add_argument("--input-dir", required=True, help="Input directory containing radar_*.npz.")
    radar_vis_cmd.add_argument("--output-dir", help="Output directory for RD/RA/RE images.")
    radar_vis_cmd.add_argument("--array-shape", help="Radar virtual array shape as ROWS,COLS or ROWSxCOLS.")
    radar_vis_cmd.add_argument("--angle-fft-size", type=int, default=64, help="FFT size for azimuth/elevation maps.")
    radar_vis_cmd.add_argument("--keep-clutter", action="store_true", help="Disable slow-time mean subtraction.")
    radar_vis_cmd.add_argument("--frame-step", type=int, default=1, help="Render every Nth radar file.")
    radar_vis_cmd.add_argument("--start-frame", type=int, help="Only render frames at or after this index.")
    radar_vis_cmd.add_argument("--limit", type=int, help="Maximum number of files to render.")
    radar_vis_cmd.add_argument("--show-gt", action="store_true", help="Overlay GT if radar files include gt_pos/gt_vel.")
    radar_vis_cmd.add_argument("--bs-position", help="Base station position as x,y,z in meters, required for GT overlay.")
    radar_vis_cmd.add_argument("--radar-yaw", type=float, help="Radar mount yaw in degrees for GT overlay.")
    radar_vis_cmd.add_argument("--radar-pitch", type=float, help="Radar mount pitch in degrees for GT overlay.")
    radar_vis_cmd.add_argument("--radar-roll", type=float, help="Radar mount roll in degrees for GT overlay.")
    radar_vis_cmd.set_defaults(func=run_radar_vis)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

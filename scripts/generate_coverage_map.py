from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SERVER_ROOT = SCRIPT_PATH.parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def expand_config_path(path_value: str) -> str:
    variables = {
        "SERVER_ROOT": str(SERVER_ROOT),
        "LAMBDA_ROOT": str(SERVER_ROOT),
        "SIONNART_ROOT": str(SERVER_ROOT.parent),
        "PROJECT_ROOT": str(SERVER_ROOT.parent.parent),
    }
    expanded = str(path_value)
    for name, replacement in variables.items():
        expanded = expanded.replace("${" + name + "}", replacement)
    return os.path.expandvars(os.path.expanduser(expanded))


def load_scenario(config_path: Path, scenario: str) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    common = raw.get("common", {})
    scenarios = raw.get("scenarios", {})
    if scenario not in scenarios:
        known = ", ".join(sorted(scenarios))
        raise KeyError(f"Unknown scenario {scenario!r}. Known scenarios: {known}")
    return deep_merge(common, scenarios[scenario])


def ue_cm_to_m(values: list[float] | tuple[float, float, float]) -> list[float]:
    return [float(values[0]) * 0.01, float(values[1]) * 0.01, float(values[2]) * 0.01]


def parse_xyz(value: str) -> list[float]:
    parts = [p.strip() for p in value.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected X,Y,Z")
    return [float(p) for p in parts]


def load_site_positions(site_csv: Path, role: str, names: list[str] | None) -> list[tuple[str, list[float]]]:
    wanted = set(names or [])
    positions: list[tuple[str, list[float]]] = []
    with site_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("role") != role:
                continue
            name = row.get("name", "")
            if wanted and name not in wanted:
                continue
            positions.append(
                (
                    name,
                    [
                        float(row["x_m"]),
                        float(row["y_m"]),
                        float(row["z_m"]),
                    ],
                )
            )
    if not positions:
        raise ValueError(f"No site positions found for role={role!r}, names={names!r} in {site_csv}")
    return positions


def noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def path_gain_linear(paths) -> float:
    a_real, a_imag = paths.a
    valid = np.array(paths.valid, dtype=np.bool_)
    power = np.array(a_real) ** 2 + np.array(a_imag) ** 2
    if valid.shape == power.shape:
        power = power * valid
    total = float(np.sum(power))
    return max(total, 0.0)


def make_grid(
    center_x: float,
    center_y: float,
    size_m: float,
    step_m: float,
    altitude_m: float,
) -> np.ndarray:
    half = size_m / 2.0
    xs = np.arange(center_x - half, center_x + half + 1e-6, step_m)
    ys = np.arange(center_y - half, center_y + half + 1e-6, step_m)
    points = []
    for y in ys:
        for x in xs:
            points.append([float(x), float(y), float(altitude_m)])
    return np.array(points, dtype=np.float32)


def write_coverage_outputs(
    out_dir: Path,
    rows: list[dict[str, float | int | str]],
    points: np.ndarray,
    point_indices: np.ndarray,
    path_gain_values: np.ndarray,
    rx_power_values: np.ndarray,
    sinr_values: np.ndarray,
    best_tx_indices: np.ndarray,
    tx_positions: list[tuple[str, list[float]]],
    frequency_hz: float,
    noise_dbm: float,
    scene_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    csv_path = out_dir / "coverage_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "point_index",
                "x_m",
                "y_m",
                "z_m",
                "path_gain_linear",
                "rx_power_dbm",
                "sinr_db",
                "best_tx",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    npz_path = out_dir / "coverage_map.npz"
    np.savez(
        npz_path,
        points_m=points,
        point_indices=point_indices.astype(np.int64),
        full_point_count=np.int64(args.full_point_count),
        path_gain_linear=path_gain_values,
        rx_power_dbm=rx_power_values,
        sinr_db=sinr_values,
        best_tx_index=best_tx_indices,
        tx_names=np.array([name for name, _ in tx_positions]),
        tx_positions_m=np.array([pos for _, pos in tx_positions], dtype=np.float32),
        tx_pos_m=np.array(tx_positions[0][1], dtype=np.float32),
        altitude_m=np.float64(args.altitude_m),
        frequency_hz=np.float64(frequency_hz),
        bandwidth_hz=np.float64(args.bandwidth_hz),
        noise_floor_dbm=np.float64(noise_dbm),
        scenario=np.array(args.scenario),
        scene_path=np.array(str(scene_path)),
        shard_index=np.int32(args.shard_index),
        num_shards=np.int32(args.num_shards),
    )
    return csv_path, npz_path


def write_heatmap(
    out_dir: Path,
    points: np.ndarray,
    sinr_values: np.ndarray,
    tx_positions: list[tuple[str, list[float]]],
    frequency_hz: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    if len(xs) * len(ys) != len(points):
        return

    grid = sinr_values.reshape(len(ys), len(xs))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        grid,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap="RdYlGn",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    ax.scatter(
        [pos[0] for _, pos in tx_positions],
        [pos[1] for _, pos in tx_positions],
        marker="^",
        s=120,
        c="blue",
        label="BS",
    )
    for name, pos in tx_positions:
        ax.text(pos[0] + 8, pos[1] + 8, name, fontsize=8, color="blue")
    ax.set_title(f"Sionna SINR coverage: {args.scenario}, {frequency_hz/1e9:.1f} GHz, UAV z={args.altitude_m:g} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax, label="SINR (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_sinr_heatmap.png", dpi=250)
    plt.close(fig)


def compute_coverage(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mitsuba as mi

    mi.set_variant(args.mi_variant)

    os.environ["LAMBDA_SCENARIO"] = args.scenario
    cfg = load_scenario(args.config, args.scenario)
    sim_name = args.sim or cfg.get("active_sim_name")
    if sim_name:
        os.environ["LAMBDA_SIM"] = sim_name

    from lambda_rf import config as lambda_config
    from lambda_rf.config import SIM_CONFIGS
    from lambda_rf.utils.antenna import setup_tx_rx_arrays
    from lambda_rf.utils.materials import override_materials_and_roughness
    from sionna.rt import PathSolver, Receiver, Transmitter, load_scene

    scene_path = Path(expand_config_path(args.xml_scene_path)) if args.xml_scene_path else Path(expand_config_path(cfg["xml_scene_path"]))
    sim = next((item for item in SIM_CONFIGS if item["name"] == sim_name), None)
    if sim is None:
        known = ", ".join(item["name"] for item in SIM_CONFIGS)
        raise KeyError(f"Unknown sim {sim_name!r}. Known sims: {known}")

    frequency_hz = float(sim["carrier_frequency"])
    lambda_config.CARRIER_FREQUENCY = frequency_hz
    if args.sites_csv:
        bs_names = [part.strip() for part in args.base_station_names.split(",") if part.strip()] if args.base_station_names else None
        tx_positions = load_site_positions(args.sites_csv, role="base_station", names=bs_names)
    else:
        tx_positions = [("BS_config", args.tx_pos_m or ue_cm_to_m(cfg["bs_ue"]))]

    if args.center_m:
        center_x, center_y, _ = args.center_m
    else:
        start_m = ue_cm_to_m(cfg.get("uav_start_ue", cfg["bs_ue"]))
        center_x, center_y = start_m[0], start_m[1]

    points = make_grid(
        center_x=center_x,
        center_y=center_y,
        size_m=args.size_m,
        step_m=args.step_m,
        altitude_m=args.altitude_m,
    )
    if args.limit and args.limit > 0:
        points = points[: args.limit]
    point_indices = np.arange(len(points), dtype=np.int64)
    args.full_point_count = len(points)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        keep = point_indices % args.num_shards == args.shard_index
        points = points[keep]
        point_indices = point_indices[keep]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_dbm = noise_floor_dbm(args.bandwidth_hz, args.noise_figure_db)
    if len(points) == 0:
        rows: list[dict[str, float | int | str]] = []
        empty_float = np.empty((0,), dtype=np.float32)
        empty_int = np.empty((0,), dtype=np.int32)
        csv_path, npz_path = write_coverage_outputs(
            out_dir=out_dir,
            rows=rows,
            points=points,
            point_indices=point_indices,
            path_gain_values=empty_float,
            rx_power_values=empty_float,
            sinr_values=empty_float,
            best_tx_indices=empty_int,
            tx_positions=tx_positions,
            frequency_hz=frequency_hz,
            noise_dbm=noise_dbm,
            scene_path=scene_path,
            args=args,
        )
        print(f"[Done] Empty shard CSV: {csv_path}")
        print(f"[Done] Empty shard NPZ: {npz_path}")
        return

    scene = load_scene(str(scene_path), merge_shapes=True)
    scene.frequency = frequency_hz
    override_materials_and_roughness(scene)
    setup_tx_rx_arrays(scene)

    tx = Transmitter("coverage_tx", position=tx_positions[0][1], display_radius=2.0)
    rx = Receiver("coverage_rx", position=points[0].tolist(), display_radius=2.0)
    scene.add(tx)
    scene.add(rx)
    solver = PathSolver()

    rows: list[dict[str, float | int | str]] = []
    sinr_values = np.full((len(points),), np.nan, dtype=np.float32)
    rx_power_values = np.full((len(points),), np.nan, dtype=np.float32)
    path_gain_values = np.full((len(points),), 0.0, dtype=np.float32)
    best_tx_indices = np.full((len(points),), -1, dtype=np.int32)

    for i, p in enumerate(points):
        point_index = int(point_indices[i])
        rx.position = p.tolist()
        best_tx_name = ""
        best_tx_idx = -1
        best_gain = 0.0
        best_rx_power_dbm = -300.0
        best_sinr_db = -300.0

        for tx_idx, (tx_name, tx_pos_m) in enumerate(tx_positions):
            tx.position = tx_pos_m
            tx.look_at(rx)
            paths = solver(
                scene=scene,
                max_depth=args.max_depth,
                los=True,
                specular_reflection=True,
                diffuse_reflection=args.diffuse,
                diffraction=args.diffraction,
                refraction=False,
                synthetic_array=False,
                seed=args.seed + point_index + 100_000 * tx_idx,
            )
            gain = path_gain_linear(paths)
            if gain > 0.0:
                rx_power_dbm = args.tx_power_dbm + 10.0 * math.log10(gain)
                sinr_db = rx_power_dbm - noise_dbm
            else:
                rx_power_dbm = -300.0
                sinr_db = -300.0

            if sinr_db > best_sinr_db:
                best_tx_name = tx_name
                best_tx_idx = tx_idx
                best_gain = gain
                best_rx_power_dbm = rx_power_dbm
                best_sinr_db = sinr_db

        path_gain_values[i] = best_gain
        rx_power_values[i] = best_rx_power_dbm
        sinr_values[i] = best_sinr_db
        best_tx_indices[i] = best_tx_idx
        rows.append(
            {
                "point_index": point_index,
                "x_m": float(p[0]),
                "y_m": float(p[1]),
                "z_m": float(p[2]),
                "path_gain_linear": best_gain,
                "rx_power_dbm": best_rx_power_dbm,
                "sinr_db": best_sinr_db,
                "best_tx": best_tx_name,
            }
        )
        if i % max(1, args.progress_every) == 0:
            print(
                f"[{i:05d}/{len(points)} shard={args.shard_index}/{args.num_shards} "
                f"global={point_index:05d}] x={p[0]:.1f}, y={p[1]:.1f}, "
                f"sinr={best_sinr_db:.2f} dB, best_tx={best_tx_name}"
            )

    csv_path, npz_path = write_coverage_outputs(
        out_dir=out_dir,
        rows=rows,
        points=points,
        point_indices=point_indices,
        path_gain_values=path_gain_values,
        rx_power_values=rx_power_values,
        sinr_values=sinr_values,
        best_tx_indices=best_tx_indices,
        tx_positions=tx_positions,
        frequency_hz=frequency_hz,
        noise_dbm=noise_dbm,
        scene_path=scene_path,
        args=args,
    )

    if args.num_shards == 1:
        write_heatmap(out_dir, points, sinr_values, tx_positions, frequency_hz, args)

    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] NPZ: {npz_path}")
    print(f"[Done] Output dir: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Sionna RT SINR coverage map for UAV routing.")
    parser.add_argument("--config", type=Path, default=SERVER_ROOT / "configs" / "scenarios.json")
    parser.add_argument("--scenario", default="block1_4p9g")
    parser.add_argument("--xml-scene-path", default=None, help="Override XML scene path, e.g. region_files/San Francisco/Full SF/fullsf_0-10G.xml.")
    parser.add_argument("--sim", default=None, help="Override simulation profile, e.g. sub6_4p9_clear or mmwave_60p0_clear.")
    parser.add_argument("--output-dir", type=Path, default=SERVER_ROOT / "data_export" / "coverage_map")
    parser.add_argument("--mi-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--tx-pos-m", type=parse_xyz, default=None, help="Override BS/TX position in Sionna meters: X,Y,Z.")
    parser.add_argument("--sites-csv", type=Path, default=None, help="CSV with selected sites; base_station rows are used as TXs.")
    parser.add_argument("--base-station-names", default=None, help="Comma-separated base station names from --sites-csv, e.g. BS1,BS2,BS3.")
    parser.add_argument("--center-m", type=parse_xyz, default=None, help="Grid center in Sionna meters: X,Y,Z. Z is ignored.")
    parser.add_argument("--size-m", type=float, default=2000.0)
    parser.add_argument("--step-m", type=float, default=50.0)
    parser.add_argument("--altitude-m", type=float, default=50.0)
    parser.add_argument("--tx-power-dbm", type=float, default=30.0)
    parser.add_argument("--bandwidth-hz", type=float, default=20e6)
    parser.add_argument("--noise-figure-db", type=float, default=7.0)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--diffuse", action="store_true")
    parser.add_argument("--diffraction", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process first N grid points.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split grid points into this many deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index in [0, num_shards).")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--vmin", type=float, default=-20.0)
    parser.add_argument("--vmax", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    compute_coverage(args)


if __name__ == "__main__":
    main()

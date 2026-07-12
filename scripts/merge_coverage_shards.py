from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def shard_path(input_root: Path, shard_index: int) -> Path:
    return input_root / f"shard_{shard_index:03d}" / "coverage_map.npz"


def load_shards(input_root: Path, num_shards: int) -> list[np.lib.npyio.NpzFile]:
    shards = []
    missing = []
    for shard_index in range(num_shards):
        path = shard_path(input_root, shard_index)
        if not path.exists():
            missing.append(path)
            continue
        shards.append(np.load(path, allow_pickle=True))
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing shard coverage files:\n{formatted}")
    return shards


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        return value.item()
    return value


def reconstruct(shards: list[np.lib.npyio.NpzFile]) -> dict[str, np.ndarray]:
    first = shards[0]
    full_count = int(scalar(first, "full_point_count", 0))
    if full_count <= 0:
        full_count = int(sum(len(shard["points_m"]) for shard in shards))

    points = np.full((full_count, 3), np.nan, dtype=np.float32)
    path_gain = np.full((full_count,), np.nan, dtype=np.float32)
    rx_power = np.full((full_count,), np.nan, dtype=np.float32)
    sinr = np.full((full_count,), np.nan, dtype=np.float32)
    best_tx = np.full((full_count,), -1, dtype=np.int32)
    seen = np.zeros((full_count,), dtype=np.bool_)

    for shard in shards:
        indices = np.asarray(shard["point_indices"], dtype=np.int64)
        if len(indices) == 0:
            continue
        if indices.min() < 0 or indices.max() >= full_count:
            raise ValueError("Shard contains point_indices outside full_point_count.")
        if np.any(seen[indices]):
            raise ValueError("Duplicate point_indices detected across shards.")
        seen[indices] = True
        points[indices] = np.asarray(shard["points_m"], dtype=np.float32)
        path_gain[indices] = np.asarray(shard["path_gain_linear"], dtype=np.float32)
        rx_power[indices] = np.asarray(shard["rx_power_dbm"], dtype=np.float32)
        sinr[indices] = np.asarray(shard["sinr_db"], dtype=np.float32)
        best_tx[indices] = np.asarray(shard["best_tx_index"], dtype=np.int32)

    missing = np.where(~seen)[0]
    if len(missing):
        preview = ", ".join(str(int(i)) for i in missing[:20])
        raise ValueError(f"Missing {len(missing)} point indices after merge, first missing: {preview}")

    return {
        "points_m": points,
        "point_indices": np.arange(full_count, dtype=np.int64),
        "path_gain_linear": path_gain,
        "rx_power_dbm": rx_power,
        "sinr_db": sinr,
        "best_tx_index": best_tx,
    }


def write_csv(output_dir: Path, merged: dict[str, np.ndarray], tx_names: np.ndarray) -> Path:
    csv_path = output_dir / "coverage_points.csv"
    points = merged["points_m"]
    best_tx_indices = merged["best_tx_index"]
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
        for idx, point in enumerate(points):
            best_idx = int(best_tx_indices[idx])
            best_name = str(tx_names[best_idx]) if 0 <= best_idx < len(tx_names) else ""
            writer.writerow(
                {
                    "point_index": idx,
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                    "z_m": float(point[2]),
                    "path_gain_linear": float(merged["path_gain_linear"][idx]),
                    "rx_power_dbm": float(merged["rx_power_dbm"][idx]),
                    "sinr_db": float(merged["sinr_db"][idx]),
                    "best_tx": best_name,
                }
            )
    return csv_path


def write_npz(output_dir: Path, merged: dict[str, np.ndarray], first: np.lib.npyio.NpzFile, num_shards: int) -> Path:
    npz_path = output_dir / "coverage_map.npz"
    np.savez(
        npz_path,
        **merged,
        full_point_count=np.int64(len(merged["points_m"])),
        tx_names=first["tx_names"],
        tx_positions_m=first["tx_positions_m"],
        tx_pos_m=first["tx_pos_m"],
        altitude_m=first["altitude_m"],
        frequency_hz=first["frequency_hz"],
        bandwidth_hz=first["bandwidth_hz"],
        noise_floor_dbm=first["noise_floor_dbm"],
        scenario=first["scenario"],
        scene_path=first["scene_path"],
        shard_index=np.int32(-1),
        num_shards=np.int32(num_shards),
        merged_from_shards=np.array(True),
    )
    return npz_path


def write_heatmap(output_dir: Path, merged: dict[str, np.ndarray], first: np.lib.npyio.NpzFile, vmin: float, vmax: float) -> None:
    points = merged["points_m"]
    sinr = merged["sinr_db"]
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    if len(xs) * len(ys) != len(points):
        return

    grid = sinr.reshape(len(ys), len(xs))
    tx_positions = np.asarray(first["tx_positions_m"], dtype=np.float32)
    tx_names = first["tx_names"]
    scenario = str(first["scenario"])
    frequency_hz = float(first["frequency_hz"])
    altitude_m = float(first["altitude_m"])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        grid,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(tx_positions[:, 0], tx_positions[:, 1], marker="^", s=120, c="blue", label="BS")
    for name, pos in zip(tx_names, tx_positions):
        ax.text(float(pos[0]) + 8, float(pos[1]) + 8, str(name), fontsize=8, color="blue")
    ax.set_title(f"Sionna SINR coverage: {scenario}, {frequency_hz/1e9:.1f} GHz, UAV z={altitude_m:g} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax, label="SINR (dB)")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_sinr_heatmap.png", dpi=250)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Sionna coverage outputs for one altitude layer.")
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing shard_000, shard_001, ...")
    parser.add_argument("--output-dir", type=Path, required=True, help="Merged altitude output directory.")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--vmin", type=float, default=-20.0)
    parser.add_argument("--vmax", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = load_shards(args.input_root.resolve(), args.num_shards)
    merged = reconstruct(shards)
    csv_path = write_csv(output_dir, merged, shards[0]["tx_names"])
    npz_path = write_npz(output_dir, merged, shards[0], args.num_shards)
    write_heatmap(output_dir, merged, shards[0], args.vmin, args.vmax)

    print(f"[Done] merged shards: {args.num_shards}")
    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] NPZ: {npz_path}")
    print(f"[Done] Output dir: {output_dir}")


if __name__ == "__main__":
    main()

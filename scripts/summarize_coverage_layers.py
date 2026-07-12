from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SERVER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = SERVER_ROOT / "data_export" / "coverage" / "fullsf_4p9g_altitude_layers"


@dataclass
class CoverageLayer:
    altitude_m: float
    path: Path
    points_m: np.ndarray
    sinr_db: np.ndarray
    rx_power_dbm: np.ndarray
    best_tx_index: np.ndarray


def altitude_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def discover_layers(input_root: Path) -> list[tuple[float, Path]]:
    discovered: list[tuple[float, Path]] = []
    pattern = re.compile(r"alt_([0-9]+(?:\.[0-9]+)?|[0-9]+p[0-9]+)m$")
    for child in input_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        altitude = float(match.group(1).replace("p", "."))
        npz_path = child / "coverage_map.npz"
        if npz_path.exists():
            discovered.append((altitude, npz_path))
    return sorted(discovered, key=lambda item: item[0])


def requested_layers(input_root: Path, altitudes: list[float]) -> list[tuple[float, Path]]:
    pairs = []
    for altitude in altitudes:
        pairs.append((altitude, input_root / f"alt_{altitude_label(altitude)}m" / "coverage_map.npz"))
    return pairs


def load_layer(altitude: float, path: Path) -> CoverageLayer:
    data = np.load(path, allow_pickle=True)
    points_m = np.asarray(data["points_m"], dtype=np.float32)
    sinr_db = np.asarray(data["sinr_db"], dtype=np.float32)
    rx_power_dbm = (
        np.asarray(data["rx_power_dbm"], dtype=np.float32)
        if "rx_power_dbm" in data.files
        else np.full_like(sinr_db, np.nan, dtype=np.float32)
    )
    best_tx_index = (
        np.asarray(data["best_tx_index"], dtype=np.int32)
        if "best_tx_index" in data.files
        else np.full_like(sinr_db, -1, dtype=np.int32)
    )
    return CoverageLayer(
        altitude_m=float(altitude),
        path=path,
        points_m=points_m,
        sinr_db=sinr_db,
        rx_power_dbm=rx_power_dbm,
        best_tx_index=best_tx_index,
    )


def require_same_xy(layers: list[CoverageLayer]) -> np.ndarray:
    base_xy = layers[0].points_m[:, :2]
    for layer in layers[1:]:
        if layer.points_m.shape != layers[0].points_m.shape or not np.allclose(layer.points_m[:, :2], base_xy):
            raise ValueError(
                "Coverage layers do not share the same XY grid. Re-run them with identical center/size/step settings."
            )
    return base_xy


def finite(values: np.ndarray) -> np.ndarray:
    result = values[np.isfinite(values)]
    return result if len(result) else np.array([np.nan], dtype=np.float32)


def write_summary_csv(layers: list[CoverageLayer], output_path: Path) -> None:
    rows = []
    for layer in layers:
        values = finite(layer.sinr_db)
        rows.append(
            {
                "altitude_m": layer.altitude_m,
                "point_count": int(len(layer.sinr_db)),
                "sinr_mean_db": float(np.nanmean(values)),
                "sinr_p10_db": float(np.nanpercentile(values, 10)),
                "sinr_median_db": float(np.nanpercentile(values, 50)),
                "sinr_p90_db": float(np.nanpercentile(values, 90)),
                "outage_ratio_sinr_lt_0": float(np.nanmean(layer.sinr_db < 0.0)),
                "weak_ratio_sinr_lt_10": float(np.nanmean(layer.sinr_db < 10.0)),
                "good_ratio_sinr_ge_10": float(np.nanmean(layer.sinr_db >= 10.0)),
                "source_npz": str(layer.path),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_stack_npz(layers: list[CoverageLayer], xy_m: np.ndarray, output_path: Path) -> None:
    np.savez(
        output_path,
        altitudes_m=np.asarray([layer.altitude_m for layer in layers], dtype=np.float32),
        xy_m=xy_m.astype(np.float32),
        sinr_db=np.stack([layer.sinr_db for layer in layers]).astype(np.float32),
        rx_power_dbm=np.stack([layer.rx_power_dbm for layer in layers]).astype(np.float32),
        best_tx_index=np.stack([layer.best_tx_index for layer in layers]).astype(np.int32),
        source_npz=np.asarray([str(layer.path) for layer in layers]),
    )


def plot_summary(layers: list[CoverageLayer], output_path: Path) -> None:
    altitudes = np.asarray([layer.altitude_m for layer in layers], dtype=np.float32)
    p10 = np.asarray([np.nanpercentile(finite(layer.sinr_db), 10) for layer in layers])
    median = np.asarray([np.nanpercentile(finite(layer.sinr_db), 50) for layer in layers])
    outage = np.asarray([np.nanmean(layer.sinr_db < 0.0) for layer in layers])
    weak = np.asarray([np.nanmean(layer.sinr_db < 10.0) for layer in layers])

    fig, ax1 = plt.subplots(figsize=(8.8, 5.0), dpi=180)
    ax1.plot(altitudes, p10, marker="o", color="#2166ac", label="P10 SINR")
    ax1.plot(altitudes, median, marker="s", color="#1b7837", label="Median SINR")
    ax1.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax1.axhline(10.0, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("UAV altitude (m)")
    ax1.set_ylabel("SINR (dB)")
    ax1.grid(True, color="#e0e0e0", linewidth=0.45)

    ax2 = ax1.twinx()
    ax2.plot(altitudes, outage * 100.0, marker="^", color="#b2182b", label="Outage < 0 dB")
    ax2.plot(altitudes, weak * 100.0, marker="v", color="#ef8a62", label="Weak < 10 dB")
    ax2.set_ylabel("Area ratio (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.set_title("Coverage quality across UAV altitude layers")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_best_altitude_map(layers: list[CoverageLayer], xy_m: np.ndarray, output_path: Path) -> None:
    sinr_stack = np.stack([layer.sinr_db for layer in layers])
    best_idx = np.nanargmax(sinr_stack, axis=0)
    best_altitude = np.asarray([layers[idx].altitude_m for idx in best_idx], dtype=np.float32)
    xs = np.unique(xy_m[:, 0])
    ys = np.unique(xy_m[:, 1])
    if len(xs) * len(ys) != len(xy_m):
        return
    grid = best_altitude.reshape(len(ys), len(xs))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    im = ax.imshow(
        grid,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap="viridis",
        vmin=min(layer.altitude_m for layer in layers),
        vmax=max(layer.altitude_m for layer in layers),
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Best SINR altitude layer at each XY grid point")
    fig.colorbar(im, ax=ax, label="Best altitude (m)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multiple Sionna coverage maps generated at altitude layers.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--altitudes", nargs="*", type=float, default=None, help="Altitude list, e.g. 50 60 ... 120.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_paths = requested_layers(input_root, args.altitudes) if args.altitudes else discover_layers(input_root)
    missing = [path for _, path in layer_paths if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing coverage_map.npz files:\n{formatted}")

    layers = [load_layer(altitude, path) for altitude, path in layer_paths]
    if not layers:
        raise ValueError(f"No coverage layers found under {input_root}")

    xy_m = require_same_xy(layers)
    summary_csv = output_dir / "coverage_layer_summary.csv"
    stack_npz = output_dir / "coverage_layers.npz"
    summary_png = output_dir / "coverage_layer_summary.png"
    best_altitude_png = output_dir / "coverage_best_altitude_map.png"

    write_summary_csv(layers, summary_csv)
    save_stack_npz(layers, xy_m, stack_npz)
    plot_summary(layers, summary_png)
    plot_best_altitude_map(layers, xy_m, best_altitude_png)

    print(f"[Done] layers: {len(layers)}")
    print(f"[Done] summary CSV: {summary_csv}")
    print(f"[Done] stacked NPZ: {stack_npz}")
    print(f"[Done] summary PNG: {summary_png}")
    print(f"[Done] best-altitude PNG: {best_altitude_png}")


if __name__ == "__main__":
    main()

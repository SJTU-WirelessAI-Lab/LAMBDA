from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lambda_rf import config
from lambda_rf.utils.radar import (
    C_M_S,
    load_radar_npz,
    parse_frame_index,
    parse_shape,
    rotation_matrix_zyx_degrees,
)


def _as_vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least 3 values")
    return arr[:3]


def _spatial_freq_axis(size: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(size, d=1.0))


def _db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def compute_radar_maps(
    cube: np.ndarray,
    radar_params: np.ndarray,
    array_shape: tuple[int, int] | list[int] = (4, 4),
    angle_fft_size: int = 64,
    remove_clutter: bool = True,
) -> dict[str, np.ndarray]:
    """Compute RD, RA, and RE power maps from one radar cube."""
    rows, cols = parse_shape(array_shape, "array_shape")
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"radar cube must have shape (ant, chirp, range_bin), got {cube.shape}")
    num_ant, num_chirps, num_range_bins = cube.shape
    if num_ant != rows * cols:
        raise ValueError(f"array shape {rows}x{cols} expects {rows * cols} antennas, got {num_ant}")
    if num_range_bins < 2 or num_range_bins % 2 != 0:
        raise ValueError("number of range bins must be an even value >= 2")

    params = np.asarray(radar_params, dtype=np.float64).reshape(-1)
    f_c, slope, sample_rate, chirp_duration, expected_bins = params[:5]
    chirp_interval = params[5] if params.size >= 6 else chirp_duration
    expected_bins = int(round(expected_bins))
    if expected_bins != num_range_bins:
        raise ValueError(f"radar_params expects {expected_bins} range bins, got {num_range_bins}")

    work = cube.astype(np.complex128, copy=True)
    if remove_clutter:
        work -= np.mean(work, axis=1, keepdims=True)
    work = np.fft.fftshift(work, axes=-1)
    cube_4d = work.reshape(rows, cols, num_chirps, num_range_bins)

    wavelength = C_M_S / float(f_c)
    fd_axis = np.fft.fftshift(np.fft.fftfreq(num_chirps, d=chirp_interval))
    velocity_axis = fd_axis * wavelength / 2.0

    mid = num_range_bins // 2
    f_pos = np.fft.fftfreq(num_range_bins, d=1.0 / sample_rate)[:mid]
    range_axis = C_M_S * f_pos / (2.0 * slope)
    pos = np.arange(mid, num_range_bins)
    neg = np.arange(mid - 1, -1, -1)

    window = np.hanning(num_chirps).astype(np.float64)
    rd_input = np.sum(cube_4d, axis=(0, 1))
    rd_spec = np.fft.fftshift(np.fft.fft((rd_input.T * window).T, axis=0), axes=0)
    rd_power = np.abs(rd_spec) ** 2
    rd_fold = rd_power[:, pos] + rd_power[:, neg]

    cube_doppler = np.fft.fftshift(
        np.fft.fft(cube_4d * window[None, None, :, None], axis=2),
        axes=2,
    )
    u_axis = _spatial_freq_axis(angle_fft_size) * 2.0
    u_axis = np.clip(u_axis, -1.0, 1.0)
    angle_axis = np.degrees(np.arcsin(u_axis))

    ra_input = np.sum(cube_doppler, axis=0)
    ra_spec = np.fft.fftshift(np.fft.fft(ra_input, n=angle_fft_size, axis=0), axes=0)
    ra_power = np.sum(np.abs(ra_spec) ** 2, axis=1)
    ra_fold = ra_power[:, pos] + ra_power[:, neg]

    re_input = np.sum(cube_doppler, axis=1)
    re_spec = np.fft.fftshift(np.fft.fft(re_input, n=angle_fft_size, axis=0), axes=0)
    re_power = np.sum(np.abs(re_spec) ** 2, axis=1)
    re_fold = re_power[:, pos] + re_power[:, neg]

    return {
        "range_axis_m": range_axis,
        "velocity_axis_m_s": velocity_axis,
        "azimuth_axis_deg": angle_axis,
        "elevation_axis_deg": angle_axis,
        "rd_db": _db(rd_fold),
        "ra_db": _db(ra_fold),
        "re_db": _db(re_fold),
    }


def _world_to_radar(target_pos: np.ndarray, bs_pos: np.ndarray, radar_world_to_local: np.ndarray) -> tuple[float, float, float]:
    vec_global = target_pos - bs_pos
    vec_local = radar_world_to_local @ vec_global
    x, y, z = vec_local
    distance = float(np.linalg.norm(vec_local))
    azimuth = float(np.degrees(np.arctan2(y, x)))
    elevation = float(np.degrees(np.arctan2(z, np.sqrt(x * x + y * y))))
    return distance, azimuth, elevation


def _radial_velocity(gt_pos: np.ndarray, gt_vel: np.ndarray, bs_pos: np.ndarray) -> float:
    los = bs_pos - gt_pos
    los = los / (np.linalg.norm(los) + 1e-12)
    return float(np.dot(gt_vel, los))


def _plot_map(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    overlay: tuple[float, float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])],
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if overlay is not None:
        plt.scatter([overlay[0]], [overlay[1]], marker="x", s=120, linewidths=2, color="red", label="GT")
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def visualize_file(
    radar_path: str | Path,
    output_dir: str | Path,
    array_shape: tuple[int, int] | list[int] = (4, 4),
    angle_fft_size: int = 64,
    remove_clutter: bool = True,
    show_gt: bool = False,
    bs_position: tuple[float, float, float] | list[float] | None = None,
    radar_yaw_deg: float = 0.0,
    radar_pitch_deg: float = 0.0,
    radar_roll_deg: float = 0.0,
) -> dict[str, np.ndarray]:
    radar_path = Path(radar_path)
    output_root = Path(output_dir)
    cube, info, radar_params = load_radar_npz(radar_path)
    maps = compute_radar_maps(
        cube=cube,
        radar_params=radar_params,
        array_shape=array_shape,
        angle_fft_size=angle_fft_size,
        remove_clutter=remove_clutter,
    )

    frame_idx = parse_frame_index(radar_path, prefix="radar")
    frame_tag = f"{frame_idx:06d}" if frame_idx is not None else radar_path.stem
    overlays: dict[str, tuple[float, float] | None] = {"rd": None, "ra": None, "re": None}
    if show_gt and bs_position is not None and info.get("gt_pos") is not None and info.get("gt_vel") is not None:
        bs_pos = _as_vec3(bs_position, "bs_position")
        gt_pos = _as_vec3(info["gt_pos"], "gt_pos")
        gt_vel = _as_vec3(info["gt_vel"], "gt_vel")
        radar_world_to_local = rotation_matrix_zyx_degrees(radar_yaw_deg, radar_pitch_deg, radar_roll_deg).T
        distance, azimuth, elevation = _world_to_radar(gt_pos, bs_pos, radar_world_to_local)
        velocity = _radial_velocity(gt_pos, gt_vel, bs_pos)
        overlays = {
            "rd": (distance, velocity),
            "ra": (distance, azimuth),
            "re": (distance, elevation),
        }

    _plot_map(
        maps["rd_db"],
        maps["range_axis_m"],
        maps["velocity_axis_m_s"],
        "Range (m)",
        "Radial Velocity (m/s)",
        output_root / "RD" / f"frame_{frame_tag}_RD.png",
        overlay=overlays["rd"],
    )
    _plot_map(
        maps["ra_db"],
        maps["range_axis_m"],
        maps["azimuth_axis_deg"],
        "Range (m)",
        "Azimuth (deg)",
        output_root / "RA" / f"frame_{frame_tag}_RA.png",
        overlay=overlays["ra"],
    )
    _plot_map(
        maps["re_db"],
        maps["range_axis_m"],
        maps["elevation_axis_deg"],
        "Range (m)",
        "Elevation (deg)",
        output_root / "RE" / f"frame_{frame_tag}_RE.png",
        overlay=overlays["re"],
    )
    return maps


def run(
    input_dir: str,
    output_dir: str | None = None,
    array_shape: str | tuple[int, int] | list[int] | None = None,
    angle_fft_size: int = 64,
    remove_clutter: bool = True,
    frame_step: int = 1,
    start_frame: int | None = None,
    limit: int | None = None,
    show_gt: bool = False,
    bs_position: str | tuple[float, float, float] | list[float] | None = None,
    radar_yaw_deg: float | None = None,
    radar_pitch_deg: float | None = None,
    radar_roll_deg: float | None = None,
) -> None:
    input_root = Path(input_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Radar input directory not found: {input_root}")
    output_root = Path(output_dir).expanduser().resolve() if output_dir else input_root.parent / "radar_vis" / input_root.name

    files = sorted(input_root.glob("radar_*.npz"))
    if start_frame is not None:
        files = [
            path for path in files
            if (parse_frame_index(path, prefix="radar") is not None and parse_frame_index(path, prefix="radar") >= start_frame)
        ]
    if frame_step > 1:
        files = files[::frame_step]
    if limit is not None:
        files = files[: max(0, int(limit))]
    if not files:
        raise FileNotFoundError(f"No radar_*.npz files found in {input_root}")

    shape = parse_shape(array_shape or config.RADAR_ARRAY_SHAPE, "array_shape")
    if isinstance(bs_position, str):
        bs_position = tuple(float(part.strip()) for part in bs_position.split(","))
    yaw = float(config.RADAR_MOUNT["yaw"] if radar_yaw_deg is None else radar_yaw_deg)
    pitch = float(config.RADAR_MOUNT["pitch"] if radar_pitch_deg is None else radar_pitch_deg)
    roll = float(config.RADAR_MOUNT["roll"] if radar_roll_deg is None else radar_roll_deg)

    print("=" * 80)
    print("[RadarVis] Rendering RD/RA/RE images.")
    print(f"[RadarVis] Input:  {input_root}")
    print(f"[RadarVis] Output: {output_root}")
    print(f"[RadarVis] Files:  {len(files)}")
    print("=" * 80)

    for idx, radar_path in enumerate(files, start=1):
        visualize_file(
            radar_path=radar_path,
            output_dir=output_root,
            array_shape=shape,
            angle_fft_size=angle_fft_size,
            remove_clutter=remove_clutter,
            show_gt=show_gt,
            bs_position=bs_position,
            radar_yaw_deg=yaw,
            radar_pitch_deg=pitch,
            radar_roll_deg=roll,
        )
        if idx == 1 or idx % 100 == 0:
            print(f"[RadarVis] rendered={idx}, latest={radar_path.name}")
    print(f"[RadarVis] Done. output={output_root}")


def main() -> None:
    raise SystemExit("Use `python -m lambda_rf radar-vis --input-dir ...`.")


if __name__ == "__main__":
    main()

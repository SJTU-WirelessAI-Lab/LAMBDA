from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lambda_rf import config
from lambda_rf import paths
from lambda_rf.utils.array_csi import (
    build_array_csi_fields,
    load_rotation_matrix_from_pose_json,
    parse_array_shape,
)
from lambda_rf.utils.subcarrier_csi import build_subcarrier_csi_fields, validate_subcarrier_config


def _frame_index(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return None


def _expand_pose_path(path_value: str) -> Path:
    variables = {
        "SERVER_ROOT": str(config.SERVER_ROOT),
        "SIONNART_ROOT": str(config.SIONNART_ROOT),
        "PROJECT_ROOT": str(config.PROJECT_ROOT),
        "LAMBDA_ROOT": str(config.LAMBDA_ROOT),
    }
    expanded = str(path_value)
    for name, replacement in variables.items():
        expanded = expanded.replace("${" + name + "}", replacement)
    return Path(expanded).expanduser().resolve()


def _resolve_profile(
    profile_name: str | None,
    num_subcarriers: int | None,
    subcarrier_spacing_hz: float | None,
) -> dict[str, Any]:
    has_override = num_subcarriers is not None or subcarrier_spacing_hz is not None
    if profile_name is not None or not (num_subcarriers is not None and subcarrier_spacing_hz is not None):
        profile = config.get_subcarrier_profile(profile_name)
        resolved_name = profile["name"]
    else:
        profile = {
            "num_subcarriers": int(num_subcarriers),
            "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
        }
        resolved_name = None

    if num_subcarriers is not None:
        profile["num_subcarriers"] = int(num_subcarriers)
    if subcarrier_spacing_hz is not None:
        profile["subcarrier_spacing_hz"] = float(subcarrier_spacing_hz)

    num, spacing = validate_subcarrier_config(
        profile["num_subcarriers"],
        profile["subcarrier_spacing_hz"],
    )
    profile["num_subcarriers"] = num
    profile["subcarrier_spacing_hz"] = spacing
    profile["profile_name"] = resolved_name if not has_override else None
    profile["profile_tag"] = paths.subcarrier_profile_tag(
        profile_name=profile["profile_name"],
        num_subcarriers=num,
        subcarrier_spacing_hz=spacing,
    )
    return profile


def _sibling_mimo_ofdm_output_dir(
    input_root: Path,
    rx_shape: tuple[int, int],
    tx_shape: tuple[int, int],
    profile_tag: str,
) -> Path:
    shape_tag = paths.array_shape_tag(rx_shape=rx_shape, tx_shape=tx_shape)
    parts = list(input_root.parts)
    for idx, part in enumerate(parts):
        if part.startswith("csi_rt_"):
            parts[idx] = "mimo_ofdm_" + part
            return Path(*parts) / shape_tag / profile_tag
        if part == "csi":
            parts[idx] = "mimo_ofdm_csi"
            return Path(*parts) / shape_tag / profile_tag
    return input_root.parent / "mimo_ofdm_csi" / input_root.name / shape_tag / profile_tag


def _load_rotation_sources(
    tx_orientation_pose: str | None,
    rx_orientation_pose: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str | None]:
    tx_rotation = None
    rx_rotation = None
    tx_orientation_source = None
    rx_orientation_source = None

    if tx_orientation_pose is None and config.TX_POSE_PATH:
        tx_orientation_pose = config.TX_POSE_PATH
    if tx_orientation_pose:
        tx_orientation_path = _expand_pose_path(tx_orientation_pose)
        tx_rotation = load_rotation_matrix_from_pose_json(tx_orientation_path)
        tx_orientation_source = str(tx_orientation_path)
    if rx_orientation_pose:
        rx_orientation_path = _expand_pose_path(rx_orientation_pose)
        rx_rotation = load_rotation_matrix_from_pose_json(rx_orientation_path)
        rx_orientation_source = str(rx_orientation_path)

    return tx_rotation, rx_rotation, tx_orientation_source, rx_orientation_source


def expand_mimo_ofdm_npz(
    input_path: str | Path,
    output_path: str | Path,
    tx_shape: tuple[int, int] | list[int],
    rx_shape: tuple[int, int] | list[int],
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    spacing_wavelengths: float = 0.5,
    tx_rotation_matrix: np.ndarray | None = None,
    rx_rotation_matrix: np.ndarray | None = None,
    tx_orientation_source: str | None = None,
    rx_orientation_source: str | None = None,
    profile_name: str | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with np.load(input_path, allow_pickle=False) as data:
        arrays: dict[str, Any] = {key: data[key] for key in data.files}

    carrier_values = np.asarray(arrays.get("carrier_frequency", 0.0), dtype=np.float64).reshape(-1)
    if carrier_values.size != 1 or float(carrier_values[0]) <= 0.0:
        raise ValueError(f"{input_path} is missing a positive scalar carrier_frequency field")

    mimo_fields = build_array_csi_fields(
        arrays=arrays,
        carrier_frequency_hz=float(carrier_values[0]),
        tx_shape=tx_shape,
        rx_shape=rx_shape,
        spacing_wavelengths=spacing_wavelengths,
        tx_rotation_matrix=tx_rotation_matrix,
        rx_rotation_matrix=rx_rotation_matrix,
        tx_orientation_source=tx_orientation_source,
        rx_orientation_source=rx_orientation_source,
    )
    arrays.update(mimo_fields)

    ofdm_fields = build_subcarrier_csi_fields(
        arrays=arrays,
        num_subcarriers=num_subcarriers,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        input_mode="array",
        profile_name=profile_name,
    )
    arrays.update(ofdm_fields)
    arrays["source_csi_path"] = np.asarray(str(input_path))
    arrays["source_mimo_ofdm_input_path"] = np.asarray(str(input_path))
    arrays["csi_product"] = np.asarray("mimo_ofdm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(output_path)


def run(
    input_dir: str | None = None,
    output_dir: str | None = None,
    tx_shape: str | tuple[int, int] | list[int] | None = None,
    rx_shape: str | tuple[int, int] | list[int] | None = None,
    spacing_wavelengths: float = 0.5,
    profile: str | None = None,
    num_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
    skip_existing: bool = False,
    start_frame: int | None = None,
    limit: int | None = None,
    tx_orientation_pose: str | None = None,
    rx_orientation_pose: str | None = None,
) -> Path:
    tx_shape_tuple = parse_array_shape(tx_shape, config.TX_ARRAY_SHAPE, "tx_shape")
    rx_shape_tuple = parse_array_shape(rx_shape, config.RX_ARRAY_SHAPE, "rx_shape")
    profile_cfg = _resolve_profile(profile, num_subcarriers, subcarrier_spacing_hz)

    input_root = Path(input_dir or paths.csi_npz_dir()).resolve()
    if output_dir:
        output_root = Path(output_dir).resolve()
    elif input_dir:
        output_root = _sibling_mimo_ofdm_output_dir(
            input_root=input_root,
            rx_shape=rx_shape_tuple,
            tx_shape=tx_shape_tuple,
            profile_tag=profile_cfg["profile_tag"],
        ).resolve()
    else:
        output_root = Path(
            paths.mimo_ofdm_csi_npz_dir(
                rx_shape=rx_shape_tuple,
                tx_shape=tx_shape_tuple,
                profile_name=profile_cfg["profile_name"],
                num_subcarriers=profile_cfg["num_subcarriers"],
                subcarrier_spacing_hz=profile_cfg["subcarrier_spacing_hz"],
            )
        ).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input path-level CSI directory not found: {input_root}")

    csi_files = sorted(input_root.glob("csi_*.npz"))
    if start_frame is not None:
        csi_files = [path for path in csi_files if (_frame_index(path) is not None and _frame_index(path) >= start_frame)]
    if limit is not None:
        csi_files = csi_files[: max(0, int(limit))]
    if not csi_files:
        raise FileNotFoundError(f"No csi_*.npz files found in {input_root}")

    tx_rotation, rx_rotation, tx_orientation_source, rx_orientation_source = _load_rotation_sources(
        tx_orientation_pose=tx_orientation_pose,
        rx_orientation_pose=rx_orientation_pose,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("[MIMO OFDM CSI] Building MIMO frequency-domain CSI.")
    print(f"[MIMO OFDM CSI] Input:   {input_root}")
    print(f"[MIMO OFDM CSI] Output:  {output_root}")
    print(
        f"[MIMO OFDM CSI] MIMO:    TX shape={tx_shape_tuple}, RX shape={rx_shape_tuple}, "
        f"spacing={spacing_wavelengths} lambda"
    )
    print(
        f"[MIMO OFDM CSI] OFDM:    {profile_cfg['profile_tag']} | "
        f"N={profile_cfg['num_subcarriers']}, df={profile_cfg['subcarrier_spacing_hz']} Hz"
    )
    print(f"[MIMO OFDM CSI] TX orientation: {tx_orientation_source or 'identity/global y-z plane'}")
    print(f"[MIMO OFDM CSI] RX orientation: {rx_orientation_source or 'identity/global y-z plane'}")
    print(f"[MIMO OFDM CSI] Files:   {len(csi_files)}")
    print("=" * 80)

    processed = 0
    skipped = 0
    for input_path in csi_files:
        output_path = output_root / input_path.name
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        expand_mimo_ofdm_npz(
            input_path=input_path,
            output_path=output_path,
            tx_shape=tx_shape_tuple,
            rx_shape=rx_shape_tuple,
            spacing_wavelengths=spacing_wavelengths,
            num_subcarriers=profile_cfg["num_subcarriers"],
            subcarrier_spacing_hz=profile_cfg["subcarrier_spacing_hz"],
            tx_rotation_matrix=tx_rotation,
            rx_rotation_matrix=rx_rotation,
            tx_orientation_source=tx_orientation_source,
            rx_orientation_source=rx_orientation_source,
            profile_name=profile_cfg["profile_tag"],
        )
        processed += 1
        if processed == 1 or processed % 100 == 0:
            print(f"[MIMO OFDM CSI] processed={processed}, latest={output_path.name}")

    print(f"[MIMO OFDM CSI] Done. processed={processed}, skipped={skipped}, output={output_root}")
    return output_root


def main() -> None:
    run()


if __name__ == "__main__":
    main()

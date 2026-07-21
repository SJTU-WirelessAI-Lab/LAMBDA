from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


SERVER_ROOT = Path(__file__).resolve().parents[1]
SIONNART_ROOT = SERVER_ROOT.parent
PROJECT_ROOT = SIONNART_ROOT.parent
LAMBDA_ROOT = Path(os.environ.get("LAMBDA_ROOT", SERVER_ROOT)).resolve()
ASSETS_ROOT = SERVER_ROOT / "assets"
CONFIG_PATH = Path(
    os.environ.get("LAMBDA_SCENARIOS_CONFIG", SERVER_ROOT / "configs" / "scenarios.json")
).resolve()

DEFAULT_SUBCARRIER_PROFILES: dict[str, dict[str, Any]] = {
    "lband_15k_1024": {
        "name": "lband_15k_1024",
        "num_subcarriers": 1024,
        "subcarrier_spacing_hz": 15_000.0,
    },
    "sub6_30k_1024": {
        "name": "sub6_30k_1024",
        "num_subcarriers": 1024,
        "subcarrier_spacing_hz": 30_000.0,
    },
    "sub6_60k_512": {
        "name": "sub6_60k_512",
        "num_subcarriers": 512,
        "subcarrier_spacing_hz": 60_000.0,
    },
    "mmwave_120k_512": {
        "name": "mmwave_120k_512",
        "num_subcarriers": 512,
        "subcarrier_spacing_hz": 120_000.0,
    },
    "mmwave_240k_256": {
        "name": "mmwave_240k_256",
        "num_subcarriers": 256,
        "subcarrier_spacing_hz": 240_000.0,
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "default_scenario": "example_csi",
    "common": {
        "output_root": "${SERVER_ROOT}/data_export",
        "scene_name": "Example",
        "scenario_name": "Released CSI",
        "data_weather": "Sunny",
        "station_name": "BS1",
        "trajectory_name": "trajectory_000",
        "carrier_frequency": 28.0e9,
        "tx_polarization": "V",
        "tx_array_shape": [4, 4],
        "rx_array_shape": [1, 1],
        "weather": {"kind": "clear", "include_gaseous_absorption": False},
        "subcarrier_profiles": DEFAULT_SUBCARRIER_PROFILES,
        "default_subcarrier_profile": "sub6_30k_1024",
        "radar": {
            "bandwidth": 1.0e9,
            "sample_rate": 100.0e6,
            "chirp_duration": 40.0e-6,
            "chirp_interval": 50.0e-6,
            "num_chirps": 128,
            "add_noise": True,
            "noise_floor_dbm": None,
            "noise_figure_db": 6.0,
            "noise_bandwidth_hz": 100.0e6,
            "noise_seed": 0,
            "tx_power_dbm": 12.0,
            "tx_gain_db": 25.0,
            "rx_gain_db": 25.0,
            "array_shape": [4, 4],
            "spacing_wavelengths": 0.49
        },
        "radar_mount": {
            "yaw": -180.0,
            "pitch": 40.0,
            "roll": 0.0
        },
    },
    "scenarios": {
        "example_csi": {
            "description": "Example configuration for released CSI and radar utilities.",
        }
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_raw_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return deepcopy(DEFAULT_CONFIG)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config() -> tuple[str, dict[str, Any], dict[str, Any]]:
    raw = _load_raw_config()
    scenarios = raw.get("scenarios", {})
    default_key = raw.get("default_scenario", "example_csi")
    requested_key = os.environ.get("LAMBDA_SCENARIO", default_key)
    if requested_key not in scenarios:
        known = ", ".join(sorted(scenarios)) or "<none>"
        raise ValueError(f"Unknown LAMBDA_SCENARIO={requested_key!r}. Known scenarios: {known}.")

    merged = _deep_merge(DEFAULT_CONFIG.get("common", {}), raw.get("common", {}))
    merged = _deep_merge(merged, scenarios[requested_key])
    return requested_key, merged, raw


SCENARIO_KEY, _CFG, _RAW_CONFIG = _load_config()


def _expand_path(value: str) -> str:
    variables = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "SIONNART_ROOT": str(SIONNART_ROOT),
        "SERVER_ROOT": str(SERVER_ROOT),
        "LAMBDA_ROOT": str(LAMBDA_ROOT),
    }
    expanded = value
    for name, replacement in variables.items():
        expanded = expanded.replace("${" + name + "}", replacement)
    return os.path.expandvars(os.path.expanduser(expanded))


def _optional_path(key: str) -> str:
    value = _CFG.get(key)
    if not value:
        return ""
    return str(Path(_expand_path(str(value))).resolve())


def _tuple(key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = _CFG.get(key, default)
    if len(value) != 2:
        raise ValueError(f"{key} must contain two integers, got {value!r}")
    return int(value[0]), int(value[1])


def _shape_from_mapping(mapping: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = mapping.get(key, default)
    if len(value) != 2:
        raise ValueError(f"{key} must contain two integers, got {value!r}")
    rows, cols = int(value[0]), int(value[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{key} values must be positive, got {value!r}")
    return rows, cols


DESCRIPTION = str(_CFG.get("description", ""))
OUTPUT_ROOT = str(Path(_expand_path(str(_CFG.get("output_root", "${SERVER_ROOT}/data_export")))).resolve())
TX_POSE_PATH = _optional_path("tx_pose_path")
RCS_MODEL_PATHS = {
    28.0e9: str((ASSETS_ROOT / "default_drone_rcs_28ghz.h5").resolve()),
    60.0e9: str((ASSETS_ROOT / "default_drone_rcs_60ghz.h5").resolve()),
    77.0e9: str((ASSETS_ROOT / "default_drone_rcs_77ghz.h5").resolve()),
}
RCS_MODEL_PATH = RCS_MODEL_PATHS.get(float(_CFG.get("carrier_frequency", 28.0e9)), "")

SCENE_NAME = str(_CFG.get("scene_name", "Example"))
SCENARIO_NAME = str(_CFG.get("scenario_name", SCENARIO_KEY))
DATA_WEATHER = str(_CFG.get("data_weather", "Sunny"))
STATION_NAME = str(_CFG.get("station_name", "BS1"))
TRAJECTORY_NAME = str(_CFG.get("trajectory_name", "trajectory_000"))

CARRIER_FREQUENCY = float(_CFG.get("carrier_frequency", 28.0e9))
TX_POLARIZATION = str(_CFG.get("tx_polarization", "V"))
WEATHER_DICT = dict(_CFG.get("weather", {"kind": "clear", "include_gaseous_absorption": False}))

TX_ARRAY_SHAPE = _tuple("tx_array_shape", (4, 4))
RX_ARRAY_SHAPE = _tuple("rx_array_shape", (1, 1))

SUBCARRIER_PROFILES = _deep_merge(
    DEFAULT_SUBCARRIER_PROFILES,
    _CFG.get("subcarrier_profiles", {}),
)
DEFAULT_SUBCARRIER_PROFILE = str(_CFG.get("default_subcarrier_profile", "sub6_30k_1024"))

RADAR_CFG = dict(_CFG.get("radar", {}))
RADAR_MOUNT = dict(_CFG.get("radar_mount", {}))
RADAR_SETTINGS = {
    "bandwidth": float(RADAR_CFG.get("bandwidth", 1.0e9)),
    "sample_rate": float(RADAR_CFG.get("sample_rate", 100.0e6)),
    "chirp_duration": float(RADAR_CFG.get("chirp_duration", 40.0e-6)),
    "chirp_interval": (
        None
        if RADAR_CFG.get("chirp_interval") is None
        else float(RADAR_CFG.get("chirp_interval"))
    ),
    "num_chirps": int(RADAR_CFG.get("num_chirps", 128)),
    "add_noise": bool(RADAR_CFG.get("add_noise", True)),
    "noise_floor_dbm": (
        None
        if RADAR_CFG.get("noise_floor_dbm") is None
        else float(RADAR_CFG.get("noise_floor_dbm"))
    ),
    "noise_figure_db": float(RADAR_CFG.get("noise_figure_db", 0.0)),
    "noise_bandwidth_hz": (
        None
        if RADAR_CFG.get("noise_bandwidth_hz") is None
        else float(RADAR_CFG.get("noise_bandwidth_hz"))
    ),
    "noise_seed": int(RADAR_CFG.get("noise_seed", 0)),
    "tx_power_dbm": float(RADAR_CFG.get("tx_power_dbm", 12.0)),
    "tx_gain_db": float(RADAR_CFG.get("tx_gain_db", 25.0)),
    "rx_gain_db": float(RADAR_CFG.get("rx_gain_db", 25.0)),
}
RADAR_ARRAY_SHAPE = _shape_from_mapping(RADAR_CFG, "array_shape", (4, 4))
RADAR_SPACING_WAVELENGTHS = float(RADAR_CFG.get("spacing_wavelengths", 0.49))
RADAR_MOUNT = {
    "yaw": float(RADAR_MOUNT.get("yaw", -180.0)),
    "pitch": float(RADAR_MOUNT.get("pitch", 40.0)),
    "roll": float(RADAR_MOUNT.get("roll", 0.0)),
}


def get_subcarrier_profile(name: str | None = None) -> dict[str, Any]:
    profile_name = name or os.environ.get("LAMBDA_SUBCARRIER_PROFILE") or DEFAULT_SUBCARRIER_PROFILE
    if profile_name not in SUBCARRIER_PROFILES:
        known = ", ".join(sorted(SUBCARRIER_PROFILES))
        raise ValueError(f"Unknown subcarrier profile {profile_name!r}. Known profiles: {known}.")
    profile = deepcopy(SUBCARRIER_PROFILES[profile_name])
    profile["name"] = str(profile.get("name", profile_name))
    return profile

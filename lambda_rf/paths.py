from __future__ import annotations

import os
from typing import Any

from lambda_rf import config


def format_float_tag(x: float) -> str:
    return f"{x:.1f}".replace(".", "p")


def _compact_float_tag(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def format_hz_tag(value_hz: float) -> str:
    value = float(value_hz)
    if value <= 0.0:
        raise ValueError(f"value_hz must be > 0, got {value_hz}")
    for suffix, scale in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        scaled = value / scale
        if scaled >= 1.0 and abs(scaled - round(scaled)) < 1e-9:
            return f"{_compact_float_tag(scaled)}{suffix}"
    if value >= 1e3:
        scaled = value / 1e3
        return f"{_compact_float_tag(scaled)}kHz"
    return f"{_compact_float_tag(value)}Hz"


def make_freq_pol_tag(f_c_hz: float | None = None, pol: str | None = None) -> str:
    f_ghz = (f_c_hz if f_c_hz is not None else config.CARRIER_FREQUENCY) / 1e9
    pol_tag = (pol or config.TX_POLARIZATION or "V").upper()
    return f"f{format_float_tag(f_ghz)}GHz_{pol_tag}"


def _weather_value(weather: Any, key: str, default: Any = None) -> Any:
    if isinstance(weather, dict):
        return weather.get(key, default)
    return getattr(weather, key, default)


def make_weather_tag(weather: Any | None = None) -> str:
    """Return the detailed weather tag used by released CSI directory names."""
    if isinstance(weather, str):
        return weather

    weather = config.WEATHER_DICT if weather is None else weather
    kind = _weather_value(weather, "kind", "clear")

    if kind == "clear":
        return "clear"
    if kind == "rain":
        rain_rate = float(_weather_value(weather, "rain_rate_mm_h", 0.0))
        tag = f"rain_R{format_float_tag(rain_rate)}mmh"
    elif kind == "cloud_fog":
        water_density = float(_weather_value(weather, "liquid_water_density_g_m3", 0.0))
        tag = f"fog_M{_compact_float_tag(water_density)}gpm3"
    elif kind == "snow":
        snow_rate = float(_weather_value(weather, "snow_rate_mm_h", 0.0))
        snow_is_lwe = bool(_weather_value(weather, "snow_is_lwe", True))
        snow_model = _weather_value(weather, "snow_model", "gunn_east")
        temperature = float(_weather_value(weather, "T_K", 273.15))
        snow_type = _weather_value(weather, "snow_type_override", None)
        snow_type = snow_type or ("wet" if temperature >= 273.15 else "dry")
        unit_tag = "lwe" if snow_is_lwe else "depth"
        tag = f"snow_{snow_model}_{snow_type}_R{format_float_tag(snow_rate)}mmh_{unit_tag}"
    else:
        tag = str(kind)

    if bool(_weather_value(weather, "include_gaseous_absorption", True)):
        tag = f"{tag}_gas_p676"
    return tag


def make_csi_output_tag(
    f_c_hz: float | None = None,
    pol: str | None = None,
    weather: Any | None = None,
) -> str:
    freq_pol_tag = make_freq_pol_tag(f_c_hz=f_c_hz, pol=pol)
    weather_tag = make_weather_tag(weather)
    return freq_pol_tag if weather_tag == "clear" else f"{freq_pol_tag}_{weather_tag}"


def trajectory_root() -> str:
    return os.path.join(
        config.OUTPUT_ROOT,
        config.SCENE_NAME,
        config.SCENARIO_NAME,
        config.DATA_WEATHER,
        config.STATION_NAME,
        config.TRAJECTORY_NAME,
    )


def csi_output_dir(pol: str | None = None, weather: Any | None = None) -> str:
    return os.path.join(trajectory_root(), "csi", make_csi_output_tag(pol=pol, weather=weather))


def csi_npz_dir(pol: str | None = None, weather: Any | None = None) -> str:
    return csi_output_dir(pol=pol, weather=weather)


def array_shape_tag(rx_shape: tuple[int, int] | list[int], tx_shape: tuple[int, int] | list[int]) -> str:
    rx_rows, rx_cols = int(rx_shape[0]), int(rx_shape[1])
    tx_rows, tx_cols = int(tx_shape[0]), int(tx_shape[1])
    return f"rx{rx_rows}x{rx_cols}_tx{tx_rows}x{tx_cols}"


def subcarrier_profile_tag(
    profile_name: str | None = None,
    num_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
) -> str:
    if profile_name:
        return str(profile_name).strip().replace(" ", "_")
    if num_subcarriers is None or subcarrier_spacing_hz is None:
        profile = config.get_subcarrier_profile()
        return str(profile["name"])
    return f"n{int(num_subcarriers)}_df{format_hz_tag(float(subcarrier_spacing_hz))}"


def array_csi_output_dir(
    pol: str | None = None,
    weather: Any | None = None,
    rx_shape: tuple[int, int] | list[int] | None = None,
    tx_shape: tuple[int, int] | list[int] | None = None,
) -> str:
    rx_shape = rx_shape or config.RX_ARRAY_SHAPE
    tx_shape = tx_shape or config.TX_ARRAY_SHAPE
    return os.path.join(
        trajectory_root(),
        "array_csi",
        make_csi_output_tag(pol=pol, weather=weather),
        array_shape_tag(rx_shape=rx_shape, tx_shape=tx_shape),
    )


def array_csi_npz_dir(
    pol: str | None = None,
    weather: Any | None = None,
    rx_shape: tuple[int, int] | list[int] | None = None,
    tx_shape: tuple[int, int] | list[int] | None = None,
) -> str:
    return array_csi_output_dir(pol=pol, weather=weather, rx_shape=rx_shape, tx_shape=tx_shape)


def subcarrier_csi_output_dir(
    pol: str | None = None,
    weather: Any | None = None,
    input_tag: str = "single",
    profile_name: str | None = None,
    num_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
) -> str:
    return os.path.join(
        trajectory_root(),
        "subcarrier_csi",
        make_csi_output_tag(pol=pol, weather=weather),
        str(input_tag).strip().replace(" ", "_"),
        subcarrier_profile_tag(
            profile_name=profile_name,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
        ),
    )


def subcarrier_csi_npz_dir(
    pol: str | None = None,
    weather: Any | None = None,
    input_tag: str = "single",
    profile_name: str | None = None,
    num_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
) -> str:
    return subcarrier_csi_output_dir(
        pol=pol,
        weather=weather,
        input_tag=input_tag,
        profile_name=profile_name,
        num_subcarriers=num_subcarriers,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
    )


def mimo_ofdm_csi_output_dir(
    pol: str | None = None,
    weather: Any | None = None,
    rx_shape: tuple[int, int] | list[int] | None = None,
    tx_shape: tuple[int, int] | list[int] | None = None,
    profile_name: str | None = None,
    num_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
) -> str:
    rx_shape = rx_shape or config.RX_ARRAY_SHAPE
    tx_shape = tx_shape or config.TX_ARRAY_SHAPE
    return os.path.join(
        trajectory_root(),
        "mimo_ofdm_csi",
        make_csi_output_tag(pol=pol, weather=weather),
        array_shape_tag(rx_shape=rx_shape, tx_shape=tx_shape),
        subcarrier_profile_tag(
            profile_name=profile_name,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
        ),
    )


def mimo_ofdm_csi_npz_dir(
    pol: str | None = None,
    weather: Any | None = None,
    rx_shape: tuple[int, int] | list[int] | None = None,
    tx_shape: tuple[int, int] | list[int] | None = None,
    profile_name: str | None = None,
    num_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
) -> str:
    return mimo_ofdm_csi_output_dir(
        pol=pol,
        weather=weather,
        rx_shape=rx_shape,
        tx_shape=tx_shape,
        profile_name=profile_name,
        num_subcarriers=num_subcarriers,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
    )

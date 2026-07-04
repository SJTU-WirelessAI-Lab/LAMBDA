# Configuration

`configs/scenarios.json` provides defaults for the public RF utilities. It is
mainly used for path conventions, array dimensions, frequency tags, weather
tags, subcarrier profiles, radar settings, and the bundled AirSim drone RCS
model.

Most commands can be run directly with explicit `--input-dir` and `--output-dir`
paths. Use the config when you want reproducible naming across multiple runs.

## Loading Rules

The config has three top-level keys:

```json
{
  "default_scenario": "example_csi",
  "common": {},
  "scenarios": {}
}
```

`common` provides defaults. Each entry under `scenarios` deep-merges over those
defaults. Runtime overrides are available through:

```bash
python -m lambda_rf --config path/to/scenarios.json list-scenarios
python -m lambda_rf --config path/to/scenarios.json mimo-ofdm-csi --scenario example_csi --input-dir path/to/csi
```

## Paths

Supported path variables:

| Variable | Meaning |
| --- | --- |
| `${SERVER_ROOT}` | Repository root. |
| `${LAMBDA_ROOT}` | Defaults to `${SERVER_ROOT}` and can be overridden by `LAMBDA_ROOT`. |
| `${SIONNART_ROOT}` | Parent of the repository root, kept for compatibility with local layouts. |
| `${PROJECT_ROOT}` | Parent of `${SIONNART_ROOT}`, kept for compatibility with local layouts. |

Path fields used by the public utilities:

| Field | Meaning |
| --- | --- |
| `output_root` | Root used when a command constructs default output paths. |
| `scene_name` | Dataset family or scene group shown in generated paths. |
| `scenario_name` | Scenario label shown in generated paths. |
| `data_weather` | Condition layer, for example `Sunny`, `Rainy`, `Foggy`, `Snowy`, or `Night`. |
| `station_name` | Base-station label. |
| `trajectory_name` | Trajectory label. |
| `tx_pose_path` | Optional pose JSON used as the default TX array orientation source. |

The default utility path convention is:

```text
<output_root>/<scene_name>/<scenario_name>/<data_weather>/<station_name>/<trajectory_name>/
  csi/<freq_pol_weather_tag>/
  mimo_ofdm_csi/<freq_pol_weather_tag>/<array_shape_tag>/<profile>/
  radar_raw/<freq_pol_weather_tag>/
  radar_vis/<freq_pol_weather_tag>/
```

When `--input-dir` is passed, MIMO OFDM outputs are written as sibling
directories unless `--output-dir` is also passed.

## Frequency And Weather Tags

Frequency/polarization tags use:

```text
f<GHz with p decimal>GHz_<POL>
```

Examples:

```text
f4p9GHz_V
f60p0GHz_V
f77p0GHz_V
```

Clear-condition CSI keeps the short frequency tag. Rain, fog, and snow tags add
the configured weather suffix:

```text
f60p0GHz_V_rain_R10p0mmh_gas_p676
f60p0GHz_V_fog_M0p1gpm3_gas_p676
f60p0GHz_V_snow_gunn_east_dry_R2p0mmh_lwe_gas_p676
```

## MIMO OFDM CSI

`mimo-ofdm-csi` expands released path-level CSI into MIMO path coefficients and
then builds OFDM frequency-domain CSI in the same output file. The default
array model is far-field plane-wave steering. The optional
`--array-model spherical-wave` mode uses path vertices to synthesize near-field
phase and delay differences for each antenna pair.

```bash
python -m lambda_rf mimo-ofdm-csi \
  --input-dir path/to/csi/f60p0GHz_V \
  --output-dir path/to/mimo_ofdm_csi/f60p0GHz_V/rx1x1_tx4x4/sub6_30k_1024 \
  --tx-shape 4,4 \
  --rx-shape 1,1 \
  --profile sub6_30k_1024
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--tx-shape ROWS,COLS` | TX MIMO planar array shape. |
| `--rx-shape ROWS,COLS` | RX MIMO planar array shape. |
| `--spacing-wavelengths` | MIMO element spacing in carrier wavelengths, default `0.5`. |
| `--tx-orientation-pose` | Pose JSON whose quaternion rotates TX local MIMO coordinates into the world frame. |
| `--rx-orientation-pose` | Pose JSON whose quaternion rotates RX local MIMO coordinates into the world frame. |
| `--array-model` | `far-field` for plane-wave steering, or `spherical-wave` for path-vertex near-field phases and delays. |
| `--profile NAME` | Use a named OFDM subcarrier profile. |
| `--num-subcarriers N` | Override profile subcarrier count. |
| `--subcarrier-spacing HZ` | Override profile subcarrier spacing in Hz. |
| `--start-frame` | Only process frames at or after this index. |
| `--limit` | Maximum number of files to process. |
| `--skip-existing` | Skip existing output files. |

Default profiles:

| Profile | Subcarriers | Spacing | Nominal bandwidth |
| --- | ---: | ---: | ---: |
| `lband_15k_1024` | 1024 | 15 kHz | 15.36 MHz |
| `sub6_30k_1024` | 1024 | 30 kHz | 30.72 MHz |
| `sub6_60k_512` | 512 | 60 kHz | 30.72 MHz |
| `mmwave_120k_512` | 512 | 120 kHz | 61.44 MHz |
| `mmwave_240k_256` | 256 | 240 kHz | 61.44 MHz |

You can override profiles from the command line:

```bash
python -m lambda_rf mimo-ofdm-csi \
  --input-dir path/to/csi/f4p9GHz_V \
  --num-subcarriers 2048 \
  --subcarrier-spacing 30000
```

Added NPZ fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `a_mimo_real` / `a_mimo_imag` | `(rx_ant, tx_ant, path)` | MIMO path coefficients. |
| `rx_array_pos` / `tx_array_pos` | `(num_ant, 3)` | World-frame element offsets, in meters. |
| `rx_array_pos_local` / `tx_array_pos_local` | `(num_ant, 3)` | Centered local y-z plane element offsets, in meters. |
| `rx_array_rotation` / `tx_array_rotation` | `(3, 3)` | Local-to-world rotation matrix. |
| `rx_array_shape` / `tx_array_shape` | `(2,)` | `[rows, cols]`. |
| `array_spacing_wavelengths` | scalar | Element spacing in wavelengths. |
| `array_model` | scalar string | `far_field_steering_from_single_link_with_optional_orientation` or `spherical_wavefront_from_path_vertices`. |
| `tau_mimo` | `(rx_ant, tx_ant, path)` when spherical-wave is used | Per-antenna-pair path delay. |
| `path_length_mimo` | `(rx_ant, tx_ant, path)` when spherical-wave is used | Per-antenna-pair geometric path length. |
| `near_field_reference` / `near_field_spreading` | scalar string when spherical-wave is used | Reference delay and amplitude convention metadata. |
| `source_csi_path` | scalar string | Source path-level CSI file path. |
| `h_freq_real` / `h_freq_imag` | `(subcarrier,)`, `(rx_ant, tx_ant, subcarrier)`, or compatible batch shape | Frequency-domain CSI. |
| `subcarrier_frequencies_hz` | `(subcarrier,)` | Baseband frequency offset of each subcarrier. |
| `subcarrier_spacing_hz` | scalar | Subcarrier spacing. |
| `num_subcarriers` | scalar | Number of subcarriers. |
| `subcarrier_input_mode` | scalar string | `array`. |
| `subcarrier_profile` | scalar string | Profile name or generated profile tag. |
| `source_mimo_ofdm_input_path` | scalar string | Input file used by final MIMO OFDM generation. |
| `csi_product` | scalar string | `mimo_ofdm`. |

Spherical-wave mode requires path-level CSI files that include `vertices`,
`interactions`, `tx_pos`, and `uav_pos`. Files without path vertices can still
be processed with the default far-field model.

## CSI Reader

```bash
python -m lambda_rf read-csi path/to/csi_000000.npz
python -m lambda_rf read-csi path/to/csi_000000.npz --top 0 --csv paths.csv
```

The reader prints path count, strongest paths, delay, angles, and available NPZ
fields. The CSV export is useful for quick inspection and plotting.

## Radar Signal Generation

`radar` synthesizes FMCW radar cubes from released path-level CSI files. It does
not require scene XML files. It always uses the bundled AirSim default-drone RCS
model:

```text
assets/default_drone_rcs.h5
```

Install radar dependencies first:

```bash
pip install -e ".[radar]"
```

Generate radar raw files:

```bash
python -m lambda_rf radar \
  --input-dir path/to/csi/f60p0GHz_V \
  --output-dir path/to/radar_raw/f60p0GHz_V \
  --imu-dir path/to/imu
```

The same defaults can be stored under `common.radar` in
`configs/scenarios.json`, including `bandwidth`, `sample_rate`,
`chirp_duration`, `chirp_interval`, `add_noise`, noise parameters, and
`array_shape`.

Useful options:

| Option | Meaning |
| --- | --- |
| `--imu-dir` | Optional directory containing `imu_*.json` orientation files. Identity orientation is used when omitted. |
| `--bandwidth` | FMCW bandwidth in Hz. |
| `--sample-rate` | ADC sample rate in Hz. |
| `--chirp-duration` | Chirp duration in seconds. |
| `--chirp-interval` | Chirp-to-chirp interval/PRI in seconds, including idle gap. |
| `--idle-time` | Idle gap after each chirp in seconds; ignored when `--chirp-interval` is set. |
| `--num-chirps` | Number of chirps per frame. |
| `--noise-floor-dbm` | Receiver noise floor in dBm. Use config `null` to derive thermal noise from bandwidth and noise figure. |
| `--noise-figure-db` | Receiver noise figure in dB when thermal noise is derived. |
| `--noise-bandwidth` | Noise bandwidth in Hz. Defaults to the ADC sample rate. |
| `--add-noise` | Add complex Gaussian receiver noise before the range FFT. |
| `--array-shape` | Radar virtual array shape, default `4,4`. |
| `--radar-yaw`, `--radar-pitch`, `--radar-roll` | Radar mount orientation in degrees. |
| `--array-model` | `far-field` for plane-wave steering, or `spherical-wave` for per-antenna near-field delays from CSI vertices. |

Output NPZ fields include:

| Field | Shape | Meaning |
| --- | --- | --- |
| `radar_data` | `(ant, chirp, range_bin)` | Complex range-FFT radar cube. |
| `radar_params` | `(6,)` | `[f_c, slope, sample_rate, chirp_duration, num_samples, chirp_interval]`. Legacy `(5,)` files are still accepted by visualization. |
| `radar_chirp_duration_s` | scalar | FMCW ramp duration in seconds. |
| `radar_chirp_interval_s` | scalar | Chirp-to-chirp interval/PRI in seconds. |
| `radar_idle_time_s` | scalar | Idle gap after each chirp in seconds. |
| `radar_add_noise` | scalar bool | Whether receiver noise was added. |
| `radar_noise_floor_dbm` | scalar | Effective receiver noise floor in dBm. |
| `radar_noise_power_w` / `radar_noise_std` | scalar | Complex Gaussian noise power and per-real-component standard deviation. |
| `radar_array_pos` | `(ant, 3)` | Virtual array element positions in meters. |
| `radar_array_shape` | `(2,)` | `[rows, cols]`. |
| `rcs_model` | scalar string | `airsim_default_drone`. |
| `radar_array_model` | scalar string | `far-field` or `spherical-wave`. |
| `radar_model` | scalar string | Radar synthesis model identifier. |
| `source_csi_path` | scalar string | Source CSI file path. |
| `gt_pos` / `gt_vel` | `(3,)` when present | Ground-truth target position and velocity copied from CSI. |

## Radar Visualization

`radar-vis` renders Range-Doppler, Range-Azimuth, and Range-Elevation images
from `radar_*.npz` files:

```bash
python -m lambda_rf radar-vis \
  --input-dir path/to/radar_raw/f60p0GHz_V \
  --output-dir path/to/radar_vis/f60p0GHz_V
```

It writes:

```text
RD/frame_000000_RD.png
RA/frame_000000_RA.png
RE/frame_000000_RE.png
```

GT overlay is available when radar files include `gt_pos/gt_vel`:

```bash
python -m lambda_rf radar-vis \
  --input-dir path/to/radar_raw/f60p0GHz_V \
  --output-dir path/to/radar_vis/f60p0GHz_V \
  --show-gt \
  --bs-position 0,0,0
```

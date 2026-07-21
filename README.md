# LAMBDA: A Low-Altitude Multimodal Base Dataset for UAV Sensing and Communication

LAMBDA is a high-fidelity multimodal base dataset for low-altitude UAV sensing
and communication, providing synchronized wireless, visual, motion, LiDAR, and
radar resources across shared trajectories, coordinate systems, and frame
indices.

This repository helps dataset users inspect released CSI files and run common
post-processing steps used by the tutorials and paper experiments, including
MIMO OFDM CSI, radar signal synthesis, and radar visualization.

- Website: https://www.lambda6g.net/
- Documentation: https://www.lambda6g.net/documentation
- Tutorials: https://www.lambda6g.net/tutorials
- Scenarios: https://www.lambda6g.net/scenarios
- Download: https://www.lambda6g.net/download

## Repository Contents

```text
lambda_rf/                 Python package and command line interface
examples/                  Runnable CSI loading and label generation examples
notebooks/                 Tutorial notebooks matching the website walkthroughs
configs/scenarios.json     Example utility configuration
scripts/                   CSI packaging helpers
assets/default_drone_rcs_28ghz.h5
assets/default_drone_rcs_77ghz.h5
tests/                     Unit tests for CSI readers and post-processing math
CONFIG.md                  Configuration reference for the public utilities
CITATION.cff               Citation metadata
REFERENCES.bib             Dataset BibTeX entry
LICENSE                    Apache-2.0 license
pyproject.toml             Python package metadata
```

The main package is `lambda_rf`, and the console command is `lambda-rf`.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Radar generation and visualization use the bundled RCS H5 file and plotting
utilities. Install the optional radar dependencies when using those commands:

```bash
pip install -e ".[radar]"
```

Check that the command line interface is available:

```bash
python -m lambda_rf list-scenarios
lambda-rf list-scenarios
```

## Tutorial Notebooks

The website explains the tasks and dataset context. The runnable tutorial code
is kept in this repository:

```text
notebooks/00_load_csi.ipynb
notebooks/01_mimo_ofdm_csi.ipynb
notebooks/02_generate_radar_and_visualize.ipynb
```

Set the path variables in the first code cell of each notebook to point to your
downloaded LAMBDA data. The notebooks use small limits by default so that users
can verify paths before running a full split.

## Paper Use Cases

The paper's beam-prediction and cross-scene RGB plus LiDAR localization
experiments are implemented under:

```text
examples/paper_use_cases/beam_prediction/
examples/paper_use_cases/cross_scene_localization/
```

These examples use separate PyTorch dependencies and downloaded datasets, so
they are not installed as part of the lightweight `lambda_rf` core package.
See `examples/paper_use_cases/README.md` for data preparation, validation, and
reproduction commands.

## Quick Start

Inspect one released path-level CSI file:

```bash
python examples/load_csi_npz.py path/to/csi_000000.npz
python -m lambda_rf read-csi path/to/csi_000000.npz
```

Generate beam labels from path-level CSI with a DFT codebook baseline:

```bash
python examples/generate_beam_labels.py \
  --input-dir path/to/csi/f4p9GHz_V \
  --output-csv beam_labels.csv \
  --carrier-frequency 4.9e9 \
  --array-shape 8,8 \
  --codebook-shape 16,16
```

Generate final MIMO OFDM CSI from existing path-level CSI:

```bash
python -m lambda_rf mimo-ofdm-csi \
  --input-dir path/to/csi/f60p0GHz_V \
  --output-dir path/to/derived_csi/mimo_ofdm_csi/f60p0GHz_V/rx1x1_tx4x4/sub6_30k_1024 \
  --tx-shape 4,4 \
  --rx-shape 1,1 \
  --profile sub6_30k_1024
```

The default array model is far-field plane-wave steering. Use
`--array-model spherical-wave` to synthesize element-wise near-field phases and
per-antenna-pair delays from compact path geometry fields:

```bash
python -m lambda_rf mimo-ofdm-csi \
  --input-dir path/to/csi/f60p0GHz_V \
  --output-dir path/to/derived_csi/mimo_ofdm_csi/spherical_wave/f60p0GHz_V/rx1x1_tx4x4/sub6_30k_1024 \
  --tx-shape 4,4 \
  --rx-shape 1,1 \
  --profile sub6_30k_1024 \
  --array-model spherical-wave
```

Generate FMCW radar raw cubes from released path-level CSI. Frequency-matched
AirSim default-drone FEKO models are bundled for 28, 60, and 77 GHz:

```bash
python -m lambda_rf radar \
  --input-dir path/to/csi/f28p0GHz_V \
  --output-dir path/to/radar_raw/f28p0GHz_V \
  --imu-dir path/to/imu \
  --chirp-interval 5e-5 \
  --add-noise
```

`--chirp-interval` sets the PRI including idle gap. You can also pass
`--idle-time` to specify only the gap after each chirp. Radar also accepts
`--array-model spherical-wave` to use per-radar-antenna near-field delays from
the same `path_vertices` and `path_interaction_count` fields used by MIMO OFDM
CSI. Theta-linear incidence uses coherent `E_theta`; unsupported radar bands
fail before synthesis unless an explicit frequency-matched `--rcs-model` is
provided.

Render Range-Doppler, Range-Azimuth, and Range-Elevation images:

```bash
python -m lambda_rf radar-vis \
  --input-dir path/to/radar_raw/f28p0GHz_V \
  --output-dir path/to/radar_vis/f28p0GHz_V
```

The website tutorials provide the narrative walkthroughs that correspond to
the notebooks:

```text
https://www.lambda6g.net/tutorials
```

## Configuration

The example configuration is:

```text
configs/scenarios.json
```

It stores defaults used when a command needs an output-root convention, MIMO
shape, carrier-frequency tag, weather tag, OFDM profile, radar settings,
and the bundled AirSim drone RCS model. Most public
commands also accept explicit `--input-dir` and `--output-dir` arguments, which
is the recommended path for downloaded CSI data.

See [CONFIG.md](CONFIG.md) for command options, path conventions, and
output-field details.

## Tests

```bash
python -m unittest discover tests
```

## Citation

If you use LAMBDA or this utility code in research, cite the LAMBDA dataset.
GitHub can read the metadata in [CITATION.cff](CITATION.cff).

Dataset DOI:

```text
https://doi.org/10.57760/sciencedb.36052
```

BibTeX is available in [REFERENCES.bib](REFERENCES.bib).

## License

This utility code is released under the Apache License 2.0. See [LICENSE](LICENSE).

# Use cases

This directory contains two downstream workflows built with LAMBDA:

- `beam_prediction/`: cross-dataset RGB-only 60 GHz beam prediction, trained
  on LAMBDA Open Ground and evaluated on DeepSense Scenario 23.
- `cross_scene_localization/`: cross-scene RGB-LiDAR UAV localization, trained
  on Block 1 and evaluated on Square 1.

They live under `examples/` because they require PyTorch and prepared
experiment datasets. No dataset samples, scene meshes, trained weights, or
DeepSense files are included.

## Environment

Use Python 3.9 or newer, install a PyTorch build compatible with your system
and CUDA version, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Run the commands below from `examples/paper_use_cases/`.

## Prepare the released LAMBDA data

Prepare a separate experiment copy instead of reorganizing the authoritative
downloaded dataset in place.

### Beam prediction

Use the released Open Ground trajectory:

```text
Suburbs/Open Ground/Sunny/1_bs_1_uav_z_traj
```

Extract `cam.zip` into `cam/`, `pose.zip` into `pose/`, and
`csi/f60p0GHz_V.zip` into `csi_60G/`. Keep the released `sensors/` directory:

```text
<LAMBDA_DATA_ROOT>/
`-- Suburbs/Open Ground/Sunny/1_bs_1_uav_z_traj/
    |-- cam/img_*.png
    |-- pose/drone_pose_*.json
    |-- csi_60G/csi_*.npz
    `-- sensors/*.json
```

The default workflow uses this scene, an
ImageNet-initialized ResNet-50 classifier, and the sky-up 64-beam codebook.

### Cross-scene localization

Prepare these two released vertical trajectories:

```text
Urban Area/Block 1/Sunny/1_bs_1_uav_z_traj
Urban Area/Square 1/Sunny/1_bs_1_uav_z_traj
```

For both trajectories, prepare camera frames under `cam/`, then extract
`pose.zip` and `lidar.zip` at the trajectory root. These two archives already
contain their `pose/` and `lidar/` directory wrappers; extracting either archive
into a same-named subdirectory would create an incorrect `pose/pose/` or
`lidar/lidar/` nesting:

```text
<LAMBDA_DATA_ROOT>/Urban Area/<scene>/Sunny/1_bs_1_uav_z_traj/
|-- cam/img_*.png
|-- pose/drone_pose_*.json
|-- lidar/lidar_*.pcd
`-- sensors/*.json
```

The default v20 workflow trains on Block 1 at 60--120 m, tests on Square 1 at
70--110 m, and evaluates at most 500 test samples per height.

## Prepare DeepSense Scenario 23

Obtain Scenario 23 from
[DeepSense 6G](https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23),
extract the archive, and set `DEEPSENSE_ROOT` to the directory that directly
contains `scenario23.csv`:

```text
<DEEPSENSE_ROOT>/
|-- scenario23.csv
`-- unit1/camera_data/*.jpg
```

The CSV must contain `index`, `unit1_rgb`, and `unit1_beam_index`.
`unit1_rgb` is resolved relative to `DEEPSENSE_ROOT`; one-based beam indices
are converted to zero-based class labels. DeepSense files are external inputs
and are not included in this repository.

For results using Scenario 23, follow the
[DeepSense citation guidance](https://www.deepsense6g.net/citations) and cite:

1. A. Alkhateeb et al., "DeepSense 6G: A Large-Scale Real-World Multi-Modal
   Sensing and Communication Dataset," *IEEE Communications Magazine*,
   vol. 61, no. 9, pp. 122-128, 2023,
   [doi:10.1109/MCOM.006.2200730](https://doi.org/10.1109/MCOM.006.2200730).
2. G. Charan et al., "Towards Real-World 6G Drone Communication: Position and
   Camera Aided Beam Prediction," *2022 IEEE Global Communications Conference
   (GLOBECOM)*, pp. 2951-2956, 2022,
   [doi:10.1109/GLOBECOM48099.2022.10000718](https://doi.org/10.1109/GLOBECOM48099.2022.10000718).

## Cross-dataset beam prediction

```bash
python -m beam_prediction \
  --data-root /path/to/prepared/LAMBDA \
  --deepsense-root /path/to/scenario23 \
  --do_few_shot
```

This command uses the default scene, backbone, codebook, and CSV labels. See
`beam_prediction/README.md` for the protocol summary.

## Cross-scene UAV localization

Run the localization workflow with its default configuration:

```bash
python -m cross_scene_localization \
  --data-root /path/to/prepared/LAMBDA \
  --out-dir runs/localization
```

The model, sampling, optimization, and top-k LiDAR settings are provided as CLI
defaults. A startup check verifies the prepared camera/LiDAR poses and image
resolution against the audited configuration. Use `--seed` for one run or
`--seed-list` for repeated runs. See
`cross_scene_localization/README.md` for the workflow summary and use
`python -m cross_scene_localization --help` for the complete interface.

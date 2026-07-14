# Paper use cases

These are the runnable implementations of the two downstream experiments
reported for LAMBDA:

- `beam_prediction/`: RGB-only 60 GHz beam prediction trained on LAMBDA,
  with zero-shot and few-shot evaluation on DeepSense Scenario 23.
- `cross_scene_localization/`: cross-scene RGB heatmap localization with
  base-station LiDAR range association.

The code is intentionally kept under `examples/` instead of the core
`lambda_rf` package because it requires PyTorch and experiment-specific
datasets. No dataset samples, scene meshes, trained weights, or DeepSense
files are included.

## Environment

Use Python 3.9 or newer and install a PyTorch build compatible with your
system and, when applicable, its CUDA version. Then install the remaining
dependencies:

```bash
pip install -r requirements.txt
```

Run commands from this directory so the two packages are importable.

## Prepare the released LAMBDA data

Prepare a separate experiment copy for each use case. Do not reorganize the
authoritative downloaded dataset in place.

### Beam prediction

Use the released Open Ground trajectory:

```text
Suburbs/Open Ground/Sunny/1_bs_1_uav_z_traj
```

Extract `cam.zip` into `cam/`, `pose.zip` into `pose/`, and
`csi/f60p0GHz_V.zip` into `csi_60G/`. Keep the released `sensors/` directory.
The expected layout is:

```text
<LAMBDA_DATA_ROOT>/
└── Suburbs/Open Ground/Sunny/1_bs_1_uav_z_traj/
    ├── cam/img_*.png
    ├── pose/drone_pose_*.json
    ├── csi_60G/csi_*.npz
    └── sensors/*.json
```

### Cross-scene localization

Use these two released vertical trajectories:

```text
Urban Area/Block 1/Sunny/1_bs_1_uav_z_traj
Urban Area/Square 1/Sunny/1_bs_1_uav_z_traj
```

For both trajectories, prepare the released camera frames under `cam/`,
extract `pose.zip` into `pose/`, and extract `lidar.zip` into `lidar/`.
The paper protocol trains on Block 1 at 60--120 m and tests on Square 1 at
70--110 m.

The expected layout for each localization trajectory is:

```text
<LAMBDA_DATA_ROOT>/
└── Urban Area/
    ├── Block 1/Sunny/1_bs_1_uav_z_traj/
    └── Square 1/Sunny/1_bs_1_uav_z_traj/
        ├── cam/img_*.png
        ├── pose/drone_pose_*.json
        ├── lidar/lidar_*.pcd
        └── sensors/*.json
```

## Prepare DeepSense Scenario 23

The beam-prediction experiment uses DeepSense Scenario 23 for zero-shot and
few-shot evaluation. Obtain the dataset from
[DeepSense 6G](https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23):

1. Sign in to the DeepSense 6G website.
2. Under **Download Dataset**, download the Scenario 23 dataset.
3. Extract `scenario23.zip`.
4. Set `DEEPSENSE_ROOT` to the extracted Scenario 23 directory that directly
   contains `scenario23.csv`.

The complete official archive also contains `resources/` and `unit2/`. This
use case reads `scenario23.csv`, the Unit 1 RGB images, and the Unit 1 60 GHz
power files. Its required layout is:

```text
<DEEPSENSE_ROOT>/
|-- scenario23.csv
`-- unit1/
    |-- camera_data/*.jpg
    `-- mmWave_data/*.txt
```

The CSV must contain `index`, `unit1_rgb`, `unit1_pwr_60ghz`, and
`unit1_beam_index`. The two resource columns are interpreted as paths relative
to `DEEPSENSE_ROOT`. Beam indices are one-based in the CSV and are converted
to zero-based class labels by the loader. The default label source is the CSV;
use `--deepsense_label_source power` to derive labels from the power files.

DeepSense files are external inputs and are not included in this repository.

### DeepSense citation

For results that use Scenario 23 with this RGB/mmWave beam-prediction code,
follow the [DeepSense citation guidance](https://www.deepsense6g.net/citations)
and cite both of the following works:

1. A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, J. Morais, U. Demirhan,
   and N. Srinivas, "DeepSense 6G: A Large-Scale Real-World Multi-Modal
   Sensing and Communication Dataset," *IEEE Communications Magazine*,
   vol. 61, no. 9, pp. 122-128, 2023,
   [doi:10.1109/MCOM.006.2200730](https://doi.org/10.1109/MCOM.006.2200730).
2. G. Charan, A. Hredzak, C. Stoddard, B. Berrey, M. Seth, H. Nunez, and
   A. Alkhateeb, "Towards Real-World 6G Drone Communication: Position and
   Camera Aided Beam Prediction," *2022 IEEE Global Communications Conference
   (GLOBECOM)*, pp. 2951-2956, 2022,
   [doi:10.1109/GLOBECOM48099.2022.10000718](https://doi.org/10.1109/GLOBECOM48099.2022.10000718).

## Validate the code

The CLI checks below read no dataset files:

```bash
python -m beam_prediction --help
python -m cross_scene_localization --help
```

## Beam prediction

DeepSense Scenario 23 is an external evaluation dependency and is not part of
LAMBDA. A small path and model run is:

```bash
python -m beam_prediction \
  --data-root /path/to/prepared/LAMBDA \
  --deepsense-root /path/to/scenario23_dev \
  --save_dir runs/beam_smoke \
  --stride 200 \
  --test_stride 20 \
  --limit_train_per_scene 128 \
  --limit_test 256 \
  --epochs 1 \
  --batch_size 16 \
  --num_workers 2 \
  --no_pretrained
```

See `beam_prediction/README.md` and `python -m beam_prediction --help` for the
full experiment interface.

## Cross-scene localization

Set the prepared LAMBDA root before importing or running the localization
package:

```bash
export LAMBDA_DATA_ROOT=/path/to/prepared/LAMBDA
python -m cross_scene_localization \
  --out-dir runs/localization
```

Use `LAMBDA_LOCALIZATION_OUTPUT` to change the default result directory.
See `cross_scene_localization/README.md` and
`python -m cross_scene_localization --help` for all options.

## Scope and validation status

The modular code preserves the original experiment functions and classes.
Checks cover syntax, imports, CLI construction, coordinate transforms,
beam-codebook construction, heatmap decoding, and small real-data runs. Full
training and paper metric reproduction require the prepared datasets and a
suitable CUDA-capable GPU.

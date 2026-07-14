# Cross-scene RGB plus LiDAR localization

This package trains an RGB heatmap model and associates predicted image
locations with base-station LiDAR ranges for cross-scene 3D UAV localization.
It preserves the paper experiment's camera, LiDAR, sampling, and metric
settings.

```bash
export LAMBDA_DATA_ROOT=/path/to/prepared/LAMBDA
python -m cross_scene_localization --help
python -m cross_scene_localization --out-dir runs/localization
```

The paper experiment trains on Block 1 at 60--120 m and tests on Square 1 at
70--110 m. The prepared dataset root must contain those two public
vertical-trajectory directories described in the parent README.

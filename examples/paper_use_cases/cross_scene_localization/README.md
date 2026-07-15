# Cross-scene RGB-LiDAR UAV localization

This package provides a v20 localization workflow. It trains the RGB
U-Net heatmap model on Block 1 at 60--120 m and tests on Square 1 at 70--110 m.
Predicted image locations are associated with base-station LiDAR ranges for 3D
UAV localization.

The protocol uses a fixed validation split stratified by height and image
location, followed by globally nested 5%, 10%, 25%, 50%, 75%, and 100%
training subsets. The Square 1 test set is fixed at up to 500 samples per
height. Median LiDAR association and top-k LiDAR reranking are enabled by
default with the parameters shown in the parent README command.

Pass a prepared dataset root containing the Block 1 and Square 1 vertical
trajectories through `--data-root`, then run the command in the parent README.
The `LAMBDA_DATA_ROOT` environment variable remains available as a fallback.
Before loading samples, the workflow verifies that both scenes' camera/LiDAR
poses and image resolution still match the audited geometry constants.
Use `--seed` for a single run or pass comma-separated values to `--seed-list`
for repeated runs.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


DATASET_ROOT_CONFIGURED = bool(os.environ.get("LAMBDA_DATA_ROOT"))
DATASET_ROOT = Path(os.environ.get("LAMBDA_DATA_ROOT", "__LAMBDA_DATA_ROOT_NOT_SET__")).expanduser()


BLOCK_ROOT = str(DATASET_ROOT / "Urban Area" / "Block 1" / "Sunny" / "1_bs_1_uav_z_traj")


SQUARE_ROOT = str(DATASET_ROOT / "Urban Area" / "Square 1" / "Sunny" / "1_bs_1_uav_z_traj")


DEFAULT_OUTPUT_DIR = os.environ.get("LAMBDA_LOCALIZATION_OUTPUT", "runs/cross_scene_localization")


FX = FY = 805.6


CX, CY = 960.0, 540.0


IMG_W, IMG_H = 1920, 1080


CROP_SIZE = 1080


CROP_U_MIN = (IMG_W - CROP_SIZE) // 2


CROP_U_MAX = IMG_W - CROP_U_MIN


CROP_V_MIN = 0


CROP_V_MAX = IMG_H


SCENE_CAM_PARAMS = {
    BLOCK_ROOT: {"pitch": 40, "yaw": -180, "roll": 0},
    SQUARE_ROOT: {"pitch": 30, "yaw": 0, "roll": 0},
}


SCENE_LIDAR_SETTINGS_WORLD = {
    BLOCK_ROOT: np.array([-8.10000381469727, 157.0, 35.70000076293945], dtype=np.float32),
    SQUARE_ROOT: np.array([-258.899938964844, -77.0999984741211, 41.7999923706055], dtype=np.float32),
}


SCENE_LIDAR_PARAMS = {
    BLOCK_ROOT: {"pitch": 0, "yaw": -180, "roll": 0},
    SQUARE_ROOT: {"pitch": 0, "yaw": 0, "roll": 0},
}


SCENE_LIDAR_TRANSFORM_MODE = {
    BLOCK_ROOT: "settings_local_ned_euler",
    SQUARE_ROOT: "settings_local_world_yflip_zflip",
}


SCENE_NAMES = {
    BLOCK_ROOT: "Block_1",
    SQUARE_ROOT: "Square_1",
}


SCENE_HEIGHTS = {
    BLOCK_ROOT: ("60m", "70m", "80m", "90m", "100m", "110m", "120m"),
    SQUARE_ROOT: ("70m", "80m", "90m", "100m", "110m"),
}


@dataclass
class ExperimentSpec:
    name: str
    protocol: str
    train_root: str
    train_heights: Tuple[str, ...]
    test_root: str
    test_heights: Tuple[str, ...]


EXPERIMENTS = [
    ExperimentSpec(
        name="cross_scene_block1_square1_mid_70_80_90_100_110",
        protocol="v20_rgb_lidar_global_sampling",
        train_root=BLOCK_ROOT,
        train_heights=("60m", "70m", "80m", "90m", "100m", "110m", "120m"),
        test_root=SQUARE_ROOT,
        test_heights=("70m", "80m", "90m", "100m", "110m"),
    ),
]


FORMAL_EXPERIMENT_NAME = EXPERIMENTS[0].name
FORMAL_SCALE_UNIT = "percent_global"
FORMAL_MODEL_BACKBONE = "rgb_unet"
FORMAL_SELECTION_METRIC = "mean_px"
FORMAL_LIDAR_RANGE_STRATEGY = "median"

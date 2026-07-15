from __future__ import annotations

import os
import re


DATA_ROOT_DEFAULT = os.environ.get("LAMBDA_DATA_ROOT")


DEEPSENSE_ROOT_DEFAULT = os.environ.get("DEEPSENSE_ROOT")


SCENE_PATHS = {
    "Open_Ground": "Suburbs/Open Ground/Sunny/1_bs_1_uav_z_traj",
}


FORMAL_TRAIN_SCENE = "Open_Ground"
FORMAL_TEST_DATASET = "deepsense"
FORMAL_BACKBONE = "resnet50_paper"
FORMAL_CODEBOOK_FRAME = "sky_up"
FORMAL_LAMBDA_LABEL_MODE = "strongest_path_angle"
FORMAL_DEEPSENSE_LABEL_SOURCE = "csv"


FRAME_RE = re.compile(r"(\d{6})")

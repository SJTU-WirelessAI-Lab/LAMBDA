from __future__ import annotations

import os
import re


DATA_ROOT_DEFAULT = os.environ.get("LAMBDA_DATA_ROOT")


DEEPSENSE_ROOT_DEFAULT = os.environ.get("DEEPSENSE_ROOT")


SCENE_PATHS = {
    "Open_Ground": "Suburbs/Open Ground/Sunny/1_bs_1_uav_z_traj",
}


FRAME_RE = re.compile(r"(\d{6})")

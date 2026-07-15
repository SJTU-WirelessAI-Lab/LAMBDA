import argparse
import ast
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
USE_CASE_ROOT = ROOT / "examples" / "paper_use_cases"


def load_definitions(path: Path, names: Sequence[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
    ]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f"Missing definitions in {path}: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class ReleasedUseCaseTest(unittest.TestCase):
    def test_beam_defaults_match_released_protocol(self):
        namespace = load_definitions(
            USE_CASE_ROOT / "beam_prediction" / "cli.py",
            ["build_parser"],
            {
                "argparse": argparse,
                "DATA_ROOT_DEFAULT": "prepared-lambda",
                "DEEPSENSE_ROOT_DEFAULT": "scenario23",
                "FORMAL_BACKBONE": "resnet50_paper",
                "FORMAL_CODEBOOK_FRAME": "sky_up",
                "FORMAL_DEEPSENSE_LABEL_SOURCE": "csv",
                "FORMAL_LAMBDA_LABEL_MODE": "strongest_path_aoa",
                "FORMAL_TEST_DATASET": "deepsense",
                "FORMAL_TRAIN_SCENE": "Open_Ground",
            },
        )
        args = namespace["build_parser"]().parse_args([])
        self.assertEqual(args.backbone, "resnet50_paper")
        self.assertEqual(args.k_shots, [64, 128, 256, 512, 1024])
        self.assertTrue(args.few_shot_reset_head)

    def test_deepsense_labels_are_validated_instead_of_clipped(self):
        namespace = load_definitions(
            USE_CASE_ROOT / "beam_prediction" / "data.py",
            ["validate_deepsense_beam_labels"],
            {"np": np, "List": List, "Dict": Dict},
        )
        validate = namespace["validate_deepsense_beam_labels"]
        np.testing.assert_array_equal(
            validate(
                [
                    {"index": "first", "unit1_beam_index": "1"},
                    {"index": "last", "unit1_beam_index": "64"},
                ]
            ),
            np.array([0, 63], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, r"\[1, 64\].*bad.*65"):
            validate([{"index": "bad", "unit1_beam_index": "65"}])

    def test_localization_defaults_enable_topk_lidar(self):
        namespace = load_definitions(
            USE_CASE_ROOT / "cross_scene_localization" / "cli.py",
            ["build_parser"],
            {
                "argparse": argparse,
                "os": os,
                "DEFAULT_OUTPUT_DIR": "runs/localization",
                "FORMAL_EXPERIMENT_NAME": "block1_to_square1",
                "EXPERIMENTS": [SimpleNamespace(name="block1_to_square1")],
                "FORMAL_LIDAR_RANGE_STRATEGY": "median",
                "FORMAL_MODEL_BACKBONE": "rgb_unet",
                "FORMAL_SCALE_UNIT": "percent_global",
                "FORMAL_SELECTION_METRIC": "mean_px",
            },
        )
        args = namespace["build_parser"]().parse_args(["--data-root", "prepared-lambda"])
        self.assertTrue(args.eval_topk_lidar)
        self.assertEqual(args.topk_lidar_k, 10)
        self.assertEqual(args.topk_lidar_nms_kernel, 9)

    def test_v20_global_subsets_are_deterministic_and_nested(self):
        namespace = load_definitions(
            USE_CASE_ROOT / "cross_scene_localization" / "sampling.py",
            ["_stable_group_seed", "select_nested_train_subset_global_percent"],
            {
                "random": random,
                "List": List,
                "Dict": Dict,
                "Any": Any,
                "Optional": Optional,
            },
        )
        select = namespace["select_nested_train_subset_global_percent"]
        pool = [
            {"scene": "Block_1", "height": f"{60 + 10 * (i % 7)}m", "fid_int": i}
            for i in range(100)
        ]
        args = SimpleNamespace()
        selected_5 = select(pool, 5, 2026, args)
        selected_25 = select(pool, 25, 2026, args)
        selected_100 = select(pool, 100, 2026, args)
        repeated_25 = select(pool, 25, 2026, args)

        ids = lambda rows: {int(row["fid_int"]) for row in rows}
        self.assertEqual(len(selected_5), 5)
        self.assertEqual(len(selected_25), 25)
        self.assertEqual(len(selected_100), 100)
        self.assertLessEqual(ids(selected_5), ids(selected_25))
        self.assertLessEqual(ids(selected_25), ids(selected_100))
        self.assertEqual(selected_25, repeated_25)

    def test_sensor_gate_accepts_match_and_rejects_drift(self):
        geometry_path = USE_CASE_ROOT / "cross_scene_localization" / "geometry.py"
        scene_cam_params = {}
        scene_lidar_params = {}
        scene_lidar_world = {}
        namespace = load_definitions(
            geometry_path,
            [
                "_load_sensor_pose",
                "_quaternion_to_rotation",
                "_sensor_pose_rotation",
                "_rotation_error_degrees",
                "validate_sensor_configuration",
            ],
            {
                "json": json,
                "math": math,
                "os": os,
                "Path": Path,
                "np": np,
                "Tuple": Tuple,
                "Sequence": Sequence,
                "IMG_W": 1920,
                "IMG_H": 1080,
                "SCENE_CAM_PARAMS": scene_cam_params,
                "SCENE_LIDAR_PARAMS": scene_lidar_params,
                "SCENE_LIDAR_SETTINGS_WORLD": scene_lidar_world,
            },
        )
        validate = namespace["validate_sensor_configuration"]

        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            sensors = scene / "sensors"
            camera = scene / "cam"
            sensors.mkdir(parents=True)
            camera.mkdir()
            scene_key = str(scene)
            scene_cam_params[scene_key] = {"pitch": 0, "yaw": 0, "roll": 0}
            scene_lidar_params[scene_key] = {"pitch": 0, "yaw": 0, "roll": 0}
            scene_lidar_world[scene_key] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            def write_pose(name: str, position: dict) -> None:
                payload = {
                    "sensor_name": name,
                    "world_transform": {
                        "position": position,
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                }
                (sensors / f"{name}_world_pose.json").write_text(json.dumps(payload), encoding="utf-8")

            position = {"x": 1.0, "y": 2.0, "z": 3.0}
            write_pose("RoofCam", position)
            write_pose("RoofLidar", position)

            try:
                from PIL import Image
            except ImportError:
                self.skipTest("Pillow is not installed")
            Image.new("RGB", (8, 6)).save(camera / "img_000000.png")

            validate([scene_key], expected_image_size=(8, 6))
            write_pose("RoofLidar", {"x": 1.1, "y": 2.0, "z": 3.0})
            with self.assertRaisesRegex(RuntimeError, "RoofLidar position"):
                validate([scene_key], expected_image_size=(8, 6))

    def test_sensor_gate_runs_before_experiments(self):
        tree = ast.parse(
            (USE_CASE_ROOT / "cross_scene_localization" / "cli.py").read_text(encoding="utf-8-sig")
        )
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [
            (node.func.id, node.lineno)
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        gate_line = min(line for name, line in calls if name == "validate_sensor_configuration")
        run_line = min(line for name, line in calls if name == "run_experiment")
        self.assertLess(gate_line, run_line)


if __name__ == "__main__":
    unittest.main()

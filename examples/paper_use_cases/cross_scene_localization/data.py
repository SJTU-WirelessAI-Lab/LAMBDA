from __future__ import annotations

import glob
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import CROP_SIZE, SCENE_HEIGHTS, SCENE_NAMES
from .geometry import load_camera_pose, project_to_pixel_raw, raw_uv_to_img, world_to_camcv
from .lidar import lidar_file_for_fid


def parse_height(scene_root: str, fid_int: int) -> str:
    heights = SCENE_HEIGHTS[scene_root]
    idx = min(fid_int // 4201, len(heights) - 1)
    return heights[idx]


def build_scene_index(
    scene_root: str,
    heights: Tuple[str, ...],
    img_size: int,
    frame_stride: int,
    max_samples_per_height: Optional[int],
    seed: int,
    require_lidar_file: bool,
) -> Tuple[List[Dict[str, Any]], Counter]:
    cam_world, R_ned = load_camera_pose(scene_root)
    rgb_files = sorted(glob.glob(os.path.join(scene_root, "cam", "img_*.png")))
    wanted_heights = set(heights)
    samples: List[Dict[str, Any]] = []
    skipped = Counter()

    for rgb_path in rgb_files:
        fid = os.path.basename(rgb_path).replace("img_", "").replace(".png", "")
        try:
            fid_int = int(fid)
        except ValueError:
            skipped["bad_fid"] += 1
            continue

        height = parse_height(scene_root, fid_int)
        if height not in wanted_heights:
            skipped["height_filtered"] += 1
            continue
        if frame_stride > 1 and fid_int % frame_stride != 0:
            skipped["frame_stride_filtered"] += 1
            continue

        pose_path = os.path.join(scene_root, "pose", f"drone_pose_{fid}.json")
        if not os.path.exists(pose_path):
            skipped["missing_pose"] += 1
            continue
        lidar_path = lidar_file_for_fid(scene_root, fid_int)
        if require_lidar_file and lidar_path is None:
            skipped["missing_lidar"] += 1
            continue

        with open(pose_path, "r", encoding="utf-8") as f:
            pos = json.load(f)["position"]
        uav_world = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
        xyz_cv = world_to_camcv(uav_world, cam_world, R_ned)
        uv = project_to_pixel_raw(xyz_cv, require_crop_visible=True)
        if uv is None:
            skipped["not_visible_in_model_crop"] += 1
            continue
        u_raw, v_raw = uv
        u_img, v_img = raw_uv_to_img(u_raw, v_raw, img_size)
        range_gt = float(np.linalg.norm(xyz_cv))

        samples.append({
            "scene_root": scene_root,
            "scene": SCENE_NAMES.get(scene_root, os.path.basename(scene_root)),
            "fid": fid,
            "fid_int": fid_int,
            "height": height,
            "rgb_path": rgb_path,
            "lidar_path": lidar_path,
            "uav_world": uav_world,
            "xyz_cv": xyz_cv,
            "range_gt": range_gt,
            "u_raw": float(u_raw),
            "v_raw": float(v_raw),
            "u_img": float(u_img),
            "v_img": float(v_img),
        })

    if max_samples_per_height is not None:
        rng = random.Random(seed)
        limited: List[Dict[str, Any]] = []
        by_h: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in samples:
            by_h[s["height"]].append(s)
        for h, hs in by_h.items():
            rng.shuffle(hs)
            limited.extend(hs[:max_samples_per_height])
        rng.shuffle(limited)
        samples = limited

    return samples, skipped


def print_dataset_summary(name: str, samples: List[Dict[str, Any]], skipped: Counter) -> None:
    print(f"\n[Dataset: {name}]")
    print(f"  samples: {len(samples)}")
    print(f"  skipped: {dict(skipped)}")
    if not samples:
        return
    hcnt = Counter(s["height"] for s in samples)
    scnt = Counter(s["scene"] for s in samples)
    us = np.array([s["u_img"] for s in samples])
    vs = np.array([s["v_img"] for s in samples])
    xyz = np.stack([s["xyz_cv"] for s in samples])
    lidar_exists = sum(1 for s in samples if s.get("lidar_path"))
    print(f"  per scene: {dict(sorted(scnt.items()))}")
    print(f"  per height: {dict(sorted(hcnt.items()))}")
    print(f"  lidar files available: {lidar_exists}/{len(samples)}")
    print(f"  u_img [{us.min():.1f}, {us.max():.1f}] mean={us.mean():.1f}")
    print(f"  v_img [{vs.min():.1f}, {vs.max():.1f}] mean={vs.mean():.1f}")
    print(f"  X/Y/Z cv mean={xyz.mean(axis=0)} range_mean={np.linalg.norm(xyz, axis=1).mean():.2f}")


def make_gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    hm = np.exp(-((xs[None, :] - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    cx_int = int(np.clip(round(float(cx)), 0, w - 1))
    cy_int = int(np.clip(round(float(cy)), 0, h - 1))
    hm[cy_int, cx_int] = 1.0
    return hm


class RGBHeatmapDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], img_size: int, hm_size: int, hm_sigma: float, augment: bool, allow_hflip: bool = True):
        self.samples = list(samples)
        self.img_size = img_size
        self.hm_size = hm_size
        self.hm_sigma = hm_sigma
        self.augment = augment
        self.allow_hflip = allow_hflip
        self.center_crop = transforms.CenterCrop(CROP_SIZE)
        self.rgb_resize = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.to_tensor = transforms.ToTensor()
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Restored V19.5 augmentation. The restored augmentation variant in V19.7
        # increased cross-scene long-tail false positives in our audit.
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.06),
            transforms.RandomGrayscale(p=0.05),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        do_hflip = self.augment and self.allow_hflip and (random.random() < 0.5)
        u_img = float(s["u_img"])
        v_img = float(s["v_img"])
        if do_hflip:
            u_img = (self.img_size - 1) - u_img

        scale = self.hm_size / self.img_size
        hm = make_gaussian_heatmap(self.hm_size, self.hm_size, u_img * scale, v_img * scale, self.hm_sigma)
        heatmap = torch.from_numpy(hm[None, :, :])

        img = Image.open(s["rgb_path"]).convert("RGB")
        if self.augment:
            img = self.color_aug(img)
        if do_hflip:
            img = TF.hflip(img)
        img = self.rgb_resize(self.center_crop(img))
        rgb = self.rgb_normalize(self.to_tensor(img))

        meta = {
            "scene_root": s["scene_root"],
            "scene": s["scene"],
            "fid": s["fid"],
            "fid_int": s["fid_int"],
            "height": s["height"],
            "rgb_path": s["rgb_path"],
            "lidar_path": s.get("lidar_path"),
            "u_gt_img": float(s["u_img"]),
            "v_gt_img": float(s["v_img"]),
            "u_gt_raw": float(s["u_raw"]),
            "v_gt_raw": float(s["v_raw"]),
            "range_gt": float(s["range_gt"]),
            "xyz_cv": s["xyz_cv"].astype(np.float32),
            "uav_world": s["uav_world"].astype(np.float32),
        }
        return rgb, heatmap, meta


def collate_heatmap(batch):
    xs, hms, metas = zip(*batch)
    return torch.stack(xs), torch.stack(hms), list(metas)

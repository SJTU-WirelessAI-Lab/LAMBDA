from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import (
    FORMAL_CODEBOOK_FRAME,
    FORMAL_DEEPSENSE_LABEL_SOURCE,
    FORMAL_LAMBDA_LABEL_MODE,
    FRAME_RE,
    SCENE_PATHS,
)
from .geometry import PhotoULACodebook, load_roofcam_pose


class LambdaRGB60BeamDataset(Dataset):
    """LAMBDA RGB + 60 GHz photo-codebook labels for one scene."""

    def __init__(
        self,
        data_root: str,
        scene: str,
        codebook: PhotoULACodebook,
        stride: int = 1,
        img_size: int = 224,
        augment: bool = False,
        center_codebook_crop: bool = False,
        rgb_hfov: float = 110.0,
        cache_dir: Optional[str] = "runs/cache_rgb60",
        limit: Optional[int] = None,
        label_mode: str = FORMAL_LAMBDA_LABEL_MODE,
        codebook_frame: str = FORMAL_CODEBOOK_FRAME,
    ) -> None:
        super().__init__()
        if label_mode != FORMAL_LAMBDA_LABEL_MODE:
            raise ValueError(f"The released experiment uses label_mode={FORMAL_LAMBDA_LABEL_MODE!r}")
        if codebook_frame != FORMAL_CODEBOOK_FRAME:
            raise ValueError(f"The released experiment uses codebook_frame={FORMAL_CODEBOOK_FRAME!r}")
        if scene not in SCENE_PATHS:
            raise KeyError(f"Unknown LAMBDA scene {scene!r}; choices={list(SCENE_PATHS)}")
        self.scene = scene
        self.codebook = codebook
        self.label_mode = label_mode
        self.codebook_frame = codebook_frame
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.scene_dir = Path(data_root) / SCENE_PATHS[scene]
        self.cam_dir = self.scene_dir / "cam"
        self.csi60_dir = self.scene_dir / "csi_60G"
        self.pose_dir = self.scene_dir / "pose"
        for d in [self.cam_dir, self.csi60_dir, self.pose_dir]:
            if not d.is_dir():
                raise FileNotFoundError(f"Missing LAMBDA folder: {d}")

        cam_ids = self._frame_ids(self.cam_dir, "img_*.png")
        csi_ids = self._frame_ids(self.csi60_dir, "csi_*.npz")
        pose_ids = self._frame_ids(self.pose_dir, "drone_pose_*.json")
        frames = sorted(cam_ids & csi_ids & pose_ids)[:: max(1, stride)]
        if limit is not None:
            frames = frames[: int(limit)]
        if not frames:
            raise RuntimeError(f"No aligned LAMBDA RGB/CSI frames in {self.scene_dir}")
        self.frames = frames

        self.t_cam, self.R_cam = load_roofcam_pose(self.scene_dir)
        cache = self._cache_candidates(
            f".rgb60_photo_labels_ula{codebook.n_ant}_beam{codebook.n_beams}"
            f"_fov{int(round(codebook.fov_deg))}_{label_mode}_{codebook_frame}_stride{stride}.npy"
        )
        self.labels, self.strongest_angles = self._load_or_build_labels(cache)

        self.img_tf = build_image_transform(
            img_size,
            augment=augment,
            center_codebook_crop=center_codebook_crop,
            rgb_hfov=rgb_hfov,
            codebook_fov=codebook.fov_deg,
        )

    @staticmethod
    def _frame_ids(folder: Path, glob: str) -> set:
        prefix, suffix = glob.split("*")[0], glob.split("*")[-1]
        out = set()
        for f in folder.iterdir():
            if not f.is_file():
                continue
            if not (f.name.startswith(prefix) and f.name.endswith(suffix)):
                continue
            m = FRAME_RE.search(f.name)
            if m:
                out.add(m.group(1))
        return out

    def _cache_candidates(self, filename: str) -> List[Path]:
        out: List[Path] = []
        if self.cache_dir is not None:
            out.append(self.cache_dir / self.scene / filename)
        out.append(self.scene_dir / filename)
        return out

    def _load_or_build_labels(self, cache_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        for path in cache_paths:
            if not path.exists():
                continue
            try:
                stored = np.load(path, allow_pickle=True).item()
                if list(stored.get("frames", [])) == self.frames:
                    return stored["labels"], stored["strongest_angles"]
            except Exception:
                pass

        labels = np.zeros(len(self.frames), dtype=np.int64)
        angles = np.zeros(len(self.frames), dtype=np.float32)
        t0 = time.time()
        for i, fid in enumerate(self.frames):
            with np.load(self.csi60_dir / f"csi_{fid}.npz") as data:
                _, angle = self.codebook.best_beam_from_csi(
                    {k: data[k] for k in data.keys()},
                    self.R_cam,
                    self.scene,
                    self.codebook_frame,
                )
            labels[i] = self.codebook.angle_to_index(np.array([angle]))[0]
            angles[i] = angle
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{self.scene}] RGB60 labels {i+1}/{len(self.frames)} ({rate:.0f} fps)")

        payload = {"frames": self.frames, "labels": labels, "strongest_angles": angles}
        for path in cache_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, payload)
                break
            except (PermissionError, OSError) as e:
                print(f"  [{self.scene}] cache write skipped: {path} ({e})")
        return labels, angles

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fid = self.frames[idx]
        with Image.open(self.cam_dir / f"img_{fid}.png") as im:
            image = self.img_tf(im.convert("RGB"))
        return {
            "image": image,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "frame_id": fid,
            "source": self.scene,
        }


def validate_deepsense_beam_labels(
    rows: List[Dict[str, object]],
    n_beams: int = 64,
) -> np.ndarray:
    """Convert one-based DeepSense labels after validating their declared range."""
    labels_one_based = np.array([int(row["unit1_beam_index"]) for row in rows], dtype=np.int64)
    invalid = np.flatnonzero((labels_one_based < 1) | (labels_one_based > n_beams))
    if invalid.size:
        i = int(invalid[0])
        frame = rows[i].get("index", i)
        value = int(labels_one_based[i])
        raise ValueError(
            f"DeepSense unit1_beam_index must be in [1, {n_beams}]; "
            f"frame {frame!r} has {value}."
        )
    return labels_one_based - 1


class DeepSenseRGB60BeamDataset(Dataset):
    """DeepSense scenario23 RGB + 60 GHz beam labels."""

    def __init__(
        self,
        deepsense_root: str,
        csv_name: str = "scenario23.csv",
        stride: int = 1,
        img_size: int = 224,
        augment: bool = False,
        center_codebook_crop: bool = False,
        rgb_hfov: float = 110.0,
        codebook_fov: float = 90.0,
        label_source: str = FORMAL_DEEPSENSE_LABEL_SOURCE,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        if label_source != FORMAL_DEEPSENSE_LABEL_SOURCE:
            raise ValueError(f"The released experiment uses label_source={FORMAL_DEEPSENSE_LABEL_SOURCE!r}")
        self.root = Path(deepsense_root)
        self.scene = "DeepSense_scenario23"
        rows = self._read_csv(self.root / csv_name)[:: max(1, stride)]
        if limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            raise RuntimeError(f"No DeepSense rows in {self.root / csv_name}")
        self.rows = rows
        self.rgb_paths = [self.root / r["unit1_rgb"] for r in rows]
        for p in self.rgb_paths[:1]:
            if not p.is_file():
                raise FileNotFoundError(f"Missing DeepSense RGB image: {p}")

        self.labels = validate_deepsense_beam_labels(rows)
        self.frames = [str(r["index"]) for r in rows]
        self.img_tf = build_image_transform(
            img_size,
            augment=augment,
            center_codebook_crop=center_codebook_crop,
            rgb_hfov=rgb_hfov,
            codebook_fov=codebook_fov,
        )

    @staticmethod
    def _read_csv(csv_path: Path) -> List[Dict[str, object]]:
        if not csv_path.is_file():
            raise FileNotFoundError(f"DeepSense CSV not found: {csv_path}")
        out: List[Dict[str, object]] = []
        required = ("index", "unit1_rgb", "unit1_beam_index")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            missing = [c for c in required if c not in (reader.fieldnames or [])]
            if missing:
                raise RuntimeError(f"DeepSense CSV missing columns {missing}: {csv_path}")
            for r in reader:
                out.append(
                    {
                        "index": int(r["index"]),
                        "unit1_rgb": r["unit1_rgb"].lstrip("./"),
                        "unit1_beam_index": int(r["unit1_beam_index"]),
                    }
                )
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with Image.open(self.rgb_paths[idx]) as im:
            image = self.img_tf(im.convert("RGB"))
        return {
            "image": image,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "frame_id": self.frames[idx],
            "source": self.scene,
        }


class CenterCodebookCrop:
    """Crop central horizontal FoV corresponding to the codebook region.

    The crop is deterministic and centered, so it preserves beam left/right
    ordering while removing image margins outside the codebook FoV.
    """

    def __init__(self, rgb_hfov: float = 110.0, codebook_fov: float = 90.0) -> None:
        if rgb_hfov <= 0 or codebook_fov <= 0:
            raise ValueError("FoV values must be positive")
        self.rgb_hfov = float(rgb_hfov)
        self.codebook_fov = float(codebook_fov)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.codebook_fov >= self.rgb_hfov:
            return img
        w, h = img.size
        half_ratio = math.tan(math.radians(self.codebook_fov / 2.0)) / math.tan(
            math.radians(self.rgb_hfov / 2.0)
        )
        half_ratio = min(1.0, max(0.05, half_ratio))
        half_w = 0.5 * w * half_ratio
        left = int(round(w / 2.0 - half_w))
        right = int(round(w / 2.0 + half_w))
        return img.crop((left, 0, right, h))


def build_image_transform(
    img_size: int,
    augment: bool,
    center_codebook_crop: bool = False,
    rgb_hfov: float = 110.0,
    codebook_fov: float = 90.0,
) -> transforms.Compose:
    ops: List[object] = []
    if center_codebook_crop:
        ops.append(CenterCodebookCrop(rgb_hfov=rgb_hfov, codebook_fov=codebook_fov))
    ops.append(transforms.Resize((img_size, img_size)))
    if augment:
        # Keep geometry fixed: beam labels are tied to photo-horizontal
        # position, so random crop/flip would corrupt supervision.
        ops.append(transforms.ColorJitter(0.25, 0.25, 0.20, 0.05))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)

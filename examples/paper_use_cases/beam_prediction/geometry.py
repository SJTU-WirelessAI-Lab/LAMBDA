from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def quat_to_rotation(q: Dict[str, float]) -> np.ndarray:
    """Quaternion dict -> 3x3 rotation matrix."""
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_roofcam_pose(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = scene_dir / "sensors" / "RoofCam_world_pose.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RoofCam pose: {path}")
    with open(path) as f:
        tf = json.load(f)["world_transform"]
    t = np.array(
        [tf["position"]["x"], tf["position"]["y"], tf["position"]["z"]],
        dtype=np.float64,
    )
    R = quat_to_rotation(tf["orientation"])
    return t, R


def world_to_photo_frame(pos_world: np.ndarray, t_cam: np.ndarray, R_cam: np.ndarray) -> np.ndarray:
    """Map UE world positions/vectors to the RGB photo frame.

    Convention used throughout this script:
        photo +X = image center optical axis
        photo +Y = image right direction (roll is zero in these captures)
        photo +Z = image up direction

    The same convention was used for the codebook visualizations in
    runs/beam_rgb60_viz/beam_codebook_*.  With this repository's quaternion
    helper, world->photo is rel @ R.T.
    """
    return (pos_world.astype(np.float64) - t_cam[None, :]) @ R_cam.T


def directions_world_to_photo_frame(
    vec_world: np.ndarray,
    R_cam: np.ndarray,
    scene: str = "",
    codebook_frame: str = "photo",
) -> np.ndarray:
    """Map UE world direction vectors to the photo frame used by the codebook.

    Most LAMBDA z-trace scenes use RoofCam local +X as the photo optical axis
    and local +Y as image right. Square_3 stores the camera with local +Z as
    the optical axis; keep the same photo convention by remapping local
    (X,Y,Z) -> photo (Z,Y,-X).
    """
    vec_world = vec_world.astype(np.float64)
    local = vec_world @ R_cam.T
    if codebook_frame == "sky_up":
        # Sky-pointing 1D ULA: broadside is UE-world +Z and the array scans
        # along the photo horizontal right direction.  For a ULA, the phase is
        # controlled by the projection onto the array axis; synthesize a
        # forward/right pair whose atan2 gives asin(projection).
        right = np.clip(local[:, 1], -1.0, 1.0)
        forward = np.sqrt(np.maximum(0.0, 1.0 - right * right))
        return np.stack([forward, right, np.zeros_like(right)], axis=1)
    if codebook_frame != "photo":
        raise ValueError(f"Unknown codebook frame {codebook_frame!r}")
    if scene == "Square_3":
        return np.stack([local[:, 2], local[:, 1], -local[:, 0]], axis=1)
    return local


class PhotoULACodebook:
    """1xN horizontal ULA with K beams uniformly spaced in photo angle."""

    def __init__(
        self,
        n_ant: int = 16,
        n_beams: int = 64,
        fov_deg: float = 90.0,
    ) -> None:
        self.n_ant = int(n_ant)
        self.n_beams = int(n_beams)
        self.fov_deg = float(fov_deg)
        self.beam_width_deg = self.fov_deg / self.n_beams
        left = -self.fov_deg / 2.0
        self.centers_deg = left + (np.arange(self.n_beams) + 0.5) * self.beam_width_deg
        self.centers_rad = np.deg2rad(self.centers_deg)
        self.elem = np.arange(self.n_ant, dtype=np.float64)
        self.W = self._steering_from_angles(self.centers_rad)

    def angle_to_index(self, angle_deg: np.ndarray) -> np.ndarray:
        idx = np.floor((angle_deg + self.fov_deg / 2.0) / self.beam_width_deg)
        return np.clip(idx.astype(np.int64), 0, self.n_beams - 1)

    def _steering_from_angles(self, angle_rad: np.ndarray) -> np.ndarray:
        # ULA lies along photo +Y (image right).  The spatial frequency is
        # sin(alpha), where alpha is horizontal angle from the optical axis.
        u = np.sin(angle_rad).reshape(1, -1)
        phase = np.pi * self.elem.reshape(-1, 1) @ u
        return np.exp(1j * phase) / math.sqrt(self.n_ant)

    def best_beam_from_csi(
        self,
        csi_npz: Dict[str, np.ndarray],
        R_cam: np.ndarray,
        scene: str = "",
        codebook_frame: str = "photo",
    ) -> Tuple[int, float]:
        """Return (best_beam, strongest_path_angle_deg).

        LAMBDA's theta_r/phi_r vector points along the arriving wave direction,
        i.e. approximately UAV->BS for the LoS path.  The photo-horizontal
        codebook needs BS->UAV, so we negate that direction before rotating it
        into the photo frame.
        """
        a = csi_npz["a_real"].reshape(-1) + 1j * csi_npz["a_imag"].reshape(-1)
        theta = csi_npz["theta_r"].reshape(-1)
        phi = csi_npz["phi_r"].reshape(-1)
        valid = csi_npz["valid"].reshape(-1).astype(bool)
        if not np.any(valid):
            return self.n_beams // 2, 0.0

        a = a[valid]
        theta = theta[valid]
        phi = phi[valid]

        arrival_world = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=1,
        )
        bs_to_path_world = -arrival_world
        path_photo = directions_world_to_photo_frame(bs_to_path_world, R_cam, scene, codebook_frame)
        angles = np.arctan2(path_photo[:, 1], path_photo[:, 0])

        A = self._steering_from_angles(angles)
        y = (self.W.conj().T @ A) @ a
        power = np.abs(y) ** 2

        strongest = int(np.argmax(np.abs(a)))
        strongest_angle_deg = float(np.rad2deg(angles[strongest]))
        return int(np.argmax(power)), strongest_angle_deg

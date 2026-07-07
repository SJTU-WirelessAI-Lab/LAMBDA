from __future__ import annotations

import numpy as np


def representative_interactions(interactions: np.ndarray | None, num_paths: int) -> np.ndarray | None:
    if interactions is None:
        return None
    values = np.asarray(interactions, dtype=np.int32)
    if values.size == 0:
        return np.zeros((0, num_paths), dtype=np.int32)
    if values.shape[-1] != num_paths:
        raise ValueError(f"interactions must end with path dimension {num_paths}, got {values.shape}")
    if values.ndim == 1:
        return values.reshape(1, num_paths)
    if values.ndim == 2:
        return values

    steps = values.shape[0]
    middle = int(np.prod(values.shape[1:-1]))
    flat = values.reshape(steps, middle, num_paths)
    reduced = flat[:, 0, :].copy()
    for step_idx in range(steps):
        for path_idx in range(num_paths):
            nonzero = np.flatnonzero(flat[step_idx, :, path_idx] != 0)
            if nonzero.size:
                reduced[step_idx, path_idx] = flat[step_idx, nonzero[0], path_idx]
    return reduced


def representative_vertices(vertices: np.ndarray | None, num_paths: int) -> np.ndarray | None:
    if vertices is None:
        return None
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim < 3 or values.shape[-1] != 3 or values.shape[-2] != num_paths:
        raise ValueError(f"vertices must have shape (..., {num_paths}, 3), got {values.shape}")
    if values.ndim == 3:
        return values

    steps = values.shape[0]
    middle = int(np.prod(values.shape[1:-2]))
    flat = values.reshape(steps, middle, num_paths, 3)
    reduced = np.full((steps, num_paths, 3), np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(flat), axis=-1)
    for step_idx in range(steps):
        for path_idx in range(num_paths):
            rows = np.flatnonzero(finite[step_idx, :, path_idx])
            if rows.size:
                reduced[step_idx, path_idx, :] = flat[step_idx, rows[0], path_idx, :]
    return reduced


def compact_path_vertices(
    interactions: np.ndarray | None,
    vertices: np.ndarray | None,
    num_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the first and last interaction vertices needed by spherical-wave arrays."""
    endpoints = np.full((2, num_paths, 3), np.nan, dtype=np.float32)
    interaction_count = np.zeros(num_paths, dtype=np.int32)
    if num_paths <= 0:
        return endpoints, interaction_count

    reduced_vertices = representative_vertices(vertices, num_paths)
    if reduced_vertices is None or reduced_vertices.size == 0:
        return endpoints, interaction_count

    reduced_interactions = representative_interactions(interactions, num_paths)
    finite = np.all(np.isfinite(reduced_vertices), axis=-1)
    if reduced_interactions is not None and reduced_interactions.shape == finite.shape:
        valid_steps = finite & (reduced_interactions != 0)
    else:
        valid_steps = finite

    for path_idx in range(num_paths):
        steps = np.flatnonzero(valid_steps[:, path_idx])
        interaction_count[path_idx] = int(steps.size)
        if steps.size:
            endpoints[0, path_idx, :] = reduced_vertices[steps[0], path_idx, :]
            endpoints[1, path_idx, :] = reduced_vertices[steps[-1], path_idx, :]

    return endpoints, interaction_count

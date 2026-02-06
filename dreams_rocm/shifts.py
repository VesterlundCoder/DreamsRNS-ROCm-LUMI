"""
Shift generation for Dreams pipeline.

Shifts are starting points for the walk in the (n, k) plane.
Each shift defines a base value added to the axis at step 0.
"""

from typing import Tuple, Optional, List
import numpy as np


def generate_shifts(
    n_shifts: int = 100,
    dim: int = 1,
    method: str = "grid",
    bounds: Tuple[int, int] = (-1000, 1000),
    seed: int = 42,
    cmf_idx: int = 0,
) -> np.ndarray:
    """Generate shift values for walk exploration.

    Args:
        n_shifts: Number of shifts to generate.
        dim: Number of dimensions (axes).
        method: "grid", "random", or "sphere".
        bounds: (min, max) inclusive range for shift values.
        seed: Random seed (combined with cmf_idx for per-CMF determinism).
        cmf_idx: CMF index for seed variation.

    Returns:
        int32 array of shape (n_shifts, dim).
    """
    rng = np.random.default_rng(seed=seed + cmf_idx * 1000)

    if method == "random":
        return rng.integers(
            bounds[0], bounds[1] + 1,
            size=(n_shifts, dim), dtype=np.int32
        )

    elif method == "grid":
        if dim == 1:
            vals = np.linspace(bounds[0], bounds[1], n_shifts, dtype=np.int32)
            return vals.reshape(-1, 1)
        else:
            side = int(np.ceil(n_shifts ** (1 / dim)))
            grids = [
                np.linspace(bounds[0], bounds[1], side, dtype=np.int32)
                for _ in range(dim)
            ]
            mesh = np.meshgrid(*grids, indexing='ij')
            shifts = np.stack([m.flatten() for m in mesh], axis=1)
            # Trim to exact count, shuffled for variety
            if len(shifts) > n_shifts:
                idx = rng.permutation(len(shifts))[:n_shifts]
                shifts = shifts[idx]
            return shifts.astype(np.int32)

    elif method == "sphere":
        radius = (bounds[1] - bounds[0]) / 2
        center = (bounds[0] + bounds[1]) / 2
        points = rng.standard_normal((n_shifts, dim))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid div by zero
        points = points / norms
        radii = rng.uniform(0, 1, (n_shifts, 1)) ** (1 / dim)
        points = center + points * radii * radius
        return points.astype(np.int32)

    else:
        raise ValueError(f"Unknown shift method: {method}")


def generate_shifts_deterministic(
    n_shifts: int = 100,
    start: int = 1,
    step: int = 1,
    dim: int = 1,
) -> np.ndarray:
    """Generate simple sequential shifts for reproducibility.

    Shifts are [start, start+step, start+2*step, ...] for dim=1.
    For dim>1, uses a grid pattern.

    Args:
        n_shifts: Number of shifts.
        start: Starting value.
        step: Step between consecutive shifts.
        dim: Number of dimensions.

    Returns:
        int32 array of shape (n_shifts, dim).
    """
    if dim == 1:
        vals = np.arange(start, start + n_shifts * step, step, dtype=np.int32)
        return vals[:n_shifts].reshape(-1, 1)
    else:
        side = int(np.ceil(n_shifts ** (1 / dim)))
        grids = [
            np.arange(start, start + side * step, step, dtype=np.int32)[:side]
            for _ in range(dim)
        ]
        mesh = np.meshgrid(*grids, indexing='ij')
        shifts = np.stack([m.flatten() for m in mesh], axis=1)
        return shifts[:n_shifts].astype(np.int32)

"""
Exhaustive trajectory and shift generators for full-dimension CMF sweeps.

Supports arbitrary dimension d (designed for dim=4 via 2F2 and dim=6 via 3F3).

Trajectories:
  - Primitive integer vectors in Z^d with L_inf <= k_max
  - Canonical normalization: gcd-reduced, first nonzero component positive
  - Deterministic ordering by L2 norm then lexicographic

Shifts:
  - Sobol low-discrepancy sequence in [0,1)^d
  - Quantized to rationals with small denominators
  - Deduplicated exactly

Preset exhaust sizes:
  dim=4 (2F2 / 3x3): ~1120 trajectories (k_max=3), 512 shifts
  dim=6 (3F3 / 4x4): ~7448 trajectories (k_max=2), 1024 shifts
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Set, Optional

import numpy as np


# ── Trajectories (primitive integer vectors) ─────────────────────────────

def _gcd_many(nums: Iterable[int]) -> int:
    g = 0
    for x in nums:
        g = math.gcd(g, abs(int(x)))
    return g


def canonical_direction(v: Tuple[int, ...]) -> Tuple[int, ...]:
    """Normalize a d-dimensional direction to canonical form.

    1. Divide by gcd of absolute values (remove scaling)
    2. First nonzero component must be positive (remove ± ambiguity)
    """
    if all(x == 0 for x in v):
        raise ValueError("Zero vector not allowed")

    g = _gcd_many(v)
    if g > 1:
        v = tuple(int(x // g) for x in v)

    for x in v:
        if x != 0:
            if x < 0:
                v = tuple(-int(t) for t in v)
            break
    return v


def generate_primitive_trajectories(
    dim: int,
    k_max: int,
    *,
    include_axis: bool = True,
) -> List[Tuple[int, ...]]:
    """Generate all primitive directions v in Z^dim with L_inf(v) <= k_max.

    Args:
        dim: Number of dimensions.
        k_max: Maximum absolute value per component.
        include_axis: Include axis-aligned unit vectors.

    Returns:
        Sorted list of unique canonical direction tuples.
    """
    dirs: Set[Tuple[int, ...]] = set()
    rng = range(-k_max, k_max + 1)

    for v in itertools.product(rng, repeat=dim):
        if all(x == 0 for x in v):
            continue
        if not include_axis:
            if sum(1 for x in v if x != 0) == 1:
                continue
        dirs.add(canonical_direction(tuple(int(x) for x in v)))

    def _sort_key(v):
        l2 = sum(x * x for x in v)
        return (l2, v)

    return sorted(dirs, key=_sort_key)


# ── Shifts (Sobol + quantize to rationals) ───────────────────────────────

@dataclass(frozen=True)
class RationalShift:
    """A shift represented as per-component rational numbers."""
    nums: Tuple[int, ...]
    dens: Tuple[int, ...]

    def as_floats(self) -> Tuple[float, ...]:
        return tuple(n / d for n, d in zip(self.nums, self.dens))

    def as_tokens(self) -> List[str]:
        """Convert to Euler2AI-style shift tokens like '1_2', '-3_4'."""
        return [f"{n}_{d}" for n, d in zip(self.nums, self.dens)]

    def __repr__(self) -> str:
        parts = [f"{n}/{d}" for n, d in zip(self.nums, self.dens)]
        return f"RationalShift({', '.join(parts)})"


def _sobol_points(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate Sobol low-discrepancy points in [0,1)^dim."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        pts = sampler.random(n)
        return pts
    except Exception:
        # Fallback: R-sequence (Kronecker low-discrepancy)
        pts = np.empty((n, dim), dtype=np.float64)
        phi = 1.0
        for _ in range(10):
            phi = (1.0 + phi) ** (1.0 / (dim + 1))
        alphas = np.array([1.0 / (phi ** (j + 1)) for j in range(dim)])
        for i in range(n):
            for j in range(dim):
                x = (0.5 + (i + 1) * alphas[j] + 0.000123 * seed) % 1.0
                pts[i, j] = x
        return pts


def _quantize_rational(
    x: float,
    denoms: List[int],
    mode: str = "centered",
) -> Tuple[int, int]:
    """Quantize a float to the nearest rational with denominator in denoms.

    Args:
        x: Value in [0, 1).
        denoms: Allowed denominators.
        mode: "centered" maps to [-1/2, 1/2), "unit" keeps [0, 1).
    """
    y = (x - 0.5) if mode == "centered" else x

    best_num, best_den = 0, denoms[0]
    best_err = float("inf")
    for d in denoms:
        n = int(round(y * d))
        err = abs(y - n / d)
        if err < best_err:
            best_err = err
            best_num, best_den = n, d

    return best_num, best_den


def generate_unique_shifts(
    dim: int,
    n: int,
    *,
    denoms: Optional[List[int]] = None,
    seed: int = 0,
    mode: str = "centered",
) -> List[RationalShift]:
    """Generate n unique rational shifts in dim dimensions.

    Uses Sobol low-discrepancy sequence, quantized to small-denominator
    rationals, then deduplicated.

    Args:
        dim: Number of dimensions.
        n: Target number of unique shifts.
        denoms: Allowed denominators (default: [2, 3, 4, 6, 8]).
        seed: Random seed for Sobol scramble.
        mode: "centered" for [-1/2, 1/2) style, "unit" for [0, 1).

    Returns:
        List of n unique RationalShift objects.
    """
    if denoms is None:
        denoms = [2, 3, 4, 6, 8]

    # Oversample 4x to survive deduplication
    pts = _sobol_points(n * 4, dim, seed=seed)
    out: List[RationalShift] = []
    seen: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

    for row in pts:
        nums = []
        dens_list = []
        for j in range(dim):
            num, den = _quantize_rational(float(row[j]), denoms, mode=mode)
            nums.append(num)
            dens_list.append(den)

        rs = RationalShift(nums=tuple(nums), dens=tuple(dens_list))
        key = (rs.nums, rs.dens)
        if key in seen:
            continue
        seen.add(key)
        out.append(rs)
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(
            f"Could not produce {n} unique shifts (got {len(out)}). "
            f"Increase oversample or add more denoms."
        )
    return out


def generate_unique_shifts_halves(
    dim: int,
    n: int,
    *,
    seed: int = 0,
) -> List[RationalShift]:
    """Generate shifts using only halves (denom=2), Euler2AI compatible.

    This is the simplest representation: each component is k/2 for k in Z.
    """
    return generate_unique_shifts(dim, n, denoms=[2], seed=seed, mode="centered")


# ── Preset exhaust configurations ────────────────────────────────────────

EXHAUST_PRESETS = {
    4: {"k_max": 3, "n_shifts_full": 512},    # 2F2: ~1,120 traj
    5: {"k_max": 3, "n_shifts_full": 512},    # 3F2: ~8,161 traj
    6: {"k_max": 2, "n_shifts_full": 1024},   # 3F3: ~7,448 traj
    7: {"k_max": 2, "n_shifts_full": 1024},   # 4F3: ~37,969 traj
    9: {"k_max": 1, "n_shifts_full": 1024},   # 5F4: ~9,841 traj
}


def exhaust_trajectories(dim: int) -> List[Tuple[int, ...]]:
    """Get the full primitive trajectory set for a given dimension.

    dim=4: k_max=3, ~1,120 trajectories
    dim=5: k_max=3, ~8,161 trajectories
    dim=6: k_max=2, ~7,448 trajectories
    dim=7: k_max=2, ~37,969 trajectories
    dim=9: k_max=1, ~9,841 trajectories
    """
    preset = EXHAUST_PRESETS.get(dim)
    if preset is None:
        raise ValueError(f"No exhaust preset for dim={dim}. Available: {list(EXHAUST_PRESETS.keys())}")
    return generate_primitive_trajectories(dim, preset["k_max"])


def exhaust_shifts(dim: int, seed: int = 0) -> List[RationalShift]:
    """Get the full shift set for a given dimension.

    Args:
        dim: 4, 5, 6, 7, or 9.
        seed: Sobol seed.

    Returns:
        List of unique RationalShift objects.
    """
    preset = EXHAUST_PRESETS.get(dim)
    if preset is None:
        raise ValueError(f"No exhaust preset for dim={dim}.")

    n = preset["n_shifts_full"]

    return generate_unique_shifts(dim, n, seed=seed, mode="centered")


# ── Summary / introspection ──────────────────────────────────────────────

def exhaust_summary(dim: int) -> dict:
    """Return trajectory/shift counts and compute estimates for a dimension."""
    trajs = exhaust_trajectories(dim)
    preset = EXHAUST_PRESETS[dim]
    n_shifts = preset["n_shifts_full"]
    return {
        "dim": dim,
        "k_max": preset["k_max"],
        "n_trajectories": len(trajs),
        "n_shifts": n_shifts,
        "runs_per_cmf": len(trajs) * n_shifts,
    }


if __name__ == "__main__":
    from .cmf_generator import FAMILY_CONFIG
    for fam, cfg in FAMILY_CONFIG.items():
        d = cfg["dim"]
        s = exhaust_summary(d)
        print(f"\n{fam} (dim={d}):")
        print(f"  Trajectories: {s['n_trajectories']:>6,} (k_max={s['k_max']})")
        print(f"  Shifts:       {s['n_shifts']:>6,}")
        print(f"  Runs/CMF:     {s['runs_per_cmf']:>12,}")

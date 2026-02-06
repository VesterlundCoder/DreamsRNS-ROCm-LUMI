"""
Trajectory generation and normalization for Dreams pipeline.

A trajectory is a 2-D direction (dn, dk) defining a walk direction
in the (n, k) plane.  To avoid duplicates and scaled copies:

1. Reduce by gcd: (dn, dk) := (dn/g, dk/g) where g = gcd(|dn|, |dk|)
2. Fix sign convention: prefer dn > 0; if dn == 0, then dk > 0
3. Ensure uniqueness via a set/hash

Provides:
  - A generator to produce at least 1000 unique trajectories
  - Deterministic ordering (sorted by (dn, dk))
  - Reproducibility across runs
"""

from typing import List, Tuple, Optional, Set
from math import gcd, atan2


def normalize_trajectory(dn: int, dk: int) -> Tuple[int, int]:
    """Normalize a 2-D direction to canonical form.

    Rules:
      - Reduce by gcd of absolute values
      - If dn < 0, negate both (prefer dn > 0)
      - If dn == 0 and dk < 0, negate dk (prefer dk > 0)

    Args:
        dn: Step in n-direction.
        dk: Step in k-direction.

    Returns:
        Normalized (dn, dk) tuple.

    Raises:
        ValueError if both dn and dk are 0.
    """
    if dn == 0 and dk == 0:
        raise ValueError("Zero trajectory (0, 0) is not valid")

    g = gcd(abs(dn), abs(dk))
    dn, dk = dn // g, dk // g

    # Sign convention: prefer dn > 0
    if dn < 0:
        dn, dk = -dn, -dk
    elif dn == 0 and dk < 0:
        dk = -dk

    return (dn, dk)


def generate_trajectories(
    count: int = 1000,
    max_component: int = 50,
    include_axes: bool = True,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Generate deterministic, unique, normalized 2-D trajectories.

    Strategy:
      1. Start with axis-aligned trajectories (1,0), (0,1)
      2. Enumerate (dn, dk) pairs in expanding shells
      3. Normalize and deduplicate
      4. Sort by (dn, dk) for deterministic ordering

    Args:
        count: Minimum number of trajectories to generate.
        max_component: Maximum absolute value per component.
        include_axes: Include axis-aligned directions.
        seed: Not used (generation is fully deterministic), kept for API compat.

    Returns:
        Sorted list of at least `count` unique normalized trajectories.
    """
    seen: Set[Tuple[int, int]] = set()
    result: List[Tuple[int, int]] = []

    def _add(dn: int, dk: int):
        if dn == 0 and dk == 0:
            return
        normed = normalize_trajectory(dn, dk)
        if normed not in seen:
            seen.add(normed)
            result.append(normed)

    # Phase 1: axis-aligned
    if include_axes:
        _add(1, 0)
        _add(0, 1)

    # Phase 2: expand in shells by max(|dn|, |dk|)
    for shell in range(1, max_component + 1):
        if len(result) >= count:
            break
        # All (dn, dk) where max(|dn|, |dk|) == shell
        for dn in range(-shell, shell + 1):
            for dk in [-shell, shell]:
                _add(dn, dk)
            if abs(dn) == shell:
                for dk in range(-shell + 1, shell):
                    _add(dn, dk)

    # Phase 3: if still short, fill with larger components
    if len(result) < count:
        for dn in range(max_component + 1, max_component * 2 + 1):
            for dk in range(-dn, dn + 1):
                _add(dn, dk)
                if len(result) >= count:
                    break
            if len(result) >= count:
                break

    # Sort for deterministic ordering: by (dn, dk) lexicographic
    result.sort()

    return result[:count] if len(result) > count else result


def generate_trajectories_by_angle(
    count: int = 1000,
    max_component: int = 50,
) -> List[Tuple[int, int]]:
    """Generate trajectories sorted by angle then by norm.

    Alternative ordering that groups trajectories by direction.

    Args:
        count: Number of trajectories.
        max_component: Maximum component magnitude.

    Returns:
        List of unique normalized trajectories sorted by angle.
    """
    trajs = generate_trajectories(count, max_component)

    def angle_key(t: Tuple[int, int]) -> Tuple[float, float]:
        dn, dk = t
        return (atan2(dk, dn), dn * dn + dk * dk)

    trajs.sort(key=angle_key)
    return trajs


def trajectories_to_directions(
    trajectories: List[Tuple[int, int]],
) -> List[List[int]]:
    """Convert trajectory list to direction arrays for the walk kernel.

    Each trajectory (dn, dk) becomes a direction vector [dn, dk]
    for the 2-D walk.

    Args:
        trajectories: List of (dn, dk) tuples.

    Returns:
        List of [dn, dk] lists.
    """
    return [list(t) for t in trajectories]

"""
Target constants for Dreams delta computation.

Provides both:
  - Low-precision float64 values for quick checks
  - High-precision mpmath values for CPU verification
"""

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Float64 target constants
# ---------------------------------------------------------------------------

ZETA_TARGETS = {
    "zeta3": 1.2020569031595942,     # Apéry's constant ζ(3)
    "zeta5": 1.0369277551433699,     # ζ(5)
    "zeta7": 1.0083492773819228,     # ζ(7)
    "pi": math.pi,
    "e": math.e,
    "phi": (1 + math.sqrt(5)) / 2,
    "ln2": math.log(2),
    "catalan": 0.9159655941772190,   # Catalan's constant
}


def get_target_value(name: str) -> float:
    """Get float64 target constant by name.

    Args:
        name: Key in ZETA_TARGETS (e.g. "zeta3", "pi").

    Returns:
        float64 value.

    Raises:
        KeyError if name is not a known target.
    """
    if name not in ZETA_TARGETS:
        raise KeyError(
            f"Unknown target '{name}'. "
            f"Available: {list(ZETA_TARGETS.keys())}"
        )
    return ZETA_TARGETS[name]


def get_target_value_hp(name: str, dps: int = 100):
    """Get high-precision target constant using mpmath.

    Args:
        name: Key in ZETA_TARGETS.
        dps: Decimal places for mpmath computation.

    Returns:
        mpmath.mpf value at requested precision.
        Falls back to float64 if mpmath unavailable.
    """
    try:
        import mpmath
        mpmath.mp.dps = dps

        hp_map = {
            "zeta3": lambda: mpmath.zeta(3),
            "zeta5": lambda: mpmath.zeta(5),
            "zeta7": lambda: mpmath.zeta(7),
            "pi": lambda: mpmath.pi,
            "e": lambda: mpmath.e,
            "phi": lambda: (1 + mpmath.sqrt(5)) / 2,
            "ln2": lambda: mpmath.log(2),
            "catalan": lambda: mpmath.catalan,
        }

        if name in hp_map:
            return hp_map[name]()
        else:
            raise KeyError(f"No high-precision definition for '{name}'")

    except ImportError:
        return get_target_value(name)


def compute_dreams_delta(
    p_val: float,
    q_val: float,
    target: float,
    log_scale: float = 0.0,
) -> tuple:
    """Compute Dreams delta from float trajectory values.

    Dreams delta = -(1 + log|err| / log|q|)
    Higher delta = better convergence.

    Args:
        p_val: Numerator from trajectory P[0, m-1].
        q_val: Denominator from trajectory P[1, m-1].
        target: Target constant.
        log_scale: Accumulated log of normalization scale.

    Returns:
        (delta, log_q) tuple.
    """
    if not math.isfinite(p_val) or not math.isfinite(q_val) or abs(q_val) < 1e-300:
        return -1e10, 0.0

    est = p_val / q_val
    abs_err = abs(est - target)

    log_abs_q = log_scale + math.log(abs(q_val))

    if abs_err < 1e-300 or log_abs_q <= 0.0:
        return (100.0 if abs_err < 1e-300 else -1e10), log_abs_q

    delta = -(1.0 + math.log(abs_err) / log_abs_q)
    return delta, log_abs_q

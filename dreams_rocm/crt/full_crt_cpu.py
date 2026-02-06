"""
Full CRT reconstruction on CPU for positive-result verification.

When partial CRT indicates delta > 0 (a "positive"), we escalate to
this module which:
1. Reconstructs the full big integer from all K primes
2. Computes high-precision delta using mpmath
3. Verifies against zeta(3), zeta(5), zeta(7)
"""

from typing import Dict, Any, Optional, List
import math

from dreams_rocm.rns.reference import crt_reconstruct, crt_reconstruct_signed
from .delta_targets import get_target_value_hp, ZETA_TARGETS


def full_crt_reconstruct(
    residues: List[int],
    primes: List[int],
    signed: bool = True,
) -> int:
    """Full CRT reconstruction using all primes.

    Args:
        residues: Array of K residues.
        primes: Array of K primes.
        signed: If True, interpret as signed integer in [-M/2, M/2).

    Returns:
        Reconstructed Python int (arbitrary precision).
    """
    if signed:
        return crt_reconstruct_signed(residues, primes)
    return crt_reconstruct(residues, primes)


def full_crt_verify(
    p_val: int,
    q_val: int,
    target_name: str = "zeta3",
    dps: int = 100,
    p_residues: Optional[List[int]] = None,
    q_residues: Optional[List[int]] = None,
    primes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Full-precision verification of a candidate hit.

    Can be called with either:
    - Direct p_val, q_val integers (already reconstructed)
    - RNS residues + primes (will reconstruct here)

    Args:
        p_val: Numerator integer (or 0 if using residues).
        q_val: Denominator integer (or 0 if using residues).
        target_name: Key into ZETA_TARGETS (e.g. "zeta3").
        dps: Decimal places for mpmath.
        p_residues: Optional RNS residues for p.
        q_residues: Optional RNS residues for q.
        primes: Optional prime moduli.

    Returns:
        Dict with: verified, delta, ratio_str, p_str, q_str,
                   target_name, dps, decision.
    """
    # Reconstruct from residues if provided
    if p_residues is not None and q_residues is not None and primes is not None:
        p_val = full_crt_reconstruct(p_residues, primes, signed=True)
        q_val = full_crt_reconstruct(q_residues, primes, signed=True)

    result: Dict[str, Any] = {
        "target_name": target_name,
        "dps": dps,
        "p_val": str(p_val),
        "q_val": str(q_val),
    }

    if q_val == 0:
        result.update(verified=False, delta=-1e10,
                      ratio_str="inf", decision="drop")
        return result

    # Try high-precision verification with mpmath
    try:
        import mpmath
        mpmath.mp.dps = dps

        target_hp = get_target_value_hp(target_name, dps)
        p_mp = mpmath.mpf(p_val)
        q_mp = mpmath.mpf(q_val)
        ratio = p_mp / q_mp
        abs_err = abs(ratio - target_hp)

        # Compute delta
        log_q = float(mpmath.log(abs(q_mp))) if q_val != 0 else 0.0

        if abs_err == 0:
            delta = 100.0
        elif log_q <= 0:
            delta = -1e10
        else:
            delta = -(1.0 + float(mpmath.log(abs_err)) / log_q)

        result.update(
            verified=True,
            delta=delta,
            abs_err=float(abs_err),
            ratio_str=mpmath.nstr(ratio, 30),
            log_q=log_q,
            decision="positive" if delta > 0 else "drop",
        )
    except ImportError:
        # mpmath not available - use float64 fallback
        ratio = p_val / q_val
        target_f = ZETA_TARGETS.get(target_name, math.pi)
        abs_err = abs(ratio - target_f)
        log_q = math.log(abs(q_val)) if q_val != 0 else 0.0

        if abs_err < 1e-300:
            delta = 100.0
        elif log_q <= 0:
            delta = -1e10
        else:
            delta = -(1.0 + math.log(abs_err) / log_q)

        result.update(
            verified=True,
            delta=delta,
            abs_err=abs_err,
            ratio_str=str(ratio),
            log_q=log_q,
            decision="positive" if delta > 0 else "drop",
            note="mpmath not available, used float64",
        )

    return result


def verify_against_all_targets(
    p_val: int,
    q_val: int,
    dps: int = 100,
) -> List[Dict[str, Any]]:
    """Verify a candidate against zeta(3), zeta(5), zeta(7).

    Returns list of verification results, one per target.
    """
    results = []
    for target_name in ["zeta3", "zeta5", "zeta7"]:
        r = full_crt_verify(p_val, q_val, target_name=target_name, dps=dps)
        results.append(r)
    return results

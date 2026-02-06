"""
Partial CRT reconstruction for fast delta-proxy computation.

Uses a small subset of primes (K_small) to reconstruct enough bits
for a float64 ratio estimate, avoiding full big-int CRT on GPU.

Strategy:
  1. Take residues from first K_small primes (typically 4-8)
  2. Reconstruct via iterative CRT (cheap: ~K_small big-int muls)
  3. Convert to float64 for p/q ratio
  4. Compute Dreams delta = -(1 + log|err| / log|q|)
"""

from typing import List, Dict, Any, Optional
import math

from dreams_rocm.rns.reference import crt_reconstruct


def partial_crt_reconstruct(
    residues: List[int],
    primes: List[int],
    K_small: Optional[int] = None,
) -> int:
    """Reconstruct integer from first K_small residues/primes.

    Args:
        residues: Full residue array (only first K_small used).
        primes: Full prime array (only first K_small used).
        K_small: Number of primes to use. Defaults to len(primes).

    Returns:
        Reconstructed integer (may wrap if true value > product of K_small primes).
    """
    if K_small is None:
        K_small = len(primes)
    K_small = min(K_small, len(residues), len(primes))
    return crt_reconstruct(residues[:K_small], primes[:K_small])


def partial_crt_delta_proxy(
    p_residues: List[int],
    q_residues: List[int],
    primes: List[int],
    target: float,
    K_small: Optional[int] = None,
    log_scale: float = 0.0,
) -> Dict[str, Any]:
    """Compute Dreams delta proxy from RNS residues using partial CRT.

    Uses K_small primes for quick reconstruction, then float64 ratio.

    Args:
        p_residues: Residues of numerator p for each prime.
        q_residues: Residues of denominator q for each prime.
        primes: Prime moduli.
        target: Target constant (e.g. zeta(3)).
        K_small: Number of primes for partial CRT (default: all).
        log_scale: Accumulated log-scale from float shadow normalization.

    Returns:
        Dictionary with keys: delta, ratio, log_q, abs_err, decision.
    """
    if K_small is None:
        K_small = min(6, len(primes))

    p_val = partial_crt_reconstruct(p_residues, primes, K_small)
    q_val = partial_crt_reconstruct(q_residues, primes, K_small)

    result: Dict[str, Any] = {
        "p_recon": p_val,
        "q_recon": q_val,
        "K_small": K_small,
    }

    # Guard: denominator zero
    if q_val == 0:
        result.update(delta=-1e10, ratio=float("inf"), log_q=0.0,
                      abs_err=float("inf"), decision="drop")
        return result

    ratio = p_val / q_val
    abs_err = abs(ratio - target)

    # log|q| including accumulated scale
    try:
        log_q = log_scale + math.log(abs(q_val)) if q_val != 0 else 0.0
    except (ValueError, OverflowError):
        log_q = 0.0

    result["ratio"] = ratio
    result["abs_err"] = abs_err

    # Dreams delta = -(1 + log|err| / log|q|)
    if abs_err < 1e-300:
        delta = 100.0  # Perfect match (numerically)
    elif log_q <= 0.0:
        delta = -1e10
    else:
        try:
            delta = -(1.0 + math.log(abs_err) / log_q)
        except (ValueError, OverflowError):
            delta = -1e10

    result["delta"] = delta
    result["log_q"] = log_q

    # Triage decision
    result["decision"] = "escalate" if delta > 0 else "drop"

    return result


def batch_partial_crt_delta(
    p_residues_batch,
    q_residues_batch,
    primes: List[int],
    target: float,
    K_small: int = 6,
    log_scales=None,
    delta_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Batch version of partial_crt_delta_proxy.

    Args:
        p_residues_batch: List of residue lists, one per shift.
        q_residues_batch: List of residue lists, one per shift.
        primes: Prime moduli.
        target: Target constant.
        K_small: Primes for partial CRT.
        log_scales: Optional log-scale per shift.
        delta_threshold: Only return results where delta > threshold.

    Returns:
        List of result dicts for entries above threshold.
    """
    B = len(p_residues_batch)
    results = []
    for b in range(B):
        ls = float(log_scales[b]) if log_scales is not None else 0.0
        r = partial_crt_delta_proxy(
            p_residues_batch[b], q_residues_batch[b],
            primes, target, K_small=K_small, log_scale=ls,
        )
        if r["delta"] > delta_threshold:
            r["shift_idx"] = b
            results.append(r)
    return results

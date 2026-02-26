#!/usr/bin/env python3
"""
precision_engine.py — Multiprecision matching engine for Level-2 confirmation.

Pipeline:
  1. CRT → exact big-integer (p, q) with signed-range fix
  2. Two-step matching:
     a) Stage 1: moderate precision (120 dps) against precomputed constants
     b) Stage 2: high precision (500 dps) only for near-misses from stage 1
  3. Constants precomputed once as decimal strings to avoid recomputation.

Requires: mpmath (always available), gmpy2 (optional, faster CRT/MPZ).
"""
from __future__ import annotations
import math
from typing import List, Dict, Tuple, Optional, Any

# ── Try gmpy2 for fast big-int arithmetic; fall back to Python ints ──
try:
    import gmpy2
    from gmpy2 import mpz, invert as _gmp_invert
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False

import mpmath


# ═══════════════════════════════════════════════════════════════════════
# CRT RECONSTRUCTION (signed range)
# ═══════════════════════════════════════════════════════════════════════

def _modinv_python(a: int, m: int) -> int:
    """Extended-GCD modular inverse (pure Python fallback)."""
    g, x, _ = _xgcd(a % m, m)
    if g != 1:
        raise ValueError(f"No inverse: gcd({a},{m})={g}")
    return x % m


def _xgcd(a: int, b: int):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _xgcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def crt_reconstruct(residues: List[int], primes: List[int]) -> int:
    """Chinese Remainder Theorem → exact integer in balanced signed range.

    Given residues r_i mod p_i, reconstructs x in [0, M) where M = ∏p_i,
    then maps to balanced range: if x > M/2, x := x - M.

    Uses gmpy2 for speed if available.
    """
    K = len(primes)
    assert len(residues) == K

    if HAS_GMPY2:
        M = mpz(1)
        for p in primes:
            M *= mpz(p)
        x = mpz(0)
        for i in range(K):
            pi = mpz(primes[i])
            Mi = M // pi
            yi = _gmp_invert(Mi, pi)
            x = (x + mpz(residues[i]) * Mi * yi) % M
        x = int(x)
        M = int(M)
    else:
        M = 1
        for p in primes:
            M *= p
        x = 0
        for i in range(K):
            pi = primes[i]
            Mi = M // pi
            yi = _modinv_python(Mi, pi)
            x = (x + residues[i] * Mi * yi) % M

    # Balanced signed range: map [0, M) → [-M/2, M/2)
    half = M >> 1
    if x > half:
        x -= M

    return x


def crt_matrix_entry(residues_per_prime: List[List[int]],
                     primes: List[int],
                     row: int, col: int, m: int) -> int:
    """CRT-reconstruct a single (row, col) matrix entry from per-prime residues.

    residues_per_prime[k] = flat row-major m×m array of residues mod primes[k].
    """
    idx = row * m + col
    r_list = [residues_per_prime[k][idx] for k in range(len(primes))]
    return crt_reconstruct(r_list, primes)


# ═══════════════════════════════════════════════════════════════════════
# PRECOMPUTED CONSTANT BANK (decimal strings, computed once)
# ═══════════════════════════════════════════════════════════════════════

_CONSTANT_CACHE: Dict[int, Dict[str, str]] = {}

SCALAR_MULTIPLIERS = [
    ("1", 1),
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("6", 6),
    ("1/2", mpmath.mpf("0.5")),
    ("1/3", mpmath.mpf(1) / 3),
    ("1/4", mpmath.mpf("0.25")),
    ("1/6", mpmath.mpf(1) / 6),
    ("1/12", mpmath.mpf(1) / 12),
]


def precompute_constants(dps: int) -> Dict[str, str]:
    """Compute all target constants at the given decimal precision.

    Returns dict mapping name → decimal string (e.g. "3.14159265358979...").
    Cached so repeated calls at the same dps are free.
    """
    if dps in _CONSTANT_CACHE:
        return _CONSTANT_CACHE[dps]

    mpmath.mp.dps = dps + 10  # guard digits
    base = {
        "pi":          mpmath.pi,
        "e":           mpmath.e,
        "ln2":         mpmath.log(2),
        "ln3":         mpmath.log(3),
        "ln5":         mpmath.log(5),
        "ln10":        mpmath.log(10),
        "euler_gamma": mpmath.euler,
        "catalan":     mpmath.catalan,
        "zeta2":       mpmath.zeta(2),
        "zeta3":       mpmath.zeta(3),
        "zeta4":       mpmath.zeta(4),
        "zeta5":       mpmath.zeta(5),
        "zeta6":       mpmath.zeta(6),
        "zeta7":       mpmath.zeta(7),
        "zeta8":       mpmath.zeta(8),
        "zeta9":       mpmath.zeta(9),
        "sqrt2":       mpmath.sqrt(2),
        "sqrt3":       mpmath.sqrt(3),
        "sqrt5":       mpmath.sqrt(5),
        "phi":         (1 + mpmath.sqrt(5)) / 2,
        "pi_sq":       mpmath.pi ** 2,
        "1/pi":        1 / mpmath.pi,
        "pi/4":        mpmath.pi / 4,
    }

    bank: Dict[str, str] = {}
    for bname, bval in base.items():
        for slabel, smul in SCALAR_MULTIPLIERS:
            if slabel == "1":
                key = bname
            else:
                key = f"{slabel}*{bname}"
            val = bval * smul
            bank[key] = mpmath.nstr(val, dps, strip_zeros=False)

    _CONSTANT_CACHE[dps] = bank
    return bank


# ═══════════════════════════════════════════════════════════════════════
# TWO-STEP MATCHING
# ═══════════════════════════════════════════════════════════════════════

def match_two_step(
    p_exact: int,
    q_exact: int,
    stage1_dps: int = 120,
    stage2_dps: int = 500,
    stage1_threshold: float = 1e-20,
    stage2_threshold: float = 1e-100,
) -> List[Dict[str, Any]]:
    """Two-step multiprecision matching.

    Stage 1: Convert p/q to stage1_dps-digit float, match against all constants.
    Stage 2: For near-misses from stage 1, re-evaluate at stage2_dps digits.

    Args:
        p_exact: CRT-reconstructed numerator (signed big int).
        q_exact: CRT-reconstructed denominator (signed big int).
        stage1_dps: decimal digits for initial screening.
        stage2_dps: decimal digits for confirmation of near-misses.
        stage1_threshold: residual cutoff for stage 1.
        stage2_threshold: residual cutoff for stage 2.

    Returns:
        List of confirmed hit dicts with keys:
          target_const, residual, digits, stage, transform
    """
    if q_exact == 0:
        return []

    # Stage 1: moderate precision
    mpmath.mp.dps = stage1_dps + 10
    est = mpmath.mpf(p_exact) / mpmath.mpf(q_exact)

    bank_s1 = precompute_constants(stage1_dps)
    near_misses = []

    # Test transforms: direct, neg, recip, neg_recip
    transforms = [("direct", est)]
    transforms.append(("neg", -est))
    if abs(est) > mpmath.mpf(10) ** (-(stage1_dps - 5)):
        transforms.append(("recip", 1 / est))
        transforms.append(("neg_recip", -1 / est))

    for tfm_name, tfm_val in transforms:
        for cname, cstr in bank_s1.items():
            cval = mpmath.mpf(cstr)
            residual = abs(tfm_val - cval)
            if residual < stage1_threshold:
                near_misses.append({
                    "target_const": cname,
                    "transform": tfm_name,
                    "s1_residual": float(residual),
                })

    if not near_misses:
        return []

    # Stage 2: high precision confirmation
    mpmath.mp.dps = stage2_dps + 10
    est_hp = mpmath.mpf(p_exact) / mpmath.mpf(q_exact)
    bank_s2 = precompute_constants(stage2_dps)

    confirmed = []
    for nm in near_misses:
        cname = nm["target_const"]
        tfm_name = nm["transform"]

        if tfm_name == "direct":
            val = est_hp
        elif tfm_name == "neg":
            val = -est_hp
        elif tfm_name == "recip":
            val = 1 / est_hp if est_hp != 0 else mpmath.mpf(0)
        elif tfm_name == "neg_recip":
            val = -1 / est_hp if est_hp != 0 else mpmath.mpf(0)
        else:
            continue

        cval = mpmath.mpf(bank_s2[cname])
        residual = abs(val - cval)

        if residual < stage2_threshold:
            if residual == 0:
                digits = stage2_dps
            else:
                digits = max(0, int(-mpmath.log10(residual)))
            confirmed.append({
                "target_const": cname,
                "transform": tfm_name,
                "residual": str(float(residual)),
                "digits": digits,
                "stage": 2,
                "stage1_residual": nm["s1_residual"],
            })

    return confirmed


def match_float_estimate(
    estimate: float,
    threshold: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Quick float64 matching (for Level-2 float shadow fallback).

    Uses the same constant bank as Level-1 but at float64.
    """
    if not math.isfinite(estimate) or estimate == 0.0:
        return []

    bank = precompute_constants(20)
    hits = []
    transforms = [("direct", estimate), ("neg", -estimate)]
    if abs(estimate) > 1e-15:
        transforms.append(("recip", 1.0 / estimate))
        transforms.append(("neg_recip", -1.0 / estimate))

    for tfm_name, tfm_val in transforms:
        for cname, cstr in bank.items():
            cval = float(cstr)
            residual = abs(tfm_val - cval)
            if residual < threshold:
                score = 300.0 if residual == 0 else -math.log10(residual)
                hits.append({
                    "target_const": cname,
                    "transform": tfm_name,
                    "residual": f"{residual:.6e}",
                    "score": round(score, 2),
                    "stage": 1,
                })
    return hits

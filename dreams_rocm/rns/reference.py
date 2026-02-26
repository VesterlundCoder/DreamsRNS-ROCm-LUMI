"""
Pure-Python RNS reference implementations.

These serve as ground truth for correctness testing.
Production walks use the native RNS-ROCm GPU library.

All operations are exact (integer arithmetic, no floating-point).
"""

from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
# Prime generation
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Miller-Rabin would be faster for large n, but trial division is fine
    for 31-bit candidates (sqrt ~ 46340 iterations max)."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def generate_primes(K: int, near_top: bool = True) -> List[int]:
    """Generate K distinct 31-bit primes.

    Args:
        K: Number of primes to generate.
        near_top: If True, pick primes near 2^31-1 (maximises per-prime
                  capacity).  If False, use a deterministic PRNG-based
                  selection matching the C++ library.

    Returns:
        Sorted list of K primes, each in [2^30, 2^31).
    """
    PRIME_MAX = (1 << 31) - 1
    PRIME_MIN = 1 << 30

    if near_top:
        primes: List[int] = []
        candidate = PRIME_MAX
        while len(primes) < K and candidate >= PRIME_MIN:
            if is_prime(candidate):
                primes.append(candidate)
            candidate -= 2  # only odd candidates
        if len(primes) < K:
            raise RuntimeError(
                f"Could not find {K} primes in [{PRIME_MIN}, {PRIME_MAX}]"
            )
        return sorted(primes)
    else:
        # Deterministic PRNG matching RNS-ROCm C++ py_generate_primes
        primes: List[int] = []
        state = 12345
        while len(primes) < K:
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            candidate = (state >> 32) & 0xFFFFFFFF
            candidate = (candidate | (1 << 30)) | 1
            if candidate < (1 << 30):
                candidate += (1 << 30)
            if is_prime(candidate) and candidate not in primes:
                primes.append(candidate)
        return sorted(primes)


def compute_barrett_mu(p: int) -> int:
    """Compute Barrett constant mu = floor(2^64 / p)."""
    return (1 << 64) // p


# ---------------------------------------------------------------------------
# Scalar modular arithmetic
# ---------------------------------------------------------------------------

def add_mod(a: int, b: int, p: int) -> int:
    """(a + b) mod p.  Assumes 0 <= a, b < p."""
    s = a + b
    return s - p if s >= p else s


def sub_mod(a: int, b: int, p: int) -> int:
    """(a - b) mod p.  Assumes 0 <= a, b < p."""
    return (a - b) % p


def mul_mod(a: int, b: int, p: int) -> int:
    """(a * b) mod p."""
    return (a * b) % p


def fma_mod(a: int, b: int, c: int, p: int) -> int:
    """(a * b + c) mod p."""
    return (a * b + c) % p


def neg_mod(a: int, p: int) -> int:
    """(-a) mod p."""
    return 0 if a == 0 else p - a


def pow_mod(base: int, exp: int, p: int) -> int:
    """base^exp mod p via Python built-in three-arg pow."""
    return pow(base, exp, p)


def inv_mod(a: int, p: int) -> int:
    """Modular inverse a^{-1} mod p using Fermat's little theorem.
    Requires p prime.  Returns 0 if a == 0."""
    if a == 0:
        return 0
    return pow(a, p - 2, p)


# ---------------------------------------------------------------------------
# RNS encode / decode helpers
# ---------------------------------------------------------------------------

def rns_encode(x: int, primes: List[int]) -> List[int]:
    """Encode a non-negative integer x into RNS residues."""
    return [x % p for p in primes]


def rns_encode_signed(x: int, primes: List[int]) -> List[int]:
    """Encode a signed integer.  For negative x, encode M + x where
    M = prod(primes)."""
    if x >= 0:
        return rns_encode(x, primes)
    M = 1
    for p in primes:
        M *= p
    return rns_encode(M + x, primes)


def crt_reconstruct(residues: List[int], primes: List[int]) -> int:
    """Reconstruct integer from RNS residues using iterative CRT
    (Garner-style).

    Returns x in [0, M) where M = prod(primes).
    """
    x = int(residues[0])
    M = int(primes[0])

    for i in range(1, len(residues)):
        p_i = int(primes[i])
        a_i = int(residues[i])

        x_mod_pi = x % p_i
        M_mod_pi = M % p_i
        M_inv = pow(M_mod_pi, p_i - 2, p_i)

        diff = (a_i - x_mod_pi) % p_i
        t = (diff * M_inv) % p_i

        x = x + M * t
        M = M * p_i

    return x


def crt_reconstruct_signed(residues: List[int], primes: List[int]) -> int:
    """Reconstruct signed integer, assuming result in [-M/2, M/2)."""
    x = crt_reconstruct(residues, primes)
    M = 1
    for p in primes:
        M *= p
    if x >= M // 2:
        x -= M
    return x


def rns_add(a_res: List[int], b_res: List[int], primes: List[int]) -> List[int]:
    """Element-wise (a + b) mod p for each prime."""
    return [add_mod(a, b, p) for a, b, p in zip(a_res, b_res, primes)]


def rns_mul(a_res: List[int], b_res: List[int], primes: List[int]) -> List[int]:
    """Element-wise (a * b) mod p for each prime."""
    return [mul_mod(a, b, p) for a, b, p in zip(a_res, b_res, primes)]


def rns_sub(a_res: List[int], b_res: List[int], primes: List[int]) -> List[int]:
    """Element-wise (a - b) mod p for each prime."""
    return [sub_mod(a, b, p) for a, b, p in zip(a_res, b_res, primes)]

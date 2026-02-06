"""
RNS (Residue Number System) module for Dreams pipeline.

Provides Python-level RNS operations with:
1. Pure Python reference implementations (always available)
2. Optional ctypes bindings to the compiled RNS-ROCm C++ library
"""

from .reference import (
    generate_primes, compute_barrett_mu,
    add_mod, sub_mod, mul_mod, inv_mod, pow_mod,
    neg_mod, fma_mod,
    rns_encode, rns_encode_signed,
)
from .bindings import (
    RnsContext, HAS_NATIVE_RNS, get_rns_library_path,
)

__all__ = [
    "generate_primes", "compute_barrett_mu",
    "add_mod", "sub_mod", "mul_mod", "inv_mod", "pow_mod",
    "neg_mod", "fma_mod",
    "rns_encode", "rns_encode_signed",
    "RnsContext", "HAS_NATIVE_RNS", "get_rns_library_path",
]

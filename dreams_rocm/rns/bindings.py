"""
ctypes-based Python bindings to the compiled RNS-ROCm shared library.

On LUMI the library is built with CMake / hipcc and produces
``librns_rocm.so``.  This module loads it at import time if it can
be found; otherwise falls back to pure-Python implementations in
``reference.py``.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_LIB_NAMES = ["librns_rocm.so", "librns_rocm_lib.so"]
_SEARCH_DIRS = [
    # In-tree build (rns_rocm_lib/build/)
    Path(__file__).resolve().parent.parent.parent / "rns_rocm_lib" / "build",
    Path(__file__).resolve().parent.parent.parent / "rns_rocm_lib" / "build" / "lib",
    # Alternative paths
    Path(__file__).resolve().parent.parent.parent / "lib",
    Path(__file__).resolve().parent.parent.parent / "build",
    Path(__file__).resolve().parent.parent.parent / "build" / "lib",
]

# Also honour an environment variable
_env_path = os.environ.get("RNS_ROCM_LIB")
if _env_path:
    _SEARCH_DIRS.insert(0, Path(_env_path))


def get_rns_library_path() -> Optional[Path]:
    """Return the path to the compiled RNS-ROCm shared library, or None."""
    for d in _SEARCH_DIRS:
        for name in _LIB_NAMES:
            p = d / name
            if p.is_file():
                return p
    return None


# ---------------------------------------------------------------------------
# Load the library (best-effort)
# ---------------------------------------------------------------------------

HAS_NATIVE_RNS = False
_lib = None

_lib_path = get_rns_library_path()
if _lib_path is not None:
    try:
        _lib = ctypes.CDLL(str(_lib_path))
        HAS_NATIVE_RNS = True
    except OSError as exc:
        import warnings
        warnings.warn(
            f"Found RNS-ROCm library at {_lib_path} but failed to load: {exc}. "
            "Using pure-Python fallback.",
            RuntimeWarning,
        )


# ---------------------------------------------------------------------------
# Thin wrapper around the loaded library
# ---------------------------------------------------------------------------

class RnsContext:
    """High-level wrapper for the native RNS-ROCm device context.

    If the native library is unavailable, methods fall back to the
    pure-Python reference implementations automatically.
    """

    def __init__(self, primes: np.ndarray):
        """
        Args:
            primes: 1-D uint32 array of K primes.
        """
        self.primes = np.asarray(primes, dtype=np.uint32).copy()
        self.K = len(self.primes)
        self._native_ctx = None

        if HAS_NATIVE_RNS and _lib is not None:
            try:
                self._init_native()
            except Exception:
                self._native_ctx = None

    # -- native helpers (only if library loaded) ----------------------------

    def _init_native(self):
        """Placeholder for full ctypes context initialisation.

        A complete binding would call rns::create_context via the C ABI
        exported from the shared library.  For Test 1 we rely on the
        pure-Python path and will enable this once the shared library is
        compiled on LUMI.
        """
        pass

    @property
    def has_gpu(self) -> bool:
        return self._native_ctx is not None

    # -- convenience methods (delegate to reference.py) ---------------------

    def encode(self, x: int) -> List[int]:
        from .reference import rns_encode
        return rns_encode(x, self.primes.tolist())

    def encode_signed(self, x: int) -> List[int]:
        from .reference import rns_encode_signed
        return rns_encode_signed(x, self.primes.tolist())

    def decode(self, residues: List[int]) -> int:
        from .reference import crt_reconstruct
        return crt_reconstruct(residues, self.primes.tolist())

    def decode_signed(self, residues: List[int]) -> int:
        from .reference import crt_reconstruct_signed
        return crt_reconstruct_signed(residues, self.primes.tolist())

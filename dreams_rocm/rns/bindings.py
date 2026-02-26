"""
ctypes-based Python bindings to the compiled RNS-ROCm shared library.

On LUMI the library is built with CMake / hipcc and produces
``librns_rocm.so``.  GPU is mandatory — no CPU fallback.
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
        # On GPU nodes this is a hard error; on dev machines it's expected
        import warnings
        warnings.warn(
            f"Found RNS-ROCm library at {_lib_path} but failed to load: {exc}. "
            "GPU execution will not be available.",
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

    # -- native helpers ----------------------------

    def _init_native(self):
        """Initialize the native GPU device context via ctypes.

        Calls rns::create_context with the prime array to set up
        GPU-side modulus data.
        """
        if _lib is None:
            return

        # Check for create_context C ABI wrapper
        if hasattr(_lib, 'rns_create_context'):
            primes_arr = self.primes.copy()
            ctx_ptr = _lib.rns_create_context(
                primes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_int(self.K),
            )
            self._native_ctx = ctx_ptr
        elif hasattr(_lib, 'create_context'):
            primes_arr = self.primes.copy()
            ctx_ptr = _lib.create_context(
                primes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_int(self.K),
            )
            self._native_ctx = ctx_ptr

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

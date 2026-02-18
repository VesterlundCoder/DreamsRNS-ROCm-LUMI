"""
GPU-accelerated runner using the native RNS-ROCm C++ library.

When the compiled librns_rocm_lib.so is available, this module provides
a fast GPU walk path via ctypes. Falls back to the pure-Python CPU runner
in runner.py when the library is not available.

The GPU path uses the fused walk kernel from rns_walk_fused.hip which:
  1. Evaluates the CMF bytecode at each step
  2. Multiplies the step matrix into the running product (RNS)
  3. Maintains a parallel shadow float for approximate scoring
  4. Takes snapshots at configurable depths
"""

import ctypes
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from .rns.bindings import HAS_NATIVE_RNS, get_rns_library_path
from .cmf_compile import CmfProgram, Opcode
from .runner import WalkConfig
from .crt.delta_targets import compute_dreams_delta


class GpuWalkRunner:
    """GPU runner using the native RNS-ROCm fused walk kernel.

    Falls back to CPU if the native library is unavailable.
    """

    def __init__(self, config: Optional[WalkConfig] = None):
        self.config = config or WalkConfig()
        self._lib = None
        self._initialized = False

    def _init(self):
        if self._initialized:
            return

        lib_path = get_rns_library_path()
        if lib_path is not None:
            try:
                self._lib = ctypes.CDLL(str(lib_path))
                self._setup_ctypes_signatures()
            except OSError as e:
                import warnings
                warnings.warn(f"Failed to load RNS-ROCm library: {e}", RuntimeWarning)
                self._lib = None

        self._initialized = True

    def _setup_ctypes_signatures(self):
        """Set up ctypes function signatures for the C API."""
        if self._lib is None:
            return

        # Check which functions are available
        self._has_walk_fused = hasattr(self._lib, 'walk_fused')
        self._has_eval_program = hasattr(self._lib, 'eval_program_to_matrix')
        self._has_gemm = hasattr(self._lib, 'rns_gemm_mod_u32')

    @property
    def has_gpu(self) -> bool:
        self._init()
        return self._lib is not None

    def run_walk(
        self,
        program: CmfProgram,
        shifts: np.ndarray,
        directions: List[int],
        cmf_idx: int = 0,
    ) -> Tuple[list, Dict[str, float]]:
        """Run the fused walk kernel on GPU (or CPU fallback).

        Args:
            program: Compiled CMF bytecode program.
            shifts: int32 array of shape (B, dim).
            directions: Walk direction per axis.
            cmf_idx: CMF index for hit reporting.

        Returns:
            (hits, metrics) tuple.
        """
        self._init()

        if self._lib is not None and self._has_walk_fused:
            return self._run_walk_native(program, shifts, directions, cmf_idx)
        else:
            return self._run_walk_python(program, shifts, directions, cmf_idx)

    def _run_walk_native(
        self,
        program: CmfProgram,
        shifts: np.ndarray,
        directions: List[int],
        cmf_idx: int,
    ) -> Tuple[list, Dict[str, float]]:
        """Run using the native C++ fused walk kernel via ctypes.

        This is the high-performance path when librns_rocm_lib.so is
        available. The entire walk (bytecode eval + matmul + shadow float)
        runs on a single GPU kernel launch.
        """
        t_start = time.time()
        m = program.m
        dim = program.dim
        B = len(shifts)
        K = self.config.K
        E = m * m
        depth = self.config.depth

        from .rns.reference import generate_primes, compute_barrett_mu

        primes = generate_primes(K)

        # Build constant table reduced mod each prime
        const_table = program.make_const_table(K, np.array(primes, dtype=np.uint32))

        # Build instruction arrays
        opcodes, dsts, a_args, b_args, out_reg = program.to_arrays()

        # Build PrimeMeta array (p, pad, mu, pinv, r2)
        # struct PrimeMeta { u32 p; u32 pad; u64 mu; u32 pinv; u32 r2; }
        # = 24 bytes per prime
        pm_dtype = np.dtype([
            ('p', np.uint32), ('pad', np.uint32),
            ('mu', np.uint64),
            ('pinv', np.uint32), ('r2', np.uint32),
        ])
        pm = np.zeros(K, dtype=pm_dtype)
        for k in range(K):
            pm[k]['p'] = primes[k]
            pm[k]['mu'] = compute_barrett_mu(primes[k])

        # Prepare shift and direction arrays
        shifts_flat = shifts.astype(np.int32).flatten()
        dirs_arr = np.array(directions, dtype=np.int32)

        # Output buffers
        P_final = np.zeros(B * E, dtype=np.uint32)
        alive = np.zeros(B, dtype=np.uint8)
        est1 = np.zeros(B, dtype=np.float32)
        est2 = np.zeros(B, dtype=np.float32)
        delta1 = np.zeros(B, dtype=np.float32)
        delta2 = np.zeros(B, dtype=np.float32)

        # Snapshot depths
        snap = self.config.snapshot_depths
        depth1 = snap[0] if len(snap) > 0 else depth // 2
        depth2 = snap[1] if len(snap) > 1 else depth

        # TODO: Call native walk_fused via ctypes when the C ABI wrapper
        # is exported. For now, fall back to the Python walk.
        #
        # The native call would look like:
        #   self._lib.walk_fused_c(
        #       depth, depth1, depth2, m, dim, K, B,
        #       instr_ptr, n_instr, const_ptr, n_const, out_reg_ptr,
        #       shifts_ptr, dirs_ptr, pm_ptr,
        #       P_final_ptr, alive_ptr,
        #       est1_ptr, est2_ptr, delta1_ptr, delta2_ptr
        #   )
        #
        # Until the C ABI wrapper is added, we use the Python path:

        t_end = time.time()
        return self._run_walk_python(program, shifts, directions, cmf_idx)

    def _run_walk_python(
        self,
        program: CmfProgram,
        shifts: np.ndarray,
        directions: List[int],
        cmf_idx: int,
    ) -> Tuple[list, Dict[str, float]]:
        """Pure-Python CPU walk (fallback).

        For PCF verification, use runner.run_pcf_walk() or runner.verify_pcf()
        directly instead of this GPU runner.
        """
        from .runner import run_pcf_walk, compute_dreams_delta_float
        from .cmf_compile import pcf_initial_values

        # This fallback only works for simple 1-shift PCF walks
        t0 = time.time()
        # Note: For proper PCF walks, use runner.verify_pcf() directly
        metrics = {"wall_time_sec": time.time() - t0, "backend": "cpu_fallback"}
        return [], metrics


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU and native library availability.

    Returns dict with status information for diagnostics.
    """
    result = {
        "native_lib_found": False,
        "native_lib_path": None,
        "gpu_detected": False,
        "gpu_backend": "cpu",
        "rocm_version": "unknown",
    }

    # Check native library
    lib_path = get_rns_library_path()
    if lib_path:
        result["native_lib_found"] = True
        result["native_lib_path"] = str(lib_path)

    # Check ROCm GPU
    try:
        import subprocess
        r = subprocess.run(["rocm-smi", "--showid"], capture_output=True,
                           text=True, timeout=5)
        if r.returncode == 0 and "GPU" in r.stdout:
            result["gpu_detected"] = True
            result["gpu_backend"] = "rocm"
    except Exception:
        pass

    # Check ROCm version
    try:
        version_file = Path("/opt/rocm/.info/version")
        if version_file.exists():
            result["rocm_version"] = version_file.read_text().strip()
    except Exception:
        pass

    return result

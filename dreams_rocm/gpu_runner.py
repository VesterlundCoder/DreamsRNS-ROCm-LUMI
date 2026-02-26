"""
GPU-accelerated runner using the native RNS-ROCm C++ library.

Requires the compiled librns_rocm_lib.so (GPU-only, no CPU fallback).

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

    Requires the native library — raises RuntimeError if unavailable.
    """

    def __init__(self, config: Optional[WalkConfig] = None):
        self.config = config or WalkConfig()
        self._lib = None
        self._initialized = False

    def _init(self):
        if self._initialized:
            return

        lib_path = get_rns_library_path()
        if lib_path is None:
            raise RuntimeError(
                "RNS-ROCm native library not found. "
                "Build librns_rocm_lib.so with hipcc and set RNS_ROCM_LIB env var, "
                "or place it in rns_rocm_lib/build/."
            )
        try:
            self._lib = ctypes.CDLL(str(lib_path))
            self._setup_ctypes_signatures()
        except OSError as e:
            raise RuntimeError(
                f"Failed to load RNS-ROCm library at {lib_path}: {e}"
            ) from e

        self._initialized = True

    def _setup_ctypes_signatures(self):
        """Set up ctypes function signatures for the C API."""
        self._has_walk_fused = hasattr(self._lib, 'walk_fused')
        self._has_eval_program = hasattr(self._lib, 'eval_program_to_matrix')
        self._has_gemm = hasattr(self._lib, 'rns_gemm_mod_u32')

        if not self._has_walk_fused:
            raise RuntimeError(
                "RNS-ROCm library loaded but walk_fused symbol not found. "
                "Ensure the library was built with GPU support (hipcc)."
            )

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
        """Run the fused walk kernel on GPU.

        Args:
            program: Compiled CMF bytecode program.
            shifts: int32 array of shape (B, dim).
            directions: Walk direction per axis.
            cmf_idx: CMF index for hit reporting.

        Returns:
            (hits, metrics) tuple.
        """
        self._init()
        return self._run_walk_native(program, shifts, directions, cmf_idx)

    def _run_walk_native(
        self,
        program: CmfProgram,
        shifts: np.ndarray,
        directions: List[int],
        cmf_idx: int,
    ) -> Tuple[list, Dict[str, float]]:
        """Run using the native C++ fused walk kernel via ctypes.

        The entire walk (bytecode eval + matmul + shadow float)
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
        pm_dtype = np.dtype([
            ('p', np.uint32), ('pad', np.uint32),
            ('mu', np.uint64),
            ('pinv', np.uint32), ('r2', np.uint32),
        ])
        pm = np.zeros(K, dtype=pm_dtype)
        for k in range(K):
            pm[k]['p'] = primes[k]
            pm[k]['mu'] = compute_barrett_mu(primes[k])

        # Pack instructions into struct array matching C Instr layout
        instr_dtype = np.dtype([
            ('op', np.uint8), ('dst', np.uint8),
            ('a', np.uint8), ('b', np.uint8),
        ])
        n_instr = len(program.instructions)
        instr_arr = np.zeros(n_instr, dtype=instr_dtype)
        for i, ins in enumerate(program.instructions):
            instr_arr[i] = (ins.op, ins.dst, ins.a, ins.b)

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
        depth1 = depth // 2
        depth2 = depth

        # Call the native GPU walk_fused kernel via ctypes
        self._lib.walk_fused_c(
            ctypes.c_int(depth),
            ctypes.c_int(depth1),
            ctypes.c_int(depth2),
            ctypes.c_int(m),
            ctypes.c_int(dim),
            ctypes.c_int(K),
            ctypes.c_int(B),
            instr_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(n_instr),
            const_table.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(program.constants)),
            out_reg.ctypes.data_as(ctypes.c_void_p),
            shifts_flat.ctypes.data_as(ctypes.c_void_p),
            dirs_arr.ctypes.data_as(ctypes.c_void_p),
            pm.ctypes.data_as(ctypes.c_void_p),
            P_final.ctypes.data_as(ctypes.c_void_p),
            alive.ctypes.data_as(ctypes.c_void_p),
            est1.ctypes.data_as(ctypes.c_void_p),
            est2.ctypes.data_as(ctypes.c_void_p),
            delta1.ctypes.data_as(ctypes.c_void_p),
            delta2.ctypes.data_as(ctypes.c_void_p),
        )

        t_end = time.time()

        # Build hits from results
        hits = []
        for bi in range(B):
            if alive[bi]:
                hits.append({
                    'cmf_idx': cmf_idx,
                    'shift_idx': bi,
                    'delta2': float(delta2[bi]),
                    'est2': float(est2[bi]),
                    'alive': True,
                })

        metrics = {
            "wall_time_sec": t_end - t_start,
            "backend": "gpu_native",
            "B": B,
            "depth": depth,
            "K": K,
        }
        return hits, metrics


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU and native library availability.

    Returns dict with status information for diagnostics.
    """
    result = {
        "native_lib_found": False,
        "native_lib_path": None,
        "gpu_detected": False,
        "gpu_backend": "none",
        "rocm_version": "unknown",
    }

    # Check native library
    lib_path = get_rns_library_path()
    if lib_path:
        result["native_lib_found"] = True
        result["native_lib_path"] = str(lib_path)

    # Check GPU (NVIDIA first, then ROCm)
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            result["gpu_detected"] = True
            result["gpu_backend"] = "cuda"
    except Exception:
        pass

    if not result["gpu_detected"]:
        try:
            import subprocess
            r = subprocess.run(["rocm-smi", "--showid"], capture_output=True,
                               text=True, timeout=5)
            if r.returncode == 0 and "GPU" in r.stdout:
                result["gpu_detected"] = True
                result["gpu_backend"] = "rocm"
        except Exception:
            pass

    # Check ROCm version (AMD only)
    if result["gpu_backend"] == "rocm":
        try:
            version_file = Path("/opt/rocm/.info/version")
            if version_file.exists():
                result["rocm_version"] = version_file.read_text().strip()
        except Exception:
            pass

    return result

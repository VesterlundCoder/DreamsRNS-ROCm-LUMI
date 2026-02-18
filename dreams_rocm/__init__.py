"""
Dreams-RNS-ROCm-LUMI: GPU-accelerated Ramanujan Dreams pipeline
using RNS arithmetic on AMD MI250X GPUs via ROCm/HIP.

Correct PCF walk convention (matching ramanujantools):
  M(n) = [[0, b(n)], [1, a(n)]]          companion form
  P(N) = A · M(1) · M(2) · ... · M(N)    A = [[1, a(0)], [0, 1]]
  p = P[0, m-1],  q = P[1, m-1]          last column extraction
  delta = -(1 + log|p/q - L| / log|q|)

Designed for LUMI-G supercomputer (small-g partition, Singularity containers).
"""

__version__ = "0.2.0"

from .cmf_compile import (
    compile_cmf_from_dict, compile_cmf_from_sympy,
    compile_pcf_from_strings, pcf_initial_values,
    CmfProgram, Opcode, Instruction, CmfCompiler,
)
from .runner import (
    WalkConfig,
    generate_rns_primes,
    crt_reconstruct,
    centered,
    compute_dreams_delta_float,
    compute_dreams_delta_exact,
    run_pcf_walk,
    verify_pcf,
)
from .trajectories import generate_trajectories, normalize_trajectory
from .shifts import generate_shifts
from .logging import RunLogger, RunManifest
from .gpu_runner import GpuWalkRunner, check_gpu_availability

__all__ = [
    "compile_cmf_from_dict", "compile_cmf_from_sympy",
    "compile_pcf_from_strings", "pcf_initial_values",
    "CmfProgram", "Opcode", "Instruction", "CmfCompiler",
    "WalkConfig",
    "generate_rns_primes", "crt_reconstruct", "centered",
    "compute_dreams_delta_float", "compute_dreams_delta_exact",
    "run_pcf_walk", "verify_pcf",
    "GpuWalkRunner", "check_gpu_availability",
    "generate_trajectories", "normalize_trajectory",
    "generate_shifts",
    "RunLogger", "RunManifest",
]

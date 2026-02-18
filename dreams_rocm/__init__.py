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
from .cmf_walk import (
    compile_cmf_spec, run_cmf_walk, run_cmf_walk_vec, run_cmf_walk_batch,
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
from .exhaust import (
    exhaust_trajectories, exhaust_shifts, exhaust_summary,
    generate_primitive_trajectories, generate_unique_shifts,
    canonical_direction, RationalShift, EXHAUST_PRESETS,
    compute_sphere_coverage,
)
from .cmf_generator import (
    generate_pFq_specs, generate_2f2_specs, generate_3f2_specs,
    generate_3f3_specs, generate_4f3_specs, generate_5f4_specs,
    build_pFq_companion_matrix, write_specs_chunked,
    CMFSpec, FAMILY_CONFIG,
)
from .logging import RunLogger, RunManifest
from .gpu_runner import GpuWalkRunner, check_gpu_availability

__all__ = [
    "compile_cmf_from_dict", "compile_cmf_from_sympy",
    "compile_pcf_from_strings", "pcf_initial_values",
    "CmfProgram", "Opcode", "Instruction", "CmfCompiler",
    "compile_cmf_spec", "run_cmf_walk", "run_cmf_walk_vec", "run_cmf_walk_batch",
    "WalkConfig",
    "generate_rns_primes", "crt_reconstruct", "centered",
    "compute_dreams_delta_float", "compute_dreams_delta_exact",
    "run_pcf_walk", "verify_pcf",
    "GpuWalkRunner", "check_gpu_availability",
    "generate_trajectories", "normalize_trajectory",
    "generate_shifts",
    "exhaust_trajectories", "exhaust_shifts", "exhaust_summary",
    "generate_primitive_trajectories", "generate_unique_shifts",
    "canonical_direction", "RationalShift", "EXHAUST_PRESETS",
    "compute_sphere_coverage",
    "generate_pFq_specs", "generate_2f2_specs", "generate_3f2_specs",
    "generate_3f3_specs", "generate_4f3_specs", "generate_5f4_specs",
    "build_pFq_companion_matrix", "write_specs_chunked",
    "CMFSpec", "FAMILY_CONFIG",
    "RunLogger", "RunManifest",
]

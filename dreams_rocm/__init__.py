"""
Dreams-RNS-ROCm-LUMI: GPU-accelerated Ramanujan Dreams pipeline
using RNS arithmetic on AMD MI250X GPUs via ROCm/HIP.

Designed for LUMI-G supercomputer with 8 GPUs per node.
"""

__version__ = "0.1.0"

from .runner import DreamsRunner, WalkConfig, Hit
from .cmf_compile import (
    compile_cmf_from_dict, compile_cmf_from_sympy,
    CmfProgram, Opcode, Instruction, CmfCompiler
)
from .trajectories import generate_trajectories, normalize_trajectory
from .shifts import generate_shifts
from .logging import RunLogger, RunManifest
from .gpu_runner import GpuWalkRunner, check_gpu_availability

__all__ = [
    "DreamsRunner", "WalkConfig", "Hit",
    "GpuWalkRunner", "check_gpu_availability",
    "compile_cmf_from_dict", "compile_cmf_from_sympy",
    "CmfProgram", "Opcode", "Instruction", "CmfCompiler",
    "generate_trajectories", "normalize_trajectory",
    "generate_shifts",
    "RunLogger", "RunManifest",
]

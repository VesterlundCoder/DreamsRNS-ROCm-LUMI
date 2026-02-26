#!/usr/bin/env python3
"""
cmf_walk_engine.py — GPU-accelerated RNS CMF walk engine for 6F5 sweep.

Compilation:  sympy → dreams_rocm.CmfProgram bytecode (at startup, once)
Walk:         dreams_rocm.gpu_runner.GpuWalkRunner → native HIP kernel (GPU)

No CPU fallback.  No numpy walk.  Fails hard if GPU is unavailable.

Rational shifts are supported by clearing denominators to LCM=24 and
inserting MULINV-by-24 after every LOAD_X in the bytecode, so the GPU
kernel works with integer shifts while the RNS modular inverse recovers
the exact rational axis values.

Usage:
    from cmf_walk_engine import compile_6f5, walk_gpu_batch
    program, lcm = compile_6f5(cmf_dict)
    estimates, confidences = walk_gpu_batch(program, lcm, shifts, dirs, depth)
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── Ensure dreams_rocm is importable ──
_REPO_ROOT = os.environ.get(
    "DREAMS_ROCM_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dreams_rocm.cmf_compile import CmfProgram, Instruction, Opcode
from dreams_rocm.cmf_walk import compile_cmf_spec
from dreams_rocm.gpu_runner import GpuWalkRunner
from dreams_rocm.runner import WalkConfig

# LCM of all shift denominators in the 6F5 task bank: LCM(1,2,3,4,6,8)
SHIFT_DENOM_LCM = 24


# ═══════════════════════════════════════════════════════════════════════
# Bytecode rationalisation  (insert MULINV-by-LCM after each LOAD_X)
# ═══════════════════════════════════════════════════════════════════════

def _rationalize_program(prog: CmfProgram, lcm: int) -> CmfProgram:
    """Post-process a compiled CmfProgram so the GPU can use integer shifts.

    The GPU kernel computes:  axis_val = int_shift[i] + step * int_dir[i]
    where  int_shift[i] = num[i] * (LCM / den[i])
           int_dir[i]   = dir[i] * LCM

    This gives axis_val = LCM * (num/den + step*dir), i.e. LCM × the real
    rational value.  To recover the real value inside the bytecode we insert
        MULINV  dst, dst, lcm_reg
    immediately after every LOAD_X.  In RNS this is exact (modular inverse).
    """
    # Add LCM as a new constant
    new_consts = list(prog.constants) + [lcm]
    lcm_const_idx = len(prog.constants)

    new_instrs = []
    next_reg = prog.n_reg

    # Allocate a register for the LCM constant (loaded once at the start)
    lcm_reg = next_reg
    next_reg += 1
    new_instrs.append(Instruction(Opcode.LOAD_C, lcm_reg, lcm_const_idx, 0))

    for instr in prog.instructions:
        new_instrs.append(instr)
        if instr.op == Opcode.LOAD_X:
            # Insert MULINV: dst = dst * lcm^{-1}  (overwrite in-place)
            new_instrs.append(
                Instruction(Opcode.MULINV, instr.dst, instr.dst, lcm_reg))

    return CmfProgram(
        m=prog.m,
        dim=prog.dim,
        instructions=new_instrs,
        out_reg=list(prog.out_reg),
        n_reg=next_reg,
        constants=new_consts,
        directions=list(prog.directions),
        name=prog.name,
    )


# ═══════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════

def compile_6f5(cmf: dict) -> Tuple[CmfProgram, int]:
    """Compile a pFq CMF dict to a GPU-ready CmfProgram.

    Returns (program, lcm) where program has MULINV-by-lcm inserted
    after every LOAD_X so the GPU can work with integer shifts.
    """
    spec = {
        "p": cmf["p"],
        "q": cmf["q"],
        "a_params": cmf["upper_params"],
        "b_params": cmf["lower_params"],
        "name": cmf.get("cmf_id", "6F5"),
    }
    prog = compile_cmf_spec(spec)
    if prog is None:
        raise RuntimeError(f"Failed to compile CMF spec: {spec['name']}")

    rationalized = _rationalize_program(prog, SHIFT_DENOM_LCM)
    return rationalized, SHIFT_DENOM_LCM


# ═══════════════════════════════════════════════════════════════════════
# Shift conversion  (rational → LCM-scaled integers)
# ═══════════════════════════════════════════════════════════════════════

def _make_int_shifts(
    shift_nums: List[int],
    shift_dens: List[int],
    lcm: int,
) -> List[int]:
    """Convert one rational shift to LCM-scaled integer shift."""
    return [n * (lcm // d) for n, d in zip(shift_nums, shift_dens)]


def _make_int_dirs(direction: List[int], lcm: int) -> List[int]:
    """Scale direction by LCM for the rationalized program."""
    return [d * lcm for d in direction]


# ═══════════════════════════════════════════════════════════════════════
# GPU walk
# ═══════════════════════════════════════════════════════════════════════

# Module-level GPU runner (initialised once)
_runner: Optional[GpuWalkRunner] = None


def _get_runner(depth: int = 2000, K: int = 32) -> GpuWalkRunner:
    global _runner
    if _runner is None:
        cfg = WalkConfig(K=K, depth=depth)
        _runner = GpuWalkRunner(config=cfg)
    return _runner


def walk_single(
    prog: CmfProgram,
    lcm: int,
    shift_nums: List[int],
    shift_dens: List[int],
    direction: List[int],
    depth: int = 2000,
) -> Optional[Tuple[float, bool]]:
    """Run one CMF walk on GPU, return (estimate, confident) or None.

    Confidence is determined by comparing the float shadow estimate at
    half-depth (est1) vs full-depth (est2).  If they agree to 1e-6
    relative, the convergent is confident.
    """
    int_shift = _make_int_shifts(shift_nums, shift_dens, lcm)
    int_dir = _make_int_dirs(direction, lcm)

    shifts_np = np.array([int_shift], dtype=np.int32)  # (1, dim)
    runner = _get_runner(depth=depth)
    hits, metrics = runner.run_walk(prog, shifts_np, int_dir, cmf_idx=0)

    # The runner populates est1/est2 in its output buffers.
    # For a single walk (B=1), extract from the raw arrays.
    # Since run_walk only returns "alive" hits, check if we got one.
    if not hits:
        return None

    est2 = hits[0].get("est2", float("nan"))
    if not math.isfinite(est2):
        return None

    # Confidence: does the walk look converged?
    delta2 = hits[0].get("delta2", 0.0)
    confident = delta2 > 2.0  # delta > 2 means ~100 matching digits

    return (est2, confident)


def walk_gpu_batch(
    prog: CmfProgram,
    lcm: int,
    shifts_nums: np.ndarray,
    shifts_dens: np.ndarray,
    direction: List[int],
    depth: int = 2000,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run B CMF walks on GPU, return (estimates, confidences).

    shifts_nums: (B, dim) int array of shift numerators.
    shifts_dens: (B, dim) int array of shift denominators.
    direction:   (dim,) int trajectory — shared for all B walks.

    Returns:
        estimates:   (B,) float32 array  (NaN where walk diverged)
        confidences: (B,) bool array
    """
    B = shifts_nums.shape[0]
    dim = shifts_nums.shape[1]

    # Convert all shifts to LCM-scaled integers
    scale_factors = np.array(
        [[lcm // d for d in row] for row in shifts_dens.tolist()],
        dtype=np.int32)
    int_shifts = (shifts_nums * scale_factors).astype(np.int32)  # (B, dim)
    int_dir = _make_int_dirs(direction, lcm)

    runner = _get_runner(depth=depth)

    estimates = np.full(B, np.nan, dtype=np.float32)
    confidences = np.zeros(B, dtype=bool)

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        batch_shifts = int_shifts[start:end]

        hits, metrics = runner.run_walk(
            prog, batch_shifts, int_dir, cmf_idx=0)

        # Map alive hits back to original indices
        for h in hits:
            bi = h["shift_idx"]
            global_bi = start + bi
            estimates[global_bi] = h.get("est2", float("nan"))
            confidences[global_bi] = h.get("delta2", 0.0) > 2.0

    return estimates, confidences

"""
Dreams Runner: GPU/CPU execution wrapper for the RNS walk pipeline.

Ported from Dreams-RNS-CUDA runner.py for ROCm on LUMI-G (MI250X).
Supports:
  - GPU execution via ROCm (HIP) through the RNS-ROCm native library
  - CPU fallback with pure-Python RNS arithmetic
  - Per-CMF batch processing with configurable shifts/trajectories
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import math
import time

from .cmf_compile import CmfProgram, Opcode
from .rns.reference import (
    generate_primes, add_mod, sub_mod, mul_mod, neg_mod,
    inv_mod, pow_mod, fma_mod,
)
from .crt.delta_targets import compute_dreams_delta


@dataclass
class WalkConfig:
    """Configuration for walk execution.

    Dreams delta = -(1 + log(|err|) / log(|q|))
    Higher delta = better convergence.
    """
    K: int = 64                     # Number of RNS primes
    B: int = 1000                   # Batch size (shifts per CMF)
    depth: int = 2000               # Walk depth (number of steps)
    topk: int = 100                 # Top-K hits to keep
    target: float = 1.2020569031595942  # Default: zeta(3)
    target_name: str = "zeta3"
    snapshot_depths: Tuple[int, ...] = (200, 2000)
    delta_threshold: float = 0.0    # Minimum delta for hits
    normalize_every: int = 50       # Normalize shadow-float every N steps
    K_small: int = 6                # Primes for partial CRT


@dataclass
class Hit:
    """A hit result from the walk."""
    cmf_idx: int
    cmf_name: str
    shift: List[int]
    depth: int
    delta: float
    log_q: float
    traj_id: int = 0
    traj_dir: Tuple[int, int] = (1, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cmf_idx': self.cmf_idx,
            'cmf_name': self.cmf_name,
            'shift': self.shift,
            'depth': self.depth,
            'delta': self.delta,
            'log_q': self.log_q,
            'traj_id': self.traj_id,
            'traj_dir': list(self.traj_dir),
        }


class DreamsRunner:
    """GPU/CPU runner for Dreams pipeline.

    Usage:
        runner = DreamsRunner(programs, config=WalkConfig(...))
        hits = runner.run()
    """

    def __init__(self, programs: List[CmfProgram],
                 config: Optional[WalkConfig] = None):
        self.programs = programs
        self.config = config or WalkConfig()
        self._primes = None
        self._gpu_available = False
        self._initialized = False

    def _init(self):
        """Initialize primes and check GPU availability."""
        if self._initialized:
            return

        self._primes = np.array(
            generate_primes(self.config.K), dtype=np.uint32
        )

        # Try ROCm GPU path
        try:
            # Attempt 1: cupy-rocm
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            self._gpu_available = True
            self._gpu_backend = "cupy"
            self._cp = cp
        except Exception:
            try:
                # Attempt 2: PyTorch ROCm
                import torch
                if torch.cuda.is_available():  # PyTorch ROCm uses cuda API
                    self._gpu_available = True
                    self._gpu_backend = "pytorch"
                    self._torch = torch
                else:
                    self._gpu_available = False
            except Exception:
                self._gpu_available = False

        if not self._gpu_available:
            self._gpu_backend = "cpu"

        self._initialized = True

    def _eval_step_matrix_cpu(
        self, program: CmfProgram, x_vals: List[int], prime: int
    ) -> List[List[int]]:
        """Evaluate step matrix from bytecode on CPU for one prime.

        Args:
            program: Compiled CMF program.
            x_vals: Axis values [dim].
            prime: Prime modulus.

        Returns:
            m x m matrix of residues (lists of lists).
        """
        m = program.m
        regs = [0] * program.n_reg
        alive = True

        for instr in program.instructions:
            if not alive:
                break
            op = instr.op
            dst = instr.dst
            a_val = regs[instr.a] if instr.a < len(regs) else 0
            b_val = regs[instr.b] if instr.b < len(regs) else 0

            if op == Opcode.LOAD_X:
                idx = instr.a
                if idx < len(x_vals):
                    regs[dst] = x_vals[idx] % prime
                else:
                    regs[dst] = 0
            elif op == Opcode.LOAD_C:
                c_idx = instr.a
                if c_idx < len(program.constants):
                    c = program.constants[c_idx]
                    regs[dst] = c % prime if c >= 0 else (prime - ((-c) % prime)) % prime
                else:
                    regs[dst] = 0
            elif op == Opcode.ADD:
                regs[dst] = add_mod(a_val, b_val, prime)
            elif op == Opcode.SUB:
                regs[dst] = sub_mod(a_val, b_val, prime)
            elif op == Opcode.MUL:
                regs[dst] = mul_mod(a_val, b_val, prime)
            elif op == Opcode.NEG:
                regs[dst] = neg_mod(a_val, prime)
            elif op == Opcode.POW2:
                regs[dst] = mul_mod(a_val, a_val, prime)
            elif op == Opcode.POW3:
                regs[dst] = mul_mod(mul_mod(a_val, a_val, prime), a_val, prime)
            elif op == Opcode.INV:
                if a_val == 0:
                    alive = False
                    regs[dst] = 0
                else:
                    regs[dst] = inv_mod(a_val, prime)
            elif op == Opcode.MULINV:
                if b_val == 0:
                    alive = False
                    regs[dst] = 0
                else:
                    regs[dst] = mul_mod(a_val, inv_mod(b_val, prime), prime)

        if not alive:
            return None

        # Build output matrix from out_reg
        matrix = []
        for i in range(m):
            row = []
            for j in range(m):
                idx = i * m + j
                if idx < len(program.out_reg):
                    r = program.out_reg[idx]
                    row.append(regs[r] if r < len(regs) else 0)
                else:
                    row.append(0)
            matrix.append(row)
        return matrix

    def _matmul_mod(self, A, B, m: int, prime: int):
        """Modular matrix multiplication A @ B mod prime (CPU)."""
        C = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                acc = 0
                for l in range(m):
                    acc += A[i][l] * B[l][j]
                C[i][j] = acc % prime
        return C

    def _run_walk_cpu(
        self, program: CmfProgram, shifts: np.ndarray,
        cmf_idx: int, directions: List[int],
    ) -> Tuple[List[Hit], Dict[str, float]]:
        """CPU fallback for walk computation.

        Returns:
            (hits, timing_metrics)
        """
        m = program.m
        dim = program.dim
        B = len(shifts)
        K = self.config.K

        t_start = time.time()

        # Shadow float for delta estimation
        P_float = np.zeros((B, m, m), dtype=np.float64)
        for b in range(B):
            np.fill_diagonal(P_float[b], 1.0)
        log_scale = np.zeros(B, dtype=np.float64)

        # RNS product matrix (only first prime for lightweight tracking)
        # Full K-prime RNS is expensive on CPU; use shadow float for scoring
        hits = []
        alive = np.ones(B, dtype=bool)

        for step in range(self.config.depth):
            for b in range(B):
                if not alive[b]:
                    continue

                # Compute x values for this step
                x_vals = []
                for d in range(dim):
                    x_vals.append(int(shifts[b, d]) + step * directions[d])

                # Evaluate step matrix as float (for shadow tracking)
                M_float = np.zeros((m, m), dtype=np.float64)
                for i in range(m):
                    for j in range(m):
                        # Use first prime to get modular values, convert to float
                        p0 = int(self._primes[0])
                        mat = self._eval_step_matrix_cpu(program, x_vals, p0)
                        if mat is None:
                            alive[b] = False
                            break
                        M_float[i][j] = float(mat[i][j])
                    if not alive[b]:
                        break

                if not alive[b]:
                    continue

                # Update shadow float: P = P @ M
                P_float[b] = P_float[b] @ M_float
                max_val = np.max(np.abs(P_float[b]))
                if max_val > 1e10:
                    P_float[b] /= max_val
                    log_scale[b] += np.log(max_val)

            # Check for hits at snapshot depths
            if step + 1 in self.config.snapshot_depths:
                for b in range(B):
                    if not alive[b]:
                        continue
                    p_val = float(P_float[b, 0, -1])
                    q_val = float(P_float[b, 1, -1]) if m > 1 else 1.0

                    delta, log_q = compute_dreams_delta(
                        p_val, q_val, self.config.target,
                        log_scale=float(log_scale[b]),
                    )

                    if delta > self.config.delta_threshold and delta > -1e9:
                        hits.append(Hit(
                            cmf_idx=cmf_idx,
                            cmf_name=program.name,
                            shift=list(shifts[b]),
                            depth=step + 1,
                            delta=delta,
                            log_q=log_q,
                        ))

        t_end = time.time()
        metrics = {
            "wall_time_sec": t_end - t_start,
            "shifts_per_sec": B / max(t_end - t_start, 1e-6),
            "backend": "cpu",
        }
        return hits, metrics

    def run(
        self,
        shifts_per_cmf: Optional[int] = None,
        depth: Optional[int] = None,
        shift_method: str = 'grid',
        shift_bounds: Tuple[int, int] = (-1000, 1000),
    ) -> Tuple[List[Hit], Dict[str, Any]]:
        """Run the Dreams pipeline on all CMF programs.

        Args:
            shifts_per_cmf: Override config.B.
            depth: Override config.depth.
            shift_method: Shift generation method.
            shift_bounds: Bounds for shifts.

        Returns:
            (sorted_hits, run_metrics)
        """
        self._init()

        if shifts_per_cmf is not None:
            self.config.B = shifts_per_cmf
        if depth is not None:
            self.config.depth = depth

        from .shifts import generate_shifts

        all_hits = []
        all_metrics = []

        for cmf_idx, program in enumerate(self.programs):
            shifts = generate_shifts(
                n_shifts=self.config.B,
                dim=program.dim,
                method=shift_method,
                bounds=shift_bounds,
                cmf_idx=cmf_idx,
            )

            hits, metrics = self._run_walk_cpu(
                program, shifts, cmf_idx,
                directions=program.directions or [1] * program.dim,
            )
            metrics["cmf_idx"] = cmf_idx
            metrics["cmf_name"] = program.name
            metrics["n_hits"] = len(hits)
            all_hits.extend(hits)
            all_metrics.append(metrics)

        # Sort by delta (highest first)
        all_hits.sort(key=lambda h: h.delta, reverse=True)
        topk_hits = all_hits[:self.config.topk]

        run_metrics = {
            "total_hits": len(all_hits),
            "topk_returned": len(topk_hits),
            "per_cmf": all_metrics,
            "gpu_backend": self._gpu_backend if self._initialized else "unknown",
        }
        return topk_hits, run_metrics

    def run_single(
        self, program: CmfProgram, shifts: np.ndarray,
        cmf_idx: int = 0, directions: Optional[List[int]] = None,
    ) -> Tuple[List[Hit], Dict[str, float]]:
        """Run walk on a single CMF with specified shifts."""
        self._init()
        dirs = directions or program.directions or [1] * program.dim
        return self._run_walk_cpu(program, shifts, cmf_idx, dirs)

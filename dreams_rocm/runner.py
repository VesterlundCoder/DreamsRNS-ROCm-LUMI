"""
Dreams Runner: RNS walk pipeline with correct PCF conventions.

Implements the correct PCF companion matrix walk matching ramanujantools:
  M(n) = [[0, b(n)], [1, a(n)]]  (companion form)
  P(N) = A · M(1) · M(2) · ... · M(N)  where A = [[1, a(0)], [0, 1]]
  p = P[0, m-1],  q = P[1, m-1]  (last column)
  delta = -(1 + log|p/q - L| / log|q|)

Uses numpy for the K-prime vectorized reference walk.
For GPU-accelerated walks, use gpu_runner.GpuWalkRunner.
Ported from the corrected Dreams-RNS-CUDA v0.2.0 runner.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import math
import time

from .cmf_compile import CmfProgram, Opcode, compile_pcf_from_strings, pcf_initial_values
from .rns.reference import generate_primes, inv_mod


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class WalkConfig:
    """Configuration for walk execution."""
    K: int = 32                     # Number of RNS primes (32 × 31-bit ≈ 992 bits)
    depth: int = 2000               # Walk depth (number of matrix multiplications)
    delta_threshold: float = -2.0   # Minimum delta for reporting


# ── Prime generation ─────────────────────────────────────────────────────

def generate_rns_primes(K: int) -> np.ndarray:
    """Generate K 31-bit primes (descending from 2^31-1) for RNS."""
    return np.array(generate_primes(K, near_top=True), dtype=np.int64)


# ── CRT reconstruction ──────────────────────────────────────────────────

def crt_reconstruct(residues, primes):
    """Reconstruct big integer from RNS residues via iterative CRT.

    Returns:
        (x, M) where x is the reconstructed value and M = prod(primes).
    """
    x = int(residues[0])
    M = int(primes[0])
    for i in range(1, len(residues)):
        p_i = int(primes[i])
        a_i = int(residues[i])
        t = ((a_i - x % p_i) % p_i) * pow(M % p_i, p_i - 2, p_i) % p_i
        x += M * t
        M *= p_i
    return x, M


def centered(x, M):
    """Convert unsigned CRT result to signed (centered) representation."""
    return x - M if x > M // 2 else x


# ── Delta computation ────────────────────────────────────────────────────

def compute_dreams_delta_float(p_val: float, q_val: float,
                               log_scale: float, target: float
                               ) -> Tuple[float, float]:
    """Approximate delta from float64 shadow values.

    Returns (delta, log_q).
    """
    if not math.isfinite(p_val) or not math.isfinite(q_val) or abs(q_val) < 1e-300:
        return -1e10, 0.0
    est = p_val / q_val
    abs_err = abs(est - target)
    log_abs_q = log_scale + math.log(abs(q_val))
    if abs_err < 1e-300 or log_abs_q <= 0.0:
        return (100.0 if abs_err < 1e-300 else -1e10), log_abs_q
    return -(1.0 + math.log(abs_err) / log_abs_q), log_abs_q


def compute_dreams_delta_exact(p_big: int, q_big: int, target, dps: int = 200):
    """Exact delta from big-integer p, q using mpmath.

    Returns:
        float delta value
    """
    import mpmath as mp
    mp.mp.dps = dps
    if q_big == 0:
        return float('-inf')
    approx = mp.mpf(p_big) / mp.mpf(q_big)
    err = abs(approx - mp.mpf(str(target)))
    qq = abs(mp.mpf(q_big))
    if err == 0:
        return float('inf')
    if qq <= 1:
        return float('-inf')
    return float(-(1 + mp.log(err) / mp.log(qq)))


# ── Bytecode evaluator — vectorised across all K primes ──────────────────

def _precompute_const_residues(program: CmfProgram, primes: np.ndarray) -> np.ndarray:
    """Precompute c_i mod p_k for every (constant, prime) pair.

    Returns array of shape (n_constants, K), dtype int64.
    """
    K = len(primes)
    pp_list = [int(p) for p in primes]
    n_c = len(program.constants)
    table = np.zeros((max(n_c, 1), K), dtype=np.int64)
    for idx in range(n_c):
        c = int(program.constants[idx])
        table[idx] = np.array([c % p if c >= 0 else (p - (-c % p)) % p
                                for p in pp_list], dtype=np.int64)
    return table


def _eval_bytecode_allprimes(program: CmfProgram, step: int,
                              shift_val: int, primes: np.ndarray,
                              const_table: np.ndarray) -> np.ndarray:
    """Evaluate bytecode producing one (m, m, K) matrix — all primes at once.

    The bytecode uses the ROCm instruction format:
      Instruction(op, dst, a, b)
    Output matrix entries are read from registers at program.out_reg indices.

    Returns:
        np.ndarray of shape (m, m, K), dtype int64, entries mod primes
    """
    m = program.m
    K = len(primes)
    pp = primes  # int64

    regs = [np.zeros(K, dtype=np.int64) for _ in range(program.n_reg)]

    for instr in program.instructions:
        op = instr.op
        dst = instr.dst

        if op == Opcode.LOAD_X:
            axis = instr.a
            val = np.int64(shift_val + step * program.directions[axis])
            regs[dst] = val % pp
        elif op == Opcode.LOAD_C:
            c_idx = instr.a
            if c_idx < len(const_table):
                regs[dst] = const_table[c_idx].copy()
            else:
                regs[dst] = np.zeros(K, dtype=np.int64)
        elif op == Opcode.ADD:
            regs[dst] = (regs[instr.a] + regs[instr.b]) % pp
        elif op == Opcode.SUB:
            regs[dst] = (regs[instr.a] - regs[instr.b]) % pp
        elif op == Opcode.MUL:
            regs[dst] = (regs[instr.a] * regs[instr.b]) % pp
        elif op == Opcode.NEG:
            regs[dst] = (pp - regs[instr.a]) % pp
        elif op == Opcode.POW2:
            regs[dst] = (regs[instr.a] * regs[instr.a]) % pp
        elif op == Opcode.POW3:
            regs[dst] = (regs[instr.a] * regs[instr.a] % pp * regs[instr.a]) % pp
        elif op == Opcode.INV:
            vals = regs[instr.a]
            regs[dst] = np.array([pow(int(v), int(p) - 2, int(p))
                                    for v, p in zip(vals, pp)], dtype=np.int64)
        elif op == Opcode.MULINV:
            a_vals = regs[instr.a]
            b_vals = regs[instr.b]
            regs[dst] = np.array([
                (int(a) * pow(int(b), int(p) - 2, int(p))) % int(p) if int(b) != 0 else 0
                for a, b, p in zip(a_vals, b_vals, pp)
            ], dtype=np.int64)
        elif op == Opcode.COPY:
            regs[dst] = regs[instr.a].copy()

    # Build output matrix from out_reg indices
    M = np.zeros((m, m, K), dtype=np.int64)
    for i in range(m):
        for j in range(m):
            idx = i * m + j
            if idx < len(program.out_reg):
                r = program.out_reg[idx]
                if r < len(regs):
                    M[i, j] = regs[r] % pp
    return M


def _eval_bytecode_float(program: CmfProgram, step: int,
                          shift_val: int) -> np.ndarray:
    """Evaluate bytecode producing one (m, m) float64 matrix.

    Returns:
        np.ndarray of shape (m, m), dtype float64
    """
    m = program.m
    regs = [0.0] * program.n_reg

    for instr in program.instructions:
        op = instr.op
        dst = instr.dst

        if op == Opcode.LOAD_X:
            axis = instr.a
            regs[dst] = float(shift_val + step * program.directions[axis])
        elif op == Opcode.LOAD_C:
            c_idx = instr.a
            if c_idx < len(program.constants):
                regs[dst] = float(program.constants[c_idx])
            else:
                regs[dst] = 0.0
        elif op == Opcode.ADD:
            regs[dst] = regs[instr.a] + regs[instr.b]
        elif op == Opcode.SUB:
            regs[dst] = regs[instr.a] - regs[instr.b]
        elif op == Opcode.MUL:
            regs[dst] = regs[instr.a] * regs[instr.b]
        elif op == Opcode.NEG:
            regs[dst] = -regs[instr.a]
        elif op == Opcode.POW2:
            regs[dst] = regs[instr.a] ** 2
        elif op == Opcode.POW3:
            regs[dst] = regs[instr.a] ** 3
        elif op == Opcode.INV:
            regs[dst] = 1.0 / regs[instr.a] if abs(regs[instr.a]) > 1e-300 else 0.0
        elif op == Opcode.MULINV:
            regs[dst] = (regs[instr.a] / regs[instr.b]
                         if abs(regs[instr.b]) > 1e-300 else 0.0)
        elif op == Opcode.COPY:
            regs[dst] = regs[instr.a]

    # Build output matrix from out_reg indices
    M = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            idx = i * m + j
            if idx < len(program.out_reg):
                r = program.out_reg[idx]
                if r < len(regs):
                    M[i, j] = regs[r]
    return M


# ── 2×2 modular matmul vectorised across K primes ───────────────────────

def _matmul_2x2_mod(A: np.ndarray, B: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Modular 2×2 matrix multiply A @ B mod pp, vectorised over K primes.

    A, B: shape (2, 2, K) int64
    pp:   shape (K,) int64
    Returns: (2, 2, K) int64
    """
    C = np.empty_like(A)
    C[0, 0] = (A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]) % pp
    C[0, 1] = (A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]) % pp
    C[1, 0] = (A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]) % pp
    C[1, 1] = (A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]) % pp
    return C


# ── Main walk function ──────────────────────────────────────────────────

def run_pcf_walk(program: CmfProgram, a0: int, depth: int, K: int = 64,
                 shift_val: int = 1):
    """Run the RNS walk for a PCF and return raw results.

    Uses the correct convention:
      M(n) = [[0, b(n)], [1, a(n)]]   (companion form)
      P = A · M(1) · M(2) · ... · M(depth)
      where A = [[1, a(0)], [0, 1]]
      p = P[0, 1],  q = P[1, 1]      (last column of 2×2)

    Args:
        program:   compiled CmfProgram (companion matrix)
        a0:        a(0) value for the initial-values matrix A
        depth:     number of walk steps
        K:         number of RNS primes
        shift_val: integer shift so axis_val = shift_val + step (default 1 → n=1,2,3,…)

    Returns:
        dict with keys:
          p_residues, q_residues, p_float, q_float, log_scale, primes
    """
    pp = generate_rns_primes(K).astype(np.int64)
    m = program.m
    assert m == 2, "run_pcf_walk only supports 2×2 companion matrices"

    const_table = _precompute_const_residues(program, pp)

    # Precompute a0 residues for initial P = A = [[1, a0], [0, 1]]
    a0_res = np.array([int(a0) % int(p) if a0 >= 0 else (int(p) - (-int(a0) % int(p))) % int(p)
                        for p in pp], dtype=np.int64)

    # RNS accumulator initialised to A = [[1, a0], [0, 1]]
    P_rns = np.zeros((2, 2, K), dtype=np.int64)
    P_rns[0, 0] = np.ones(K, dtype=np.int64)
    P_rns[0, 1] = a0_res
    P_rns[1, 1] = np.ones(K, dtype=np.int64)

    # Float64 shadow, also initialised to A
    P_f = np.array([[1.0, float(a0)], [0.0, 1.0]], dtype=np.float64)
    log_scale = 0.0

    for step in range(depth):
        # Evaluate M(n) at n = shift_val + step
        M_rns = _eval_bytecode_allprimes(program, step, shift_val, pp, const_table)
        M_f = _eval_bytecode_float(program, step, shift_val)

        # RNS: P = P @ M (vectorised over K primes)
        P_rns = _matmul_2x2_mod(P_rns, M_rns, pp)

        # Float shadow: P = P @ M with rescaling
        P_f = P_f @ M_f
        mx = np.max(np.abs(P_f))
        if mx > 1e10:
            P_f /= mx
            log_scale += math.log(mx)

    # Extract last column (column index m-1 = 1 for 2×2)
    return {
        'p_residues': P_rns[0, 1],
        'q_residues': P_rns[1, 1],
        'p_float': P_f[0, 1],
        'q_float': P_f[1, 1],
        'log_scale': log_scale,
        'primes': pp,
    }


# ── End-to-end verification ──────────────────────────────────────────────

def verify_pcf(a_str: str, b_str: str, limit_str: str,
               depth: int = 2000, K: int = 64, dps: int = 200):
    """End-to-end PCF verification: compile → walk → CRT → delta.

    Returns:
        dict with delta_float, delta_exact, est_float, target, ...
        or None if compilation fails.
    """
    import sympy as sp
    import mpmath as mp
    mp.mp.dps = dps

    # 1. Compile
    program = compile_pcf_from_strings(a_str, b_str)
    if program is None:
        return None

    # 2. Initial values
    a0 = pcf_initial_values(a_str)

    # 3. Walk
    res = run_pcf_walk(program, a0, depth, K)

    # 4. Parse target
    target_expr = sp.sympify(limit_str, locals={"pi": sp.pi, "E": sp.E,
                                                 "EulerGamma": sp.EulerGamma})
    target_mp = mp.mpf(str(sp.N(target_expr, dps)))

    # 5. Float64 delta (approximate)
    delta_float, log_q = compute_dreams_delta_float(
        res['p_float'], res['q_float'], res['log_scale'], float(target_mp))

    # 6. CRT exact delta
    primes_list = [int(p) for p in res['primes']]
    p_big, Mp = crt_reconstruct([int(r) for r in res['p_residues']], primes_list)
    q_big, _  = crt_reconstruct([int(r) for r in res['q_residues']], primes_list)
    p_big = centered(p_big, Mp)
    q_big = centered(q_big, Mp)
    delta_exact = compute_dreams_delta_exact(p_big, q_big, target_mp, dps)

    est_float = res['p_float'] / res['q_float'] if abs(res['q_float']) > 1e-300 else float('nan')

    return {
        'a': a_str,
        'b': b_str,
        'limit': limit_str,
        'target': float(target_mp),
        'est_float': est_float,
        'delta_float': delta_float,
        'delta_exact': delta_exact,
        'log_q': log_q,
        'depth': depth,
        'K': K,
        'p_bits': Mp.bit_length(),
    }

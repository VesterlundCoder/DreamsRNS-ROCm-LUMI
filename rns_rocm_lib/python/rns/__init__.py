"""
RNS-ROCm: Residue Number System library for exact modular arithmetic

This module provides Python bindings to the RNS-ROCm C++ library,
enabling fast modular arithmetic, bytecode evaluation, and matrix walks
for CMF-style computations.
"""

try:
    from rns_rocm import (
        # Op codes
        Op,
        # Prime generation
        generate_primes,
        # Scalar modular arithmetic
        add_mod, sub_mod, mul_mod, inv_mod, pow_mod,
        # Vectorized operations
        mul_mod_vec,
        # Bytecode evaluation
        eval_program,
        # Walk kernel
        walk_fused,
        # TopK selection
        topk,
        # Version
        __version__,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    __version__ = "0.2.0-pure"

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

@dataclass
class Instruction:
    """Single bytecode instruction."""
    op: int
    dst: int
    a: int
    b: int = 0

@dataclass  
class Program:
    """Bytecode program for matrix generation."""
    m: int                          # Matrix dimension
    dim: int                        # Number of x-variables
    instructions: List[Instruction]
    out_reg: List[int]              # Register indices for output matrix [m*m]
    n_reg: int                      # Total registers
    const_values: 'np.ndarray'      # Constant table [K, n_const]
    
    def to_arrays(self):
        """Convert to numpy arrays for C++ binding."""
        if not HAS_NUMPY:
            raise ImportError("numpy is required for to_arrays()")
        n = len(self.instructions)
        opcodes = np.array([i.op for i in self.instructions], dtype=np.uint8)
        dsts = np.array([i.dst for i in self.instructions], dtype=np.uint8)
        as_ = np.array([i.a for i in self.instructions], dtype=np.uint8)
        bs = np.array([i.b for i in self.instructions], dtype=np.uint8)
        out_reg = np.array(self.out_reg, dtype=np.uint16)
        return opcodes, dsts, as_, bs, out_reg

# Op codes (fallback if native not available)
class OpCodes:
    NOP = 0
    LOAD_X = 1
    LOAD_C = 2
    ADD = 3
    SUB = 4
    MUL = 5
    NEG = 6
    POW2 = 7
    POW3 = 8
    INV = 9
    MULINV = 10
    COPY = 11

# Pure Python fallbacks
def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0: return False
    return True

def generate_primes_py(K: int, seed: int = 12345) -> List[int]:
    """Generate K coprime 31-bit primes (pure Python fallback)."""
    primes = []
    state = seed
    
    def next_rand():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 32) & 0xFFFFFFFF
    
    while len(primes) < K:
        candidate = (next_rand() | (1 << 30)) | 1
        if candidate < (1 << 30):
            candidate += (1 << 30)
        if _is_prime(candidate) and candidate not in primes:
            primes.append(candidate)
    
    return primes

def add_mod_py(a: int, b: int, p: int) -> int:
    """Modular addition (pure Python)."""
    s = a + b
    return s - p if s >= p else s

def sub_mod_py(a: int, b: int, p: int) -> int:
    """Modular subtraction (pure Python)."""
    return (a - b) % p

def mul_mod_py(a: int, b: int, p: int) -> int:
    """Modular multiplication (pure Python)."""
    return (a * b) % p

def inv_mod_py(a: int, p: int) -> int:
    """Modular inverse using extended Euclidean algorithm (pure Python)."""
    return pow(a, p - 2, p)

def pow_mod_py(base: int, exp: int, p: int) -> int:
    """Modular exponentiation (pure Python)."""
    return pow(base, exp, p)

# Use native if available, else fallback
if not HAS_NATIVE:
    generate_primes = generate_primes_py
    add_mod = add_mod_py
    sub_mod = sub_mod_py
    mul_mod = mul_mod_py
    inv_mod = inv_mod_py
    pow_mod = pow_mod_py
    Op = OpCodes

def run_walk(
    prog: Program,
    shifts: 'np.ndarray',   # [B, dim]
    dirs: 'np.ndarray',     # [dim]
    primes: 'np.ndarray',   # [K]
    depth: int,
    depth1: int,
    depth2: int
) -> Dict:
    """
    Run fused walk kernel. Requires numpy.
    
    Args:
        prog: Bytecode program for step matrix
        shifts: Starting shifts [B, dim]
        dirs: Direction vector [dim]
        primes: Prime moduli [K]
        depth: Total walk depth
        depth1: First snapshot depth
        depth2: Second snapshot depth
        
    Returns:
        Dictionary with P_final, alive, est1, est2, delta1, delta2
    """
    B = shifts.shape[0]
    K = len(primes)
    
    opcodes, dsts, as_, bs, out_reg = prog.to_arrays()
    
    if HAS_NATIVE:
        return walk_fused(
            depth, depth1, depth2,
            prog.m, prog.dim, K, B,
            opcodes, dsts, as_, bs, out_reg,
            prog.const_values.astype(np.uint32),
            prog.n_reg,
            shifts.astype(np.int32),
            dirs.astype(np.int32),
            primes.astype(np.uint32)
        )
    else:
        # Pure Python fallback (slow)
        return _walk_fused_py(prog, shifts, dirs, primes, depth, depth1, depth2)

def _walk_fused_py(prog, shifts, dirs, primes, depth, depth1, depth2):
    """Pure Python walk implementation (slow fallback)."""
    B = shifts.shape[0]
    m = prog.m
    E = m * m
    p0 = int(primes[0])
    
    P_final = np.zeros((B, E), dtype=np.uint32)
    alive = np.ones(B, dtype=np.uint8)
    est1 = np.zeros(B, dtype=np.float32)
    est2 = np.zeros(B, dtype=np.float32)
    delta1 = np.zeros(B, dtype=np.float32)
    delta2 = np.zeros(B, dtype=np.float32)
    
    for b in range(B):
        P = np.eye(m, dtype=np.float64)
        P_mod = np.eye(m, dtype=np.uint64)
        
        for t in range(depth):
            # Compute x values
            x = [(int(shifts[b, j]) + t * int(dirs[j])) % p0 for j in range(prog.dim)]
            
            # Evaluate step matrix
            regs = [0] * prog.n_reg
            is_alive = True
            
            for instr in prog.instructions:
                if not is_alive:
                    break
                va = regs[instr.a]
                vb = regs[instr.b]
                
                if instr.op == OpCodes.LOAD_X:
                    regs[instr.dst] = x[instr.a]
                elif instr.op == OpCodes.LOAD_C:
                    regs[instr.dst] = int(prog.const_values[0, instr.a])
                elif instr.op == OpCodes.ADD:
                    regs[instr.dst] = (va + vb) % p0
                elif instr.op == OpCodes.SUB:
                    regs[instr.dst] = (va - vb) % p0
                elif instr.op == OpCodes.MUL:
                    regs[instr.dst] = (va * vb) % p0
                elif instr.op == OpCodes.INV:
                    if va == 0:
                        is_alive = False
                    else:
                        regs[instr.dst] = pow(va, p0 - 2, p0)
                elif instr.op == OpCodes.NEG:
                    regs[instr.dst] = (-va) % p0
            
            if not is_alive:
                alive[b] = 0
                break
            
            # Build step matrix
            M = np.array([[regs[prog.out_reg[i * m + j]] for j in range(m)] 
                          for i in range(m)], dtype=np.float64)
            M_mod = np.array([[regs[prog.out_reg[i * m + j]] for j in range(m)] 
                              for i in range(m)], dtype=np.uint64)
            
            # Multiply
            P = P @ M
            P_mod = (P_mod @ M_mod) % p0
            
            # Snapshots
            if t + 1 == depth1:
                est1[b] = np.linalg.norm(P)
                delta1[b] = abs(P[0, 1] / P[0, 0]) if P[0, 0] != 0 else 1e30
            if t + 1 == depth2:
                est2[b] = np.linalg.norm(P)
                delta2[b] = abs(P[0, 1] / P[0, 0]) if P[0, 0] != 0 else 1e30
        
        P_final[b] = P_mod.flatten().astype(np.uint32)
    
    return {
        'P_final': P_final,
        'alive': alive,
        'est1': est1,
        'est2': est2,
        'delta1': delta1,
        'delta2': delta2
    }

def select_topk(
    scores: 'np.ndarray',
    est: 'np.ndarray',
    k: int,
    ascending: bool = True
) -> Dict:
    """
    Select top-K candidates by score.
    
    Args:
        scores: Score array [B]
        est: Estimate array [B]
        k: Number to keep
        ascending: If True, smallest scores first
        
    Returns:
        Dictionary with scores, indices, est arrays
    """
    if HAS_NATIVE:
        return topk(scores.astype(np.float32), est.astype(np.float32), k, ascending)
    else:
        # Pure Python fallback
        if ascending:
            idx = np.argsort(scores)[:k]
        else:
            idx = np.argsort(scores)[::-1][:k]
        return {
            'scores': scores[idx],
            'indices': idx,
            'est': est[idx]
        }

# Import SymPy compiler if available
try:
    from .sympy_compiler import compile_matrix, disassemble, HAS_SYMPY
except ImportError:
    HAS_SYMPY = False
    compile_matrix = None
    disassemble = None

__all__ = [
    'Op', 'OpCodes', 'Instruction', 'Program',
    'generate_primes', 'generate_primes_py',
    'add_mod', 'sub_mod', 'mul_mod', 'inv_mod', 'pow_mod',
    'run_walk', 'select_topk',
    'compile_matrix', 'disassemble',
    'HAS_NATIVE', 'HAS_NUMPY', 'HAS_SYMPY', '__version__'
]

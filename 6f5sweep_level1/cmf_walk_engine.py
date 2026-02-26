#!/usr/bin/env python3
"""
cmf_walk_engine.py — Bytecode-compiled pFq CMF walk engine for 6F5 sweep.

Compiles the pFq companion matrix to bytecode at startup (sympy, once),
then runs walks as vectorised numpy float64 — no sympy at runtime.

Usage:
    from cmf_walk_engine import compile_6f5, walk_single, walk_batch
    program = compile_6f5(cmf_dict)
    est = walk_single(program, shift_nums, shift_dens, direction, depth=2000)
"""
from __future__ import annotations
import itertools, math
from enum import IntEnum
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Dict, Tuple, Optional
import numpy as np

# ── Bytecode (matches RNS-ROCm Op enum) ──
class Op(IntEnum):
    NOP=0; LOAD_X=1; LOAD_C=2; ADD=3; SUB=4; MUL=5
    NEG=6; POW2=7; POW3=8; INV=9; MULINV=10; COPY=11

@dataclass
class Instr:
    op: int; dst: int; a: int = 0; b: int = 0

@dataclass
class Program:
    m: int; dim: int
    instrs: List[Instr] = field(default_factory=list)
    out_reg: List[int] = field(default_factory=list)
    n_reg: int = 0
    consts: List[int] = field(default_factory=list)
    name: str = ""

# ── Compiler ──
class _Comp:
    MAX_REGS = 512
    def __init__(self, m, dim):
        self.m, self.dim = m, dim
        self.ins: List[Instr] = []
        self.cmap: Dict[int,int] = {}
        self.nr = 0
        self.eregs: Dict[Tuple[int,int],int] = {}

    def _a(self):
        r = self.nr; self.nr += 1
        assert self.nr < self.MAX_REGS; return r

    def _ac(self, v):
        if v not in self.cmap: self.cmap[v] = len(self.cmap)
        return self.cmap[v]

    def _e(self, op, dst, a=0, b=0): self.ins.append(Instr(op, dst, a, b))
    def lx(self, ax): r=self._a(); self._e(Op.LOAD_X,r,ax); return r
    def lc(self, v): r=self._a(); self._e(Op.LOAD_C,r,self._ac(v)); return r
    def add(self,a,b): r=self._a(); self._e(Op.ADD,r,a,b); return r
    def sub(self,a,b): r=self._a(); self._e(Op.SUB,r,a,b); return r
    def mul(self,a,b): r=self._a(); self._e(Op.MUL,r,a,b); return r
    def neg(self,a): r=self._a(); self._e(Op.NEG,r,a); return r
    def pw2(self,a): r=self._a(); self._e(Op.POW2,r,a); return r
    def pw3(self,a): r=self._a(); self._e(Op.POW3,r,a); return r
    def inv(self,a): r=self._a(); self._e(Op.INV,r,a); return r
    def mi(self,a,b): r=self._a(); self._e(Op.MULINV,r,a,b); return r

    def store(self, r, i, j): self.eregs[(i,j)] = r

    def build(self, name="") -> Program:
        out = []
        for i in range(self.m):
            for j in range(self.m):
                if (i,j) in self.eregs:
                    out.append(self.eregs[(i,j)])
                else:
                    idx = self._ac(0); r = self._a()
                    self._e(Op.LOAD_C, r, idx); out.append(r)
        cl = [0]*len(self.cmap)
        for v,i in self.cmap.items(): cl[i] = v
        return Program(m=self.m, dim=self.dim, instrs=list(self.ins),
                       out_reg=out, n_reg=self.nr, consts=cl, name=name)


# ── Sympy expression → bytecode (startup only) ──

def _cx(comp: _Comp, expr, amap):
    """Recursively compile a sympy expression."""
    import sympy as sp
    if isinstance(expr, (int, sp.Integer)):
        return comp.lc(int(expr))
    if isinstance(expr, sp.Symbol):
        return comp.lx(amap[str(expr)])
    if isinstance(expr, sp.Rational) and not isinstance(expr, sp.Integer):
        return comp.mi(comp.lc(int(expr.p)), comp.lc(int(expr.q)))
    if isinstance(expr, sp.Add):
        args = list(expr.args)
        r = _cx(comp, args[0], amap)
        for a in args[1:]:
            r = comp.add(r, _cx(comp, a, amap))
        return r
    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        if args[0] == -1:
            inner = sp.Mul(*args[1:]) if len(args) > 2 else args[1]
            return comp.neg(_cx(comp, inner, amap))
        r = _cx(comp, args[0], amap)
        for a in args[1:]:
            r = comp.mul(r, _cx(comp, a, amap))
        return r
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == 2: return comp.pw2(_cx(comp, base, amap))
        if exp == 3: return comp.pw3(_cx(comp, base, amap))
        if exp == -1: return comp.inv(_cx(comp, base, amap))
        if isinstance(exp, (int, sp.Integer)) and int(exp) > 0:
            r = _cx(comp, base, amap)
            result = r
            for _ in range(int(exp) - 1):
                result = comp.mul(result, r)
            return result
        if isinstance(exp, (int, sp.Integer)) and int(exp) < 0:
            pos = _cx(comp, sp.Pow(base, -exp), amap)
            return comp.inv(pos)
    raise ValueError(f"Unsupported expr: {type(expr)}: {expr}")


def _build_companion(p: int, q: int, a_params, b_params):
    """Build pFq r×r companion matrix as dict of sympy-expression strings."""
    r = max(p, q) + 1
    axes = [f"x{i}" for i in range(p)] + [f"y{j}" for j in range(q)]
    af = [f"({axes[i]} + {a_params[i]})" for i in range(p)]
    bf = [f"({axes[p+j]} + {b_params[j]})" for j in range(q)]

    mat: Dict[str, str] = {}
    for i in range(r):
        for j in range(r):
            mat[f"{i},{j}"] = "1" if i == j + 1 else "0"

    # Last column
    mat[f"0,{r-1}"] = " * ".join(af) if af else "1"
    for row in range(1, r - 1):
        k = r - 1 - row
        bn_terms = [" * ".join(c) for c in itertools.combinations(bf, k)] if k <= len(bf) else []
        an_terms = [" * ".join(c) for c in itertools.combinations(af, k)] if k <= len(af) else []
        bn_ek = " + ".join(bn_terms) if bn_terms else "0"
        an_ek = " + ".join(an_terms) if an_terms else "0"
        mat[f"{row},{r-1}"] = f"({bn_ek}) - ({an_ek})"
    mat[f"{r-1},{r-1}"] = " * ".join(bf) if bf else "1"
    return r, mat, axes


def compile_6f5(cmf: dict) -> Program:
    """Compile a pFq CMF dict to a bytecode Program.

    Args:
        cmf: dict with keys p, q, upper_params, lower_params.

    Returns:
        Compiled Program ready for walk_single / walk_batch.
    """
    import sympy as sp

    p_val = cmf["p"]
    q_val = cmf["q"]
    a_params = [Fraction(x) for x in cmf["upper_params"]]
    b_params = [Fraction(x) for x in cmf["lower_params"]]
    rank, mat_dict, axes = _build_companion(p_val, q_val, a_params, b_params)
    dim = p_val + q_val
    comp = _Comp(m=rank, dim=dim)
    amap = {name: idx for idx, name in enumerate(axes)}

    for i in range(rank):
        for j in range(rank):
            s = mat_dict[f"{i},{j}"]
            if s == "0":
                continue
            if s == "1":
                expr = sp.Integer(1)
            else:
                local_ns = {n: sp.Symbol(n) for n in axes}
                expr = sp.sympify(s, locals=local_ns)
            reg = _cx(comp, expr, amap)
            comp.store(reg, i, j)
            saved = set(comp.eregs.values())
            comp.nr = max(saved) + 1 if saved else 0

    name = cmf.get("cmf_id", f"{p_val}F{q_val}")
    return comp.build(name=name)


# ═══════════════════════════════════════════════════════════════════════
# Float64 bytecode evaluator  (NO sympy at runtime)
# ═══════════════════════════════════════════════════════════════════════

def _eval_matrix_float(prog: Program, axis_vals: np.ndarray) -> np.ndarray:
    """Evaluate bytecode for ONE walk, returning (m, m) float64 matrix.

    axis_vals: shape (dim,) float64 — current axis values at this step.
    """
    m = prog.m
    regs = [0.0] * prog.n_reg
    consts = prog.consts

    for ins in prog.instrs:
        op, dst, a, b = ins.op, ins.dst, ins.a, ins.b
        if op == Op.LOAD_X:
            regs[dst] = axis_vals[a]
        elif op == Op.LOAD_C:
            regs[dst] = float(consts[a]) if a < len(consts) else 0.0
        elif op == Op.ADD:
            regs[dst] = regs[a] + regs[b]
        elif op == Op.SUB:
            regs[dst] = regs[a] - regs[b]
        elif op == Op.MUL:
            regs[dst] = regs[a] * regs[b]
        elif op == Op.NEG:
            regs[dst] = -regs[a]
        elif op == Op.POW2:
            regs[dst] = regs[a] * regs[a]
        elif op == Op.POW3:
            regs[dst] = regs[a] ** 3
        elif op == Op.INV:
            regs[dst] = 1.0 / regs[a] if abs(regs[a]) > 1e-300 else 0.0
        elif op == Op.MULINV:
            regs[dst] = regs[a] / regs[b] if abs(regs[b]) > 1e-300 else 0.0

    M = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            idx = i * m + j
            r = prog.out_reg[idx]
            M[i, j] = regs[r]
    return M


def _eval_matrix_batch(prog: Program, axis_vals: np.ndarray) -> np.ndarray:
    """Evaluate bytecode for B walks simultaneously.

    axis_vals: shape (B, dim) float64
    Returns: (B, m, m) float64
    """
    m = prog.m
    B = axis_vals.shape[0]
    regs = [np.zeros(B, dtype=np.float64) for _ in range(prog.n_reg)]
    consts = prog.consts

    for ins in prog.instrs:
        op, dst, a, b = ins.op, ins.dst, ins.a, ins.b
        if op == Op.LOAD_X:
            regs[dst] = axis_vals[:, a].copy()
        elif op == Op.LOAD_C:
            regs[dst] = np.full(B, float(consts[a]) if a < len(consts) else 0.0)
        elif op == Op.ADD:
            regs[dst] = regs[a] + regs[b]
        elif op == Op.SUB:
            regs[dst] = regs[a] - regs[b]
        elif op == Op.MUL:
            regs[dst] = regs[a] * regs[b]
        elif op == Op.NEG:
            regs[dst] = -regs[a]
        elif op == Op.POW2:
            regs[dst] = regs[a] * regs[a]
        elif op == Op.POW3:
            regs[dst] = regs[a] ** 3
        elif op == Op.INV:
            safe = np.where(np.abs(regs[a]) > 1e-300, regs[a], 1.0)
            regs[dst] = np.where(np.abs(regs[a]) > 1e-300, 1.0 / safe, 0.0)
        elif op == Op.MULINV:
            safe = np.where(np.abs(regs[b]) > 1e-300, regs[b], 1.0)
            regs[dst] = np.where(np.abs(regs[b]) > 1e-300, regs[a] / safe, 0.0)

    M = np.zeros((B, m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            idx = i * m + j
            r = prog.out_reg[idx]
            M[:, i, j] = regs[r]
    return M


# ═══════════════════════════════════════════════════════════════════════
# Walk functions
# ═══════════════════════════════════════════════════════════════════════

def walk_single(
    prog: Program,
    shift_nums: List[int],
    shift_dens: List[int],
    direction: List[int],
    depth: int = 2000,
) -> Optional[Tuple[float, bool]]:
    """Run one CMF walk with dual-shadow validation.

    Two independent float64 product accumulators run with different
    Frobenius-norm renormalization schedules (every 32 and every 47 steps).
    If their estimates agree to ~1e-6 relative, the result is confident.
    Otherwise the result is flagged uncertain (caller should save anyway).

    Returns:
        (estimate, confident) tuple, or None if both shadows diverge.
        estimate: float p/q ratio.
        confident: True if both shadows agree within 1e-6 relative.
    """
    m = prog.m
    sn = np.array(shift_nums, dtype=np.float64)
    sd = np.array(shift_dens, dtype=np.float64)
    dr = np.array(direction, dtype=np.float64)
    start = sn / sd

    # Dual shadows with different renormalization cadences
    P_a = np.eye(m, dtype=np.float64)
    P_b = np.eye(m, dtype=np.float64)
    NORM_A, NORM_B = 32, 47  # coprime cadences reduce correlated bias

    for step in range(depth):
        axis_vals = start + step * dr
        M = _eval_matrix_float(prog, axis_vals)
        P_a = P_a @ M
        P_b = P_b @ M

        # Frobenius-norm renormalization (smoother than max-entry)
        if (step + 1) % NORM_A == 0:
            fn = np.sqrt(np.sum(P_a * P_a))
            if fn > 1e-300:
                P_a /= fn
        if (step + 1) % NORM_B == 0:
            fn = np.sqrt(np.sum(P_b * P_b))
            if fn > 1e-300:
                P_b /= fn

    # Extract estimates from both shadows
    qa = P_a[m - 1, m - 1]
    qb = P_b[m - 1, m - 1]
    est_a = P_a[0, m - 1] / qa if abs(qa) > 1e-300 else None
    est_b = P_b[0, m - 1] / qb if abs(qb) > 1e-300 else None

    if est_a is None and est_b is None:
        return None

    # Use shadow A as primary, B as cross-check
    primary = est_a if est_a is not None else est_b

    # Confidence: do both shadows agree?
    if est_a is not None and est_b is not None:
        denom = max(abs(est_a), abs(est_b), 1e-15)
        rel_diff = abs(est_a - est_b) / denom
        confident = rel_diff < 1e-6
    else:
        confident = False

    return (primary, confident)


def walk_batch(
    prog: Program,
    shifts_nums: np.ndarray,
    shifts_dens: np.ndarray,
    direction: List[int],
    depth: int = 2000,
    normalize_every: int = 32,
) -> np.ndarray:
    """Run B CMF walks in parallel, return array of estimates.

    shifts_nums: (B, dim) int array of shift numerators.
    shifts_dens: (B, dim) int array of shift denominators.
    direction:   (dim,) int array — shared trajectory for all B walks.

    Returns: (B,) float64 array of p/q estimates (NaN where q≈0).
    """
    m = prog.m
    dim = prog.dim
    B = shifts_nums.shape[0]

    starts = shifts_nums.astype(np.float64) / shifts_dens.astype(np.float64)
    dr = np.array(direction, dtype=np.float64)

    P = np.zeros((B, m, m), dtype=np.float64)
    for i in range(m):
        P[:, i, i] = 1.0

    for step in range(depth):
        axis_vals = starts + step * dr  # (B, dim)
        M = _eval_matrix_batch(prog, axis_vals)   # (B, m, m)
        P = np.matmul(P, M)

        if (step + 1) % normalize_every == 0:
            # Frobenius norm per walk (smoother than max-entry)
            scale = np.sqrt(np.sum(P * P, axis=(1, 2), keepdims=True))
            scale = np.where(scale > 1e-300, scale, 1.0)
            P = P / scale

    p_vals = P[:, 0, m - 1]
    q_vals = P[:, m - 1, m - 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        estimates = np.where(np.abs(q_vals) > 1e-300, p_vals / q_vals, np.nan)
    return estimates

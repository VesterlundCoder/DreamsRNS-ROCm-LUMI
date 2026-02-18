"""
CMF Compiler: Converts symbolic CMF expressions to bytecode programs.

Ported from Dreams-RNS-CUDA compiler.py with ROCm-compatible output.
Supports both SymPy matrix input and dictionary-based definition.

The bytecode is evaluated on GPU via the RNS-ROCm bytecode evaluator
or on CPU via the pure-Python fallback.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import struct
import numpy as np


class Opcode(IntEnum):
    """Bytecode opcodes matching RNS-ROCm Op enum."""
    NOP = 0
    LOAD_X = 1      # dst = x[a]
    LOAD_C = 2      # dst = const_table[a]
    ADD = 3
    SUB = 4
    MUL = 5
    NEG = 6
    POW2 = 7
    POW3 = 8
    INV = 9
    MULINV = 10
    COPY = 11


@dataclass
class Instruction:
    """Single bytecode instruction."""
    op: int
    dst: int
    a: int = 0
    b: int = 0


@dataclass
class CmfProgram:
    """Compiled CMF program ready for GPU/CPU execution.

    Attributes:
        m: Matrix dimension.
        dim: Number of axes (x-variables).
        instructions: Bytecode instructions.
        out_reg: Register indices for output matrix [m*m], row-major.
        n_reg: Total number of registers used.
        constants: List of integer constants.
        directions: Walk direction for each axis.
        name: Human-readable name (optional).
    """
    m: int
    dim: int
    instructions: List[Instruction] = field(default_factory=list)
    out_reg: List[int] = field(default_factory=list)
    n_reg: int = 0
    constants: List[int] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)
    name: str = ""

    def to_arrays(self):
        """Convert to numpy arrays for native C++ binding."""
        n = len(self.instructions)
        opcodes = np.array([i.op for i in self.instructions], dtype=np.uint8)
        dsts = np.array([i.dst for i in self.instructions], dtype=np.uint8)
        as_ = np.array([i.a for i in self.instructions], dtype=np.uint8)
        bs = np.array([i.b for i in self.instructions], dtype=np.uint8)
        out_reg = np.array(self.out_reg, dtype=np.uint16)
        return opcodes, dsts, as_, bs, out_reg

    def make_const_table(self, K: int, primes: np.ndarray) -> np.ndarray:
        """Build constant table [K, n_const] with constants reduced mod each prime.

        Args:
            K: Number of primes.
            primes: Array of K primes.

        Returns:
            uint32 array of shape [K, n_const].
        """
        n_const = max(1, len(self.constants))
        table = np.zeros((K, n_const), dtype=np.uint32)
        for k in range(K):
            p = int(primes[k])
            for c_idx, c_val in enumerate(self.constants):
                if c_val >= 0:
                    table[k, c_idx] = c_val % p
                else:
                    table[k, c_idx] = (p - ((-c_val) % p)) % p
        return table

    def serialize(self) -> bytes:
        """Serialize to binary for caching."""
        data = struct.pack('IIIII',
                           self.m, self.dim, len(self.instructions),
                           len(self.constants), self.n_reg)
        for instr in self.instructions:
            data += struct.pack('BBBB', instr.op, instr.dst, instr.a, instr.b)
        for c in self.constants:
            data += struct.pack('q', c)
        for d in self.directions:
            data += struct.pack('i', d)
        for r in self.out_reg:
            data += struct.pack('H', r)
        return data


class CmfCompiler:
    """Compiles CMF symbolic expressions to bytecode.

    Usage:
        compiler = CmfCompiler(m=4, dim=2)
        # Add matrix entries
        r0 = compiler.load_axis(0)
        r1 = compiler.load_const(1)
        r2 = compiler.add(r0, r1)
        compiler.store(r2, 0, 0)
        # ... etc
        program = compiler.build()
    """

    MAX_REGS = 512

    def __init__(self, m: int, dim: int,
                 directions: Optional[List[int]] = None,
                 name: str = ""):
        self.m = m
        self.dim = dim
        self.directions = directions or [1] * dim
        self.name = name
        self.instructions: List[Instruction] = []
        self.constants: Dict[int, int] = {}  # value -> index
        self.next_reg = 0
        self.entry_regs: Dict[Tuple[int, int], int] = {}

    def _alloc_reg(self) -> int:
        reg = self.next_reg
        self.next_reg += 1
        if self.next_reg >= self.MAX_REGS:
            raise RuntimeError("Out of registers")
        return reg

    def _add_const(self, value: int) -> int:
        if value not in self.constants:
            self.constants[value] = len(self.constants)
        return self.constants[value]

    def _emit(self, op: int, dst: int, a: int = 0, b: int = 0):
        self.instructions.append(Instruction(op, dst, a, b))

    def load_axis(self, axis: int) -> int:
        reg = self._alloc_reg()
        self._emit(Opcode.LOAD_X, reg, axis)
        return reg

    def load_const(self, value: int) -> int:
        idx = self._add_const(value)
        reg = self._alloc_reg()
        self._emit(Opcode.LOAD_C, reg, idx)
        return reg

    def add(self, r1: int, r2: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.ADD, dest, r1, r2)
        return dest

    def sub(self, r1: int, r2: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.SUB, dest, r1, r2)
        return dest

    def mul(self, r1: int, r2: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.MUL, dest, r1, r2)
        return dest

    def neg(self, r: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.NEG, dest, r)
        return dest

    def pow2(self, r: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.POW2, dest, r)
        return dest

    def pow3(self, r: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.POW3, dest, r)
        return dest

    def inv(self, r: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.INV, dest, r)
        return dest

    def mulinv(self, r_num: int, r_den: int) -> int:
        dest = self._alloc_reg()
        self._emit(Opcode.MULINV, dest, r_num, r_den)
        return dest

    def store(self, r: int, row: int, col: int):
        self.entry_regs[(row, col)] = r

    def build(self) -> CmfProgram:
        """Build the final CmfProgram."""
        # Build output register map: row-major order
        out_reg = []
        for i in range(self.m):
            for j in range(self.m):
                if (i, j) in self.entry_regs:
                    out_reg.append(self.entry_regs[(i, j)])
                else:
                    # Unset entries default to zero constant
                    idx = self._add_const(0)
                    reg = self._alloc_reg()
                    self._emit(Opcode.LOAD_C, reg, idx)
                    out_reg.append(reg)

        const_list = [0] * len(self.constants)
        for val, idx in self.constants.items():
            const_list[idx] = val

        return CmfProgram(
            m=self.m,
            dim=self.dim,
            instructions=list(self.instructions),
            out_reg=out_reg,
            n_reg=self.next_reg,
            constants=const_list,
            directions=list(self.directions),
            name=self.name,
        )


# ---------------------------------------------------------------------------
# SymPy compilation
# ---------------------------------------------------------------------------

def _compile_sympy_expr(compiler: CmfCompiler, expr, axis_symbols: Dict[str, int]) -> int:
    """Recursively compile a SymPy expression."""
    import sympy as sp

    if isinstance(expr, (int, sp.Integer)):
        return compiler.load_const(int(expr))

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name in axis_symbols:
            return compiler.load_axis(axis_symbols[name])
        raise ValueError(f"Unknown symbol: {name}")

    if isinstance(expr, sp.Rational) and not isinstance(expr, sp.Integer):
        num = compiler.load_const(int(expr.p))
        den = compiler.load_const(int(expr.q))
        return compiler.mulinv(num, den)

    if isinstance(expr, sp.Add):
        args = list(expr.args)
        result = _compile_sympy_expr(compiler, args[0], axis_symbols)
        for arg in args[1:]:
            r = _compile_sympy_expr(compiler, arg, axis_symbols)
            result = compiler.add(result, r)
        return result

    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        if args[0] == -1:
            inner = sp.Mul(*args[1:]) if len(args) > 2 else args[1]
            r = _compile_sympy_expr(compiler, inner, axis_symbols)
            return compiler.neg(r)
        result = _compile_sympy_expr(compiler, args[0], axis_symbols)
        for arg in args[1:]:
            r = _compile_sympy_expr(compiler, arg, axis_symbols)
            result = compiler.mul(result, r)
        return result

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == 2:
            r = _compile_sympy_expr(compiler, base, axis_symbols)
            return compiler.pow2(r)
        if exp == 3:
            r = _compile_sympy_expr(compiler, base, axis_symbols)
            return compiler.pow3(r)
        if exp == -1:
            r = _compile_sympy_expr(compiler, base, axis_symbols)
            return compiler.inv(r)
        if isinstance(exp, (int, sp.Integer)) and int(exp) > 0:
            r = _compile_sympy_expr(compiler, base, axis_symbols)
            result = r
            for _ in range(int(exp) - 1):
                result = compiler.mul(result, r)
            return result
        if isinstance(exp, (int, sp.Integer)) and int(exp) < 0:
            pos = _compile_sympy_expr(compiler, sp.Pow(base, -exp), axis_symbols)
            return compiler.inv(pos)
        raise ValueError(f"Unsupported power: {expr}")

    raise ValueError(f"Unsupported expression type: {type(expr)}: {expr}")


def compile_cmf_from_sympy(
    matrix,
    symbols: List,
    directions: Optional[List[int]] = None,
    name: str = "",
    K: int = 1,
) -> CmfProgram:
    """Compile a SymPy Matrix to CmfProgram.

    Args:
        matrix: Square SymPy Matrix.
        symbols: List of SymPy Symbol objects for axes.
        directions: Walk direction per axis.
        name: Human-readable name.
        K: Number of primes (for constant table).

    Returns:
        Compiled CmfProgram.
    """
    import sympy as sp

    m = matrix.rows
    assert matrix.rows == matrix.cols, "Matrix must be square"
    dim = len(symbols)

    axis_symbols = {str(s): i for i, s in enumerate(symbols)}
    compiler = CmfCompiler(m=m, dim=dim, directions=directions or [1]*dim, name=name)

    for i in range(m):
        for j in range(m):
            entry = matrix[i, j]
            if entry != 0:
                reg = _compile_sympy_expr(compiler, entry, axis_symbols)
                compiler.store(reg, i, j)
                # Reclaim intermediate registers: keep only result registers alive
                saved = set(compiler.entry_regs.values())
                compiler.next_reg = max(saved) + 1 if saved else 0

    return compiler.build()


def compile_pcf_from_strings(
    a_str: str,
    b_str: str,
) -> Optional[CmfProgram]:
    """Compile PCF(a(n), b(n)) to bytecode using correct companion matrix.

    Builds the companion matrix matching ramanujantools convention:
        M(n) = [[0, b(n)], [1, a(n)]]

    The walk variable n is mapped to LOAD_X axis 0 with shift=1, direction=1
    so that at step t the axis value is 1+t = 1, 2, 3, ...

    Returns None if the PCF contains imaginary numbers.
    """
    import sympy as sp
    from sympy.abc import n

    a_expr = sp.sympify(a_str)
    b_expr = sp.sympify(b_str)

    if a_expr.has(sp.I) or b_expr.has(sp.I):
        return None

    # Rename n -> _s so the compiler uses LOAD_X (axis 0) instead of a
    # hypothetical LOAD_N opcode. With shift=1, direction=1:
    #   axis_val = 1 + step*1 = 1, 2, 3, ... which gives n = 1, 2, 3, ...
    _s = sp.Symbol("_s")
    a_expr = a_expr.subs(n, _s)
    b_expr = b_expr.subs(n, _s)

    axis_symbols = {"_s": 0}
    compiler = CmfCompiler(m=2, dim=1, directions=[1], name=f"PCF({a_str}, {b_str})")

    # M[0,0] = 0 — skip (matrix initialised to zero by build())
    # M[0,1] = b(n)
    reg_b = _compile_sympy_expr(compiler, b_expr, axis_symbols)
    compiler.store(reg_b, 0, 1)
    saved = set(compiler.entry_regs.values())
    compiler.next_reg = max(saved) + 1 if saved else 0

    # M[1,0] = 1
    reg_one = _compile_sympy_expr(compiler, sp.Integer(1), axis_symbols)
    compiler.store(reg_one, 1, 0)
    saved = set(compiler.entry_regs.values())
    compiler.next_reg = max(saved) + 1 if saved else 0

    # M[1,1] = a(n)
    reg_a = _compile_sympy_expr(compiler, a_expr, axis_symbols)
    compiler.store(reg_a, 1, 1)

    return compiler.build()


def pcf_initial_values(a_str: str) -> int:
    """Compute a(0) for the initial-values matrix A = [[1, a(0)], [0, 1]].

    Returns a(0) evaluated as integer.
    """
    import sympy as sp
    from sympy.abc import n
    return int(sp.sympify(a_str).subs(n, 0))


def compile_cmf_from_dict(
    matrix_dict: Dict[Tuple[int, int], str],
    m: int,
    dim: int,
    axis_names: Optional[List[str]] = None,
    directions: Optional[List[int]] = None,
    name: str = "",
) -> CmfProgram:
    """Compile CMF from a dictionary of matrix entries.

    Args:
        matrix_dict: {(row, col): "expression_string"} dictionary.
        m: Matrix dimension.
        dim: Number of axes.
        axis_names: Symbol names for axes (default: x0, x1, ...).
        directions: Walk directions.
        name: Human-readable name.

    Returns:
        Compiled CmfProgram.
    """
    import sympy as sp

    if axis_names is None:
        axis_names = [f'x{i}' for i in range(dim)]

    sym_dict = {n: sp.Symbol(n) for n in axis_names}
    sym_dict['n'] = sp.Symbol('n')
    symbols = [sym_dict[n] for n in axis_names]

    M = sp.zeros(m, m)
    for (row, col), expr_str in matrix_dict.items():
        M[row, col] = sp.sympify(expr_str, locals=sym_dict)

    return compile_cmf_from_sympy(
        M, symbols, directions=directions or [1]*dim, name=name,
    )

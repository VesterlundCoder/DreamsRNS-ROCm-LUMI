"""
SymPy to RNS Bytecode Compiler

Compiles SymPy matrix expressions into bytecode for GPU evaluation.

Example:
    import sympy as sp
    from rns.sympy_compiler import compile_matrix
    
    x0, x1 = sp.symbols('x0 x1')
    M = sp.Matrix([
        [x0 + 1, x1],
        [1, x0 * x1]
    ])
    
    prog = compile_matrix(M, [x0, x1], K=32)
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import sympy as sp
    from sympy import (
        Matrix, Symbol, Integer, Rational, Add, Mul, Pow,
        S, cse, simplify
    )
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    sp = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from . import Instruction, Program, OpCodes


@dataclass
class CompilerState:
    """Internal compiler state."""
    instructions: List[Instruction] = field(default_factory=list)
    constants: List[int] = field(default_factory=list)
    reg_map: Dict[Any, int] = field(default_factory=dict)
    next_reg: int = 0
    symbols: List['sp.Symbol'] = field(default_factory=list)
    
    def alloc_reg(self) -> int:
        """Allocate a new register."""
        reg = self.next_reg
        self.next_reg += 1
        return reg
    
    def emit(self, op: int, dst: int, a: int = 0, b: int = 0):
        """Emit an instruction."""
        self.instructions.append(Instruction(op, dst, a, b))
    
    def add_constant(self, value: int) -> int:
        """Add a constant and return its index."""
        if value in self.constants:
            return self.constants.index(value)
        idx = len(self.constants)
        self.constants.append(value)
        return idx


class BytecodeCompiler:
    """Compiles SymPy expressions to RNS bytecode."""
    
    def __init__(self, matrix: 'sp.Matrix', symbols: List['sp.Symbol']):
        if not HAS_SYMPY:
            raise ImportError("sympy is required for BytecodeCompiler")
        
        self.matrix = matrix
        self.symbols = symbols
        self.m = matrix.rows
        assert matrix.rows == matrix.cols, "Matrix must be square"
        
        self.state = CompilerState(symbols=symbols)
    
    def compile(self, K: int = 1) -> Program:
        """
        Compile the matrix to a Program.
        
        Args:
            K: Number of primes (for constant table sizing)
            
        Returns:
            Program ready for evaluation
        """
        # 1. Apply CSE (Common Subexpression Elimination)
        replacements, reduced = cse(list(self.matrix))
        
        # 2. Load symbols into registers
        for i, sym in enumerate(self.symbols):
            reg = self.state.alloc_reg()
            self.state.emit(OpCodes.LOAD_X, reg, a=i)
            self.state.reg_map[sym] = reg
        
        # 3. Compile CSE temporaries
        for var, expr in replacements:
            self._compile_expr(expr)
            self.state.reg_map[var] = self.state.reg_map[expr]
        
        # 4. Compile matrix entries and collect output registers
        out_reg = []
        for entry in reduced:
            self._compile_expr(entry)
            out_reg.append(self.state.reg_map[entry])
        
        # 5. Build constant table [K, n_const]
        n_const = len(self.state.constants)
        if HAS_NUMPY:
            const_values = np.zeros((K, max(1, n_const)), dtype=np.uint32)
            for i, c in enumerate(self.state.constants):
                # Constants are the same for all primes (will be reduced at eval time)
                const_values[:, i] = c if c >= 0 else 0  # Handle negatives separately
        else:
            const_values = [[c for c in self.state.constants] for _ in range(K)]
        
        return Program(
            m=self.m,
            dim=len(self.symbols),
            instructions=self.state.instructions,
            out_reg=out_reg,
            n_reg=self.state.next_reg,
            const_values=const_values
        )
    
    def _compile_expr(self, expr) -> int:
        """Compile an expression, returning its register."""
        # Check if already compiled
        if expr in self.state.reg_map:
            return self.state.reg_map[expr]
        
        # Handle different expression types
        if isinstance(expr, Symbol):
            # Should already be loaded
            if expr not in self.state.reg_map:
                raise ValueError(f"Unknown symbol: {expr}")
            return self.state.reg_map[expr]
        
        elif isinstance(expr, Integer):
            val = int(expr)
            if val == 0:
                # Zero constant
                idx = self.state.add_constant(0)
                reg = self.state.alloc_reg()
                self.state.emit(OpCodes.LOAD_C, reg, a=idx)
            elif val == 1:
                # One constant
                idx = self.state.add_constant(1)
                reg = self.state.alloc_reg()
                self.state.emit(OpCodes.LOAD_C, reg, a=idx)
            elif val > 0:
                idx = self.state.add_constant(val)
                reg = self.state.alloc_reg()
                self.state.emit(OpCodes.LOAD_C, reg, a=idx)
            else:
                # Negative: load |val| and negate
                idx = self.state.add_constant(-val)
                reg = self.state.alloc_reg()
                self.state.emit(OpCodes.LOAD_C, reg, a=idx)
                neg_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.NEG, neg_reg, a=reg)
                reg = neg_reg
            
            self.state.reg_map[expr] = reg
            return reg
        
        elif isinstance(expr, Rational):
            # a/b -> a * inv(b)
            num = self._compile_expr(Integer(expr.p))
            den = self._compile_expr(Integer(expr.q))
            
            reg = self.state.alloc_reg()
            self.state.emit(OpCodes.MULINV, reg, a=num, b=den)
            self.state.reg_map[expr] = reg
            return reg
        
        elif isinstance(expr, Add):
            # Compile all operands
            args = list(expr.args)
            regs = [self._compile_expr(arg) for arg in args]
            
            # Chain additions
            result = regs[0]
            for i in range(1, len(regs)):
                new_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.ADD, new_reg, a=result, b=regs[i])
                result = new_reg
            
            self.state.reg_map[expr] = result
            return result
        
        elif isinstance(expr, Mul):
            # Handle coefficient and terms
            coeff = expr.as_coeff_Mul()[0]
            terms = expr.as_coeff_Mul()[1]
            
            if coeff == -1:
                # -1 * terms -> negate
                terms_reg = self._compile_expr(terms)
                neg_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.NEG, neg_reg, a=terms_reg)
                self.state.reg_map[expr] = neg_reg
                return neg_reg
            
            # Compile all factors
            args = list(expr.args)
            regs = [self._compile_expr(arg) for arg in args]
            
            # Chain multiplications
            result = regs[0]
            for i in range(1, len(regs)):
                new_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.MUL, new_reg, a=result, b=regs[i])
                result = new_reg
            
            self.state.reg_map[expr] = result
            return result
        
        elif isinstance(expr, Pow):
            base, exp = expr.args
            
            if exp == -1:
                # Inverse
                base_reg = self._compile_expr(base)
                inv_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.INV, inv_reg, a=base_reg)
                self.state.reg_map[expr] = inv_reg
                return inv_reg
            
            elif exp == 2:
                # Square
                base_reg = self._compile_expr(base)
                sqr_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.POW2, sqr_reg, a=base_reg)
                self.state.reg_map[expr] = sqr_reg
                return sqr_reg
            
            elif exp == 3:
                # Cube
                base_reg = self._compile_expr(base)
                cube_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.POW3, cube_reg, a=base_reg)
                self.state.reg_map[expr] = cube_reg
                return cube_reg
            
            elif isinstance(exp, Integer) and int(exp) > 0:
                # Positive integer power: repeated squaring
                base_reg = self._compile_expr(base)
                exp_val = int(exp)
                
                # Binary exponentiation
                result_reg = None
                temp_reg = base_reg
                
                while exp_val > 0:
                    if exp_val & 1:
                        if result_reg is None:
                            result_reg = temp_reg
                        else:
                            new_reg = self.state.alloc_reg()
                            self.state.emit(OpCodes.MUL, new_reg, a=result_reg, b=temp_reg)
                            result_reg = new_reg
                    
                    exp_val >>= 1
                    if exp_val > 0:
                        sqr_reg = self.state.alloc_reg()
                        self.state.emit(OpCodes.POW2, sqr_reg, a=temp_reg)
                        temp_reg = sqr_reg
                
                self.state.reg_map[expr] = result_reg
                return result_reg
            
            elif isinstance(exp, Integer) and int(exp) < 0:
                # Negative power: compute positive power then invert
                pos_exp = -int(exp)
                pos_expr = Pow(base, Integer(pos_exp))
                pos_reg = self._compile_expr(pos_expr)
                
                inv_reg = self.state.alloc_reg()
                self.state.emit(OpCodes.INV, inv_reg, a=pos_reg)
                self.state.reg_map[expr] = inv_reg
                return inv_reg
            
            else:
                raise NotImplementedError(f"Unsupported power: {expr}")
        
        else:
            # Try to simplify and retry
            simplified = simplify(expr)
            if simplified != expr:
                return self._compile_expr(simplified)
            
            raise NotImplementedError(f"Unsupported expression type: {type(expr)}: {expr}")


def compile_matrix(
    matrix: 'sp.Matrix',
    symbols: List['sp.Symbol'],
    K: int = 1
) -> Program:
    """
    Compile a SymPy matrix to RNS bytecode.
    
    Args:
        matrix: Square SymPy Matrix
        symbols: List of Symbol variables (x0, x1, ...)
        K: Number of primes for constant table
        
    Returns:
        Program ready for evaluation
        
    Example:
        >>> import sympy as sp
        >>> x0, x1 = sp.symbols('x0 x1')
        >>> M = sp.Matrix([[x0 + 1, x1], [1, x0 * x1]])
        >>> prog = compile_matrix(M, [x0, x1], K=32)
    """
    if not HAS_SYMPY:
        raise ImportError("sympy is required for compile_matrix")
    
    compiler = BytecodeCompiler(matrix, symbols)
    return compiler.compile(K)


def disassemble(prog: Program) -> str:
    """
    Disassemble a program to human-readable format.
    
    Args:
        prog: Compiled program
        
    Returns:
        String representation of the bytecode
    """
    OP_NAMES = {
        OpCodes.NOP: "NOP",
        OpCodes.LOAD_X: "LOAD_X",
        OpCodes.LOAD_C: "LOAD_C",
        OpCodes.ADD: "ADD",
        OpCodes.SUB: "SUB",
        OpCodes.MUL: "MUL",
        OpCodes.NEG: "NEG",
        OpCodes.POW2: "POW2",
        OpCodes.POW3: "POW3",
        OpCodes.INV: "INV",
        OpCodes.MULINV: "MULINV",
        OpCodes.COPY: "COPY",
    }
    
    lines = [f"Program: m={prog.m}, dim={prog.dim}, n_reg={prog.n_reg}"]
    lines.append(f"Constants: {list(prog.const_values[0]) if HAS_NUMPY else prog.const_values[0]}")
    lines.append("")
    lines.append("Instructions:")
    
    for i, instr in enumerate(prog.instructions):
        op_name = OP_NAMES.get(instr.op, f"OP_{instr.op}")
        if instr.op == OpCodes.LOAD_X:
            lines.append(f"  {i:3d}: r{instr.dst} = x[{instr.a}]")
        elif instr.op == OpCodes.LOAD_C:
            lines.append(f"  {i:3d}: r{instr.dst} = const[{instr.a}]")
        elif instr.op in (OpCodes.ADD, OpCodes.SUB, OpCodes.MUL, OpCodes.MULINV):
            op_sym = {OpCodes.ADD: '+', OpCodes.SUB: '-', 
                      OpCodes.MUL: '*', OpCodes.MULINV: '*inv'}[instr.op]
            lines.append(f"  {i:3d}: r{instr.dst} = r{instr.a} {op_sym} r{instr.b}")
        elif instr.op in (OpCodes.NEG, OpCodes.INV, OpCodes.POW2, OpCodes.POW3):
            op_func = {OpCodes.NEG: '-', OpCodes.INV: 'inv',
                       OpCodes.POW2: 'sqr', OpCodes.POW3: 'cube'}[instr.op]
            lines.append(f"  {i:3d}: r{instr.dst} = {op_func}(r{instr.a})")
        else:
            lines.append(f"  {i:3d}: {op_name} r{instr.dst}, r{instr.a}, r{instr.b}")
    
    lines.append("")
    lines.append(f"Output registers: {prog.out_reg}")
    
    # Format as matrix
    lines.append(f"Output matrix ({prog.m}x{prog.m}):")
    for i in range(prog.m):
        row = [f"r{prog.out_reg[i * prog.m + j]}" for j in range(prog.m)]
        lines.append(f"  [{', '.join(row)}]")
    
    return "\n".join(lines)


__all__ = ['compile_matrix', 'disassemble', 'BytecodeCompiler', 'HAS_SYMPY']

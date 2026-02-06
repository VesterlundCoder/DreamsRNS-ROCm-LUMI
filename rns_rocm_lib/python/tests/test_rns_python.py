"""
Tests for RNS-ROCm Python bindings.

Run with: pytest tests/test_rns_python.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rns import (
    generate_primes, add_mod, sub_mod, mul_mod, inv_mod, pow_mod,
    Instruction, Program, OpCodes, run_walk, select_topk,
    HAS_NATIVE
)


class TestPrimeGeneration:
    def test_generate_primes_count(self):
        primes = generate_primes(16)
        assert len(primes) == 16
    
    def test_generate_primes_unique(self):
        primes = generate_primes(32)
        assert len(primes) == len(set(primes))
    
    def test_generate_primes_31bit(self):
        primes = generate_primes(8)
        for p in primes:
            assert p >= (1 << 30), f"Prime {p} is less than 2^30"
            assert p < (1 << 32), f"Prime {p} is >= 2^32"
    
    def test_generate_primes_deterministic(self):
        p1 = generate_primes(8, seed=42)
        p2 = generate_primes(8, seed=42)
        assert p1 == p2


class TestModularArithmetic:
    @pytest.fixture
    def prime(self):
        return generate_primes(1)[0]
    
    def test_add_mod(self, prime):
        p = prime
        assert add_mod(10, 20, p) == 30
        assert add_mod(p - 1, 1, p) == 0
        assert add_mod(p - 1, 2, p) == 1
    
    def test_sub_mod(self, prime):
        p = prime
        assert sub_mod(30, 10, p) == 20
        assert sub_mod(0, 1, p) == p - 1
        assert sub_mod(10, 20, p) == (p - 10)
    
    def test_mul_mod(self, prime):
        p = prime
        assert mul_mod(10, 20, p) == 200
        assert mul_mod(p - 1, 2, p) == p - 2
    
    def test_inv_mod(self, prime):
        p = prime
        for a in [1, 2, 7, 123, p - 1]:
            inv = inv_mod(a, p)
            assert mul_mod(a, inv, p) == 1, f"inv_mod failed for {a}"
    
    def test_pow_mod(self, prime):
        p = prime
        assert pow_mod(2, 10, p) == 1024
        assert pow_mod(3, 0, p) == 1
        # Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
        assert pow_mod(7, p - 1, p) == 1


class TestProgram:
    def test_instruction_creation(self):
        instr = Instruction(op=OpCodes.ADD, dst=2, a=0, b=1)
        assert instr.op == OpCodes.ADD
        assert instr.dst == 2
    
    def test_program_to_arrays(self):
        instrs = [
            Instruction(OpCodes.LOAD_X, 0, 0),
            Instruction(OpCodes.LOAD_C, 1, 0),
            Instruction(OpCodes.ADD, 2, 0, 1),
        ]
        const_values = np.array([[1, 2]], dtype=np.uint32)
        out_reg = [2, 0, 0, 2]  # 2x2 matrix
        
        prog = Program(
            m=2, dim=1, instructions=instrs,
            out_reg=out_reg, n_reg=3, const_values=const_values
        )
        
        opcodes, dsts, as_, bs, out_r = prog.to_arrays()
        assert len(opcodes) == 3
        assert opcodes[0] == OpCodes.LOAD_X
        assert opcodes[2] == OpCodes.ADD


class TestWalk:
    def test_identity_walk(self):
        """Walk with identity matrix should preserve identity."""
        m = 2
        K = 1
        B = 4
        
        # Program that generates identity matrix
        instrs = [
            Instruction(OpCodes.LOAD_C, 0, 0),  # r0 = 1
            Instruction(OpCodes.LOAD_C, 1, 1),  # r1 = 0
        ]
        const_values = np.array([[1, 0]], dtype=np.uint32)
        out_reg = [0, 1, 1, 0]  # [[1, 0], [0, 1]]
        
        prog = Program(
            m=m, dim=1, instructions=instrs,
            out_reg=out_reg, n_reg=2, const_values=const_values
        )
        
        primes = np.array(generate_primes(K), dtype=np.uint32)
        shifts = np.zeros((B, 1), dtype=np.int32)
        dirs = np.ones(1, dtype=np.int32)
        
        result = run_walk(prog, shifts, dirs, primes, depth=10, depth1=5, depth2=10)
        
        # After identity walk, P should still be identity
        assert result['alive'].all()
        for b in range(B):
            P = result['P_final'][b].reshape(m, m)
            assert P[0, 0] == 1
            assert P[0, 1] == 0
            assert P[1, 0] == 0
            assert P[1, 1] == 1


class TestTopK:
    def test_topk_ascending(self):
        scores = np.array([5, 3, 8, 1, 9, 2], dtype=np.float32)
        est = np.arange(6, dtype=np.float32)
        
        result = select_topk(scores, est, k=3, ascending=True)
        
        # Should select indices with smallest scores: 3, 5, 1 (scores 1, 2, 3)
        assert 3 in result['indices']
        assert 5 in result['indices']
        assert 1 in result['indices']
    
    def test_topk_descending(self):
        scores = np.array([5, 3, 8, 1, 9, 2], dtype=np.float32)
        est = np.arange(6, dtype=np.float32)
        
        result = select_topk(scores, est, k=3, ascending=False)
        
        # Should select indices with largest scores: 4, 2, 0 (scores 9, 8, 5)
        assert 4 in result['indices']
        assert 2 in result['indices']
        assert 0 in result['indices']
    
    def test_topk_preserves_est(self):
        scores = np.array([5, 3, 8], dtype=np.float32)
        est = np.array([100, 200, 300], dtype=np.float32)
        
        result = select_topk(scores, est, k=2, ascending=True)
        
        # Check that est values match the selected indices
        for i, idx in enumerate(result['indices']):
            assert result['est'][i] == est[idx]


class TestNativeVsPython:
    """Test that native and Python implementations match."""
    
    @pytest.mark.skipif(not HAS_NATIVE, reason="Native module not available")
    def test_mul_mod_matches(self):
        from rns import mul_mod as native_mul
        from rns import mul_mod_py
        
        p = generate_primes(1)[0]
        for _ in range(100):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            assert native_mul(a, b, p) == mul_mod_py(a, b, p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

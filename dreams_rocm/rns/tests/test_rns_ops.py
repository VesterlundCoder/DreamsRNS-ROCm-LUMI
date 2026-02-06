"""
Unit tests for RNS modular arithmetic operations.

Compares RNS add/mul/mod reductions against Python big-int reference.
Covers edge cases: overflow, modulus product range, negative handling.
"""

import unittest
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dreams_rocm.rns.reference import (
    generate_primes, compute_barrett_mu,
    add_mod, sub_mod, mul_mod, inv_mod, pow_mod, neg_mod, fma_mod,
    rns_encode, rns_encode_signed,
    crt_reconstruct, crt_reconstruct_signed,
    rns_add, rns_mul, rns_sub,
    is_prime,
)


class TestPrimeGeneration(unittest.TestCase):
    """Test prime generation."""

    def test_generate_primes_count(self):
        for K in [4, 8, 16, 32, 64]:
            primes = generate_primes(K)
            self.assertEqual(len(primes), K)

    def test_all_primes(self):
        primes = generate_primes(32)
        for p in primes:
            self.assertTrue(is_prime(p), f"{p} is not prime")

    def test_all_31bit(self):
        primes = generate_primes(32)
        for p in primes:
            self.assertGreaterEqual(p, 1 << 30)
            self.assertLess(p, 1 << 31)

    def test_all_distinct(self):
        primes = generate_primes(64)
        self.assertEqual(len(set(primes)), 64)


class TestScalarModOps(unittest.TestCase):
    """Test scalar modular arithmetic against Python big-int reference."""

    def setUp(self):
        self.primes = generate_primes(8)
        self.rng = random.Random(42)

    def test_add_mod(self):
        for p in self.primes:
            for _ in range(100):
                a = self.rng.randint(0, p - 1)
                b = self.rng.randint(0, p - 1)
                got = add_mod(a, b, p)
                want = (a + b) % p
                self.assertEqual(got, want, f"add_mod({a},{b},{p})")

    def test_sub_mod(self):
        for p in self.primes:
            for _ in range(100):
                a = self.rng.randint(0, p - 1)
                b = self.rng.randint(0, p - 1)
                got = sub_mod(a, b, p)
                want = (a - b) % p
                self.assertEqual(got, want)

    def test_mul_mod(self):
        for p in self.primes:
            for _ in range(100):
                a = self.rng.randint(0, p - 1)
                b = self.rng.randint(0, p - 1)
                got = mul_mod(a, b, p)
                want = (a * b) % p
                self.assertEqual(got, want)

    def test_inv_mod(self):
        for p in self.primes:
            for _ in range(50):
                a = self.rng.randint(1, p - 1)  # nonzero
                inv_a = inv_mod(a, p)
                self.assertEqual((a * inv_a) % p, 1,
                                 f"inv_mod({a}, {p}) = {inv_a}")

    def test_inv_mod_zero(self):
        for p in self.primes:
            self.assertEqual(inv_mod(0, p), 0)

    def test_neg_mod(self):
        for p in self.primes:
            for _ in range(50):
                a = self.rng.randint(0, p - 1)
                got = neg_mod(a, p)
                want = (p - a) % p
                self.assertEqual(got, want)

    def test_pow_mod(self):
        for p in self.primes:
            for _ in range(50):
                base = self.rng.randint(1, p - 1)
                exp = self.rng.randint(0, 20)
                got = pow_mod(base, exp, p)
                want = pow(base, exp, p)
                self.assertEqual(got, want)

    def test_fma_mod(self):
        for p in self.primes:
            for _ in range(50):
                a = self.rng.randint(0, p - 1)
                b = self.rng.randint(0, p - 1)
                c = self.rng.randint(0, p - 1)
                got = fma_mod(a, b, c, p)
                want = (a * b + c) % p
                self.assertEqual(got, want)


class TestRnsEncodeDecodeRoundtrip(unittest.TestCase):
    """Test RNS encode -> CRT decode round-trip."""

    def setUp(self):
        self.primes = generate_primes(16)

    def test_small_integers(self):
        for x in range(100):
            residues = rns_encode(x, self.primes)
            y = crt_reconstruct(residues, self.primes)
            self.assertEqual(x, y, f"round-trip failed for x={x}")

    def test_large_integers(self):
        rng = random.Random(123)
        M = 1
        for p in self.primes:
            M *= p
        for _ in range(20):
            x = rng.randint(0, M - 1)
            residues = rns_encode(x, self.primes)
            y = crt_reconstruct(residues, self.primes)
            self.assertEqual(x, y)

    def test_signed_round_trip(self):
        rng = random.Random(456)
        M = 1
        for p in self.primes:
            M *= p
        half_M = M // 2
        for _ in range(20):
            x = rng.randint(-half_M + 1, half_M - 1)
            residues = rns_encode_signed(x, self.primes)
            y = crt_reconstruct_signed(residues, self.primes)
            self.assertEqual(x, y, f"signed round-trip failed for x={x}")


class TestRnsArithmetic(unittest.TestCase):
    """Test RNS-level add/mul/sub match big-int reference."""

    def setUp(self):
        self.primes = generate_primes(8)
        self.rng = random.Random(789)
        # Compute M for range
        self.M = 1
        for p in self.primes:
            self.M *= p

    def test_rns_add(self):
        for _ in range(50):
            a = self.rng.randint(0, self.M // 2)
            b = self.rng.randint(0, self.M // 2)
            a_res = rns_encode(a, self.primes)
            b_res = rns_encode(b, self.primes)
            c_res = rns_add(a_res, b_res, self.primes)
            c = crt_reconstruct(c_res, self.primes)
            self.assertEqual(c, (a + b) % self.M)

    def test_rns_mul(self):
        # Use smaller values to avoid overflow beyond M
        for _ in range(50):
            a = self.rng.randint(0, 1 << 60)
            b = self.rng.randint(0, 1 << 60)
            a_res = rns_encode(a, self.primes)
            b_res = rns_encode(b, self.primes)
            c_res = rns_mul(a_res, b_res, self.primes)
            c = crt_reconstruct(c_res, self.primes)
            self.assertEqual(c, (a * b) % self.M)

    def test_rns_sub(self):
        for _ in range(50):
            a = self.rng.randint(0, self.M - 1)
            b = self.rng.randint(0, self.M - 1)
            a_res = rns_encode(a, self.primes)
            b_res = rns_encode(b, self.primes)
            c_res = rns_sub(a_res, b_res, self.primes)
            c = crt_reconstruct(c_res, self.primes)
            self.assertEqual(c, (a - b) % self.M)


class TestEdgeCases(unittest.TestCase):
    """Edge cases: overflow, boundary, negative handling."""

    def test_max_value_near_M(self):
        primes = generate_primes(4)
        M = 1
        for p in primes:
            M *= p
        x = M - 1
        residues = rns_encode(x, primes)
        y = crt_reconstruct(residues, primes)
        self.assertEqual(x, y)

    def test_zero(self):
        primes = generate_primes(8)
        residues = rns_encode(0, primes)
        y = crt_reconstruct(residues, primes)
        self.assertEqual(0, y)

    def test_one(self):
        primes = generate_primes(8)
        residues = rns_encode(1, primes)
        y = crt_reconstruct(residues, primes)
        self.assertEqual(1, y)

    def test_negative_signed(self):
        primes = generate_primes(8)
        for x in [-1, -100, -999999]:
            residues = rns_encode_signed(x, primes)
            y = crt_reconstruct_signed(residues, primes)
            self.assertEqual(x, y, f"failed for x={x}")

    def test_overflow_detection(self):
        """Values exceeding M wrap around correctly."""
        primes = generate_primes(4)
        M = 1
        for p in primes:
            M *= p
        x = M + 42
        residues = rns_encode(x, primes)
        y = crt_reconstruct(residues, primes)
        self.assertEqual(y, 42)  # x mod M


if __name__ == "__main__":
    unittest.main()

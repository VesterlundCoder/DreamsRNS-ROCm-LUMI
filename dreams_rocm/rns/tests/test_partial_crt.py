"""
Unit tests for partial CRT reconstruction.

Tests that partial CRT (using K_small primes) gives enough precision
for delta proxy computation, and that full CRT matches reference.
"""

import unittest
import random
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dreams_rocm.rns.reference import (
    generate_primes, rns_encode, crt_reconstruct, crt_reconstruct_signed,
)
from dreams_rocm.crt.partial_crt import partial_crt_delta_proxy
from dreams_rocm.crt.full_crt_cpu import full_crt_verify


class TestPartialCrt(unittest.TestCase):
    """Test partial CRT reconstruction for delta proxy."""

    def setUp(self):
        self.K = 32
        self.primes = generate_primes(self.K)

    def test_partial_crt_small_values(self):
        """Partial CRT should exactly reconstruct small values."""
        K_small = 4
        small_primes = self.primes[:K_small]
        for x in [0, 1, 42, 9999, 123456789]:
            residues = rns_encode(x, small_primes)
            y = crt_reconstruct(residues, small_primes)
            self.assertEqual(x, y)

    def test_partial_crt_ratio(self):
        """Partial CRT ratio p/q should approximate the exact ratio."""
        K_small = 6
        small_primes = self.primes[:K_small]

        # Simulate a p and q value
        p_val = 314159265
        q_val = 100000000
        target = math.pi

        p_residues = rns_encode(p_val, small_primes)
        q_residues = rns_encode(q_val, small_primes)

        p_recon = crt_reconstruct(p_residues, small_primes)
        q_recon = crt_reconstruct(q_residues, small_primes)

        self.assertEqual(p_recon, p_val)
        self.assertEqual(q_recon, q_val)

        ratio = p_recon / q_recon
        self.assertAlmostEqual(ratio, target, places=5)

    def test_delta_proxy_computation(self):
        """Test the delta proxy pipeline: encode -> partial CRT -> delta."""
        K_small = 4
        small_primes = self.primes[:K_small]

        p_val = 355
        q_val = 113
        target = math.pi  # 355/113 ≈ 3.14159292... close to pi

        result = partial_crt_delta_proxy(
            p_residues=rns_encode(p_val, small_primes),
            q_residues=rns_encode(q_val, small_primes),
            primes=small_primes,
            target=target,
        )

        self.assertIn("delta", result)
        self.assertIn("ratio", result)
        # 355/113 is a famous pi approximation
        self.assertAlmostEqual(result["ratio"], 355 / 113, places=10)
        # Delta should be positive (it's a good approximation)
        self.assertGreater(result["delta"], 0.0)


class TestFullCrt(unittest.TestCase):
    """Test full CRT reconstruction on CPU."""

    def setUp(self):
        self.primes = generate_primes(16)

    def test_full_crt_exact(self):
        """Full CRT should exactly reconstruct."""
        rng = random.Random(42)
        M = 1
        for p in self.primes:
            M *= p

        for _ in range(20):
            x = rng.randint(0, min(M - 1, 10**50))
            residues = rns_encode(x, self.primes)
            y = crt_reconstruct(residues, self.primes)
            self.assertEqual(x, y)

    def test_full_crt_verify_zeta3(self):
        """Test full CRT verification against zeta(3)."""
        result = full_crt_verify(
            p_val=6004799503160661,
            q_val=4999999999999995,
            target_name="zeta3",
            dps=50,
        )
        self.assertIn("verified", result)
        self.assertIn("delta", result)


if __name__ == "__main__":
    unittest.main()

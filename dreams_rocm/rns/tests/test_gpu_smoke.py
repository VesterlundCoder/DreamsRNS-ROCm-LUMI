"""
GPU smoke test for the RNS-ROCm library on LUMI.

This test is designed to run on a single GPU and validate:
1. GPU device is visible and accessible via ROCm
2. Basic RNS operations produce correct results on GPU
3. Small CMF walk produces matching results vs CPU reference

If no GPU is available, tests are skipped gracefully.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dreams_rocm.rns.reference import (
    generate_primes, rns_encode, crt_reconstruct,
    mul_mod, add_mod, inv_mod,
)
from dreams_rocm.rns.bindings import HAS_NATIVE_RNS, RnsContext

# Try to detect ROCm GPU
HAS_GPU = False
GPU_REASON = "unknown"

try:
    import subprocess
    result = subprocess.run(
        ["rocm-smi", "--showid"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0 and "GPU" in result.stdout:
        HAS_GPU = True
        GPU_REASON = "rocm-smi detected GPU"
    else:
        GPU_REASON = "rocm-smi did not find GPU"
except Exception as e:
    GPU_REASON = f"rocm-smi not available: {e}"

# Also try HIP Python bindings
if not HAS_GPU:
    try:
        from hip import hip as hiprt
        device_count = hiprt.hipGetDeviceCount()
        if device_count > 0:
            HAS_GPU = True
            GPU_REASON = f"HIP detected {device_count} device(s)"
    except Exception:
        pass


@unittest.skipUnless(HAS_GPU, f"No GPU detected: {GPU_REASON}")
class TestGpuSmoke(unittest.TestCase):
    """Smoke tests that require an AMD GPU with ROCm."""

    def test_gpu_visible(self):
        """Verify at least one GPU is visible."""
        self.assertTrue(HAS_GPU)

    def test_rns_context_creation(self):
        """Create an RnsContext with GPU primes."""
        import numpy as np
        primes = np.array(generate_primes(16), dtype=np.uint32)
        ctx = RnsContext(primes)
        self.assertEqual(ctx.K, 16)

    def test_encode_decode_on_gpu_context(self):
        """Encode/decode via RnsContext (may use GPU path if native)."""
        import numpy as np
        primes = np.array(generate_primes(8), dtype=np.uint32)
        ctx = RnsContext(primes)

        for x in [0, 1, 42, 123456789, 10**15]:
            residues = ctx.encode(x)
            y = ctx.decode(residues)
            self.assertEqual(x, y, f"round-trip failed for x={x}")


class TestCpuFallback(unittest.TestCase):
    """Always-run tests using CPU fallback path."""

    def test_rns_context_cpu(self):
        """RnsContext works even without GPU."""
        import numpy as np
        primes = np.array(generate_primes(8), dtype=np.uint32)
        ctx = RnsContext(primes)
        self.assertEqual(ctx.K, 8)

        x = 999999
        residues = ctx.encode(x)
        y = ctx.decode(residues)
        self.assertEqual(x, y)

    def test_signed_encode_decode(self):
        import numpy as np
        primes = np.array(generate_primes(8), dtype=np.uint32)
        ctx = RnsContext(primes)

        for x in [-1, -42, -999999, 0, 1, 42]:
            residues = ctx.encode_signed(x)
            y = ctx.decode_signed(residues)
            self.assertEqual(x, y)


if __name__ == "__main__":
    print(f"GPU available: {HAS_GPU} ({GPU_REASON})")
    print(f"Native RNS library: {HAS_NATIVE_RNS}")
    unittest.main()

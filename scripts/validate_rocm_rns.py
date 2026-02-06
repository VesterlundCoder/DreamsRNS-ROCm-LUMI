#!/usr/bin/env python3
"""
Validation script for RNS-ROCm on LUMI-G.

Runs a sequence of checks:
1. GPU detection (ROCm / HIP)
2. RNS modular arithmetic correctness
3. CRT reconstruction roundtrip
4. Partial CRT delta proxy
5. Trajectory generation uniqueness
6. Small CMF walk smoke test

Usage:
    python scripts/validate_rocm_rns.py
    srun -n1 --gpus=1 python scripts/validate_rocm_rns.py  # on compute node
"""

import sys
import os
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    mark = "\u2713" if passed else "\u2717"
    print(f"  [{status}] {mark} {name}" + (f" -- {detail}" if detail else ""))
    return passed


def main():
    print("Dreams-RNS-ROCm Validation Suite")
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")

    results = []

    # ---------------------------------------------------------------
    # 1. GPU Detection
    # ---------------------------------------------------------------
    section("1. GPU / ROCm Detection")

    gpu_found = False
    gpu_detail = ""

    # Check ROCR_VISIBLE_DEVICES
    rocr_dev = os.environ.get("ROCR_VISIBLE_DEVICES", "not set")
    print(f"  ROCR_VISIBLE_DEVICES = {rocr_dev}")

    # Try rocm-smi
    try:
        import subprocess
        r = subprocess.run(["rocm-smi", "--showid"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0 and "GPU" in r.stdout:
            gpu_found = True
            gpu_detail = "rocm-smi detected GPU"
            print(f"  rocm-smi output (first 5 lines):")
            for line in r.stdout.strip().splitlines()[:5]:
                print(f"    {line}")
        else:
            gpu_detail = f"rocm-smi returned code {r.returncode}"
    except Exception as e:
        gpu_detail = f"rocm-smi not available: {e}"

    results.append(check("GPU detected via rocm-smi", gpu_found, gpu_detail))

    # Try HIP Python
    hip_ok = False
    try:
        from hip import hip as hiprt
        count = hiprt.hipGetDeviceCount()
        hip_ok = count > 0
        results.append(check("HIP Python binding", hip_ok,
                             f"{count} device(s)"))
    except Exception as e:
        results.append(check("HIP Python binding", False, str(e)))

    # ---------------------------------------------------------------
    # 2. RNS Modular Arithmetic
    # ---------------------------------------------------------------
    section("2. RNS Modular Arithmetic")

    try:
        from dreams_rocm.rns.reference import (
            generate_primes, add_mod, mul_mod, inv_mod,
            rns_encode, crt_reconstruct,
        )

        primes = generate_primes(16)
        results.append(check("Generate 16 primes", len(primes) == 16,
                             f"range [{min(primes)}, {max(primes)}]"))

        # Test mul + inv
        p = primes[0]
        a, b = 123456, 789012
        c = mul_mod(a % p, b % p, p)
        expected = (a * b) % p
        results.append(check("mul_mod correctness", c == expected))

        # Test inv_mod
        a_mod = a % p
        a_inv = inv_mod(a_mod, p)
        product = mul_mod(a_mod, a_inv, p)
        results.append(check("inv_mod correctness", product == 1))

    except Exception as e:
        results.append(check("RNS arithmetic", False, str(e)))
        traceback.print_exc()

    # ---------------------------------------------------------------
    # 3. CRT Roundtrip
    # ---------------------------------------------------------------
    section("3. CRT Reconstruction Roundtrip")

    try:
        primes = generate_primes(16)
        test_values = [0, 1, 42, 999999, 10**15, 10**30]
        all_ok = True
        for x in test_values:
            residues = rns_encode(x, primes)
            y = crt_reconstruct(residues, primes)
            if x != y:
                all_ok = False
                print(f"    FAIL: x={x}, got y={y}")

        results.append(check("CRT roundtrip (6 values)", all_ok))

    except Exception as e:
        results.append(check("CRT roundtrip", False, str(e)))
        traceback.print_exc()

    # ---------------------------------------------------------------
    # 4. Partial CRT Delta Proxy
    # ---------------------------------------------------------------
    section("4. Partial CRT Delta Proxy")

    try:
        from dreams_rocm.crt.partial_crt import partial_crt_delta_proxy
        import math

        primes = generate_primes(8)
        # 355/113 is a famous pi approximation
        p_res = rns_encode(355, primes)
        q_res = rns_encode(113, primes)

        result = partial_crt_delta_proxy(p_res, q_res, primes, math.pi,
                                         K_small=4)
        results.append(check("Partial CRT delta computed",
                             result["delta"] > 0,
                             f"delta={result['delta']:.4f}"))

    except Exception as e:
        results.append(check("Partial CRT", False, str(e)))
        traceback.print_exc()

    # ---------------------------------------------------------------
    # 5. Trajectory Generation
    # ---------------------------------------------------------------
    section("5. Trajectory Generation")

    try:
        from dreams_rocm.trajectories import (
            generate_trajectories, normalize_trajectory,
        )

        trajs = generate_trajectories(1000)
        results.append(check("Generate 1000 trajectories",
                             len(trajs) >= 1000,
                             f"got {len(trajs)}"))

        # Check uniqueness
        unique = len(set(trajs)) == len(trajs)
        results.append(check("All trajectories unique", unique))

        # Check normalization
        all_normalized = True
        for dn, dk in trajs:
            normed = normalize_trajectory(dn, dk)
            if normed != (dn, dk):
                all_normalized = False
                break
        results.append(check("All trajectories normalized", all_normalized))

        # Check determinism
        trajs2 = generate_trajectories(1000)
        results.append(check("Deterministic ordering", trajs == trajs2))

    except Exception as e:
        results.append(check("Trajectory generation", False, str(e)))
        traceback.print_exc()

    # ---------------------------------------------------------------
    # 6. CMF Compile + Small Walk
    # ---------------------------------------------------------------
    section("6. CMF Compile + Small Walk Smoke Test")

    try:
        from dreams_rocm.cmf_compile import compile_cmf_from_dict
        from dreams_rocm.runner import DreamsRunner, WalkConfig
        from dreams_rocm.shifts import generate_shifts
        import numpy as np

        # Compile Apery zeta(3) CMF
        program = compile_cmf_from_dict(
            matrix_dict={
                (0, 0): "n**3",
                (0, 1): "1",
                (1, 0): "-(n+1)**6",
                (1, 1): "34*n**3 + 51*n**2 + 27*n + 5",
            },
            m=2, dim=1,
            axis_names=["n"],
            directions=[1],
            name="Apery_Zeta3",
        )
        results.append(check("Compile Apery CMF",
                             len(program.instructions) > 0,
                             f"{len(program.instructions)} instructions"))

        # Small walk
        config = WalkConfig(
            K=8, B=10, depth=50, topk=5,
            target=1.2020569031595942,
            target_name="zeta3",
            snapshot_depths=(25, 50),
        )

        runner = DreamsRunner([program], config=config)
        shifts = generate_shifts(n_shifts=10, dim=1, method="grid",
                                 bounds=(-50, 50))

        t0 = time.time()
        hits, metrics = runner.run_single(program, shifts, cmf_idx=0)
        dt = time.time() - t0

        results.append(check("Walk completed",
                             metrics.get("wall_time_sec", 0) > 0,
                             f"{dt:.3f}s, {len(hits)} hits"))

    except Exception as e:
        results.append(check("CMF smoke test", False, str(e)))
        traceback.print_exc()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    section("Summary")

    n_pass = sum(1 for r in results if r)
    n_fail = sum(1 for r in results if not r)
    n_total = len(results)

    print(f"\n  {n_pass}/{n_total} checks passed, {n_fail} failed")

    if n_fail == 0:
        print("\n  All checks PASSED. Ready for LUMI-G deployment.")
    elif not gpu_found:
        print("\n  GPU checks failed (expected on login node).")
        print("  Run on a compute node with: srun -n1 --gpus=1 python scripts/validate_rocm_rns.py")
    else:
        print("\n  Some checks FAILED. Review output above.")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

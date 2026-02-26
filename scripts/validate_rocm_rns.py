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

import subprocess
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
    # 1. GPU Detection (supports both AMD ROCm and NVIDIA CUDA)
    # ---------------------------------------------------------------
    section("1. GPU Detection")

    gpu_found = False
    gpu_backend = "none"

    # --- NVIDIA path: try nvidia-smi first ---
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            gpu_found = True
            gpu_backend = "cuda"
            gpu_detail = f"nvidia-smi: {r.stdout.strip().splitlines()[0]}"
            print(f"  nvidia-smi: {r.stdout.strip()}")
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # --- AMD path: try rocm-smi ---
    if not gpu_found:
        rocr_dev = os.environ.get("ROCR_VISIBLE_DEVICES", "not set")
        print(f"  ROCR_VISIBLE_DEVICES = {rocr_dev}")
        try:
            r = subprocess.run(["rocm-smi", "--showid"], capture_output=True,
                               text=True, timeout=10)
            if r.returncode == 0 and "GPU" in r.stdout:
                gpu_found = True
                gpu_backend = "rocm"
                gpu_detail = "rocm-smi detected GPU"
                print(f"  rocm-smi output (first 5 lines):")
                for line in r.stdout.strip().splitlines()[:5]:
                    print(f"    {line}")
            else:
                gpu_detail = f"rocm-smi returned code {r.returncode}"
        except Exception as e:
            gpu_detail = f"rocm-smi not available: {e}"

    results.append(check("GPU detected", gpu_found,
                         f"backend={gpu_backend}, {gpu_detail}" if gpu_found
                         else "no GPU found via nvidia-smi or rocm-smi"))

    # HIP Python binding (AMD-only; expected to be absent on NVIDIA)
    if gpu_backend != "cuda":
        hip_ok = False
        try:
            from hip import hip as hiprt
            count = hiprt.hipGetDeviceCount()
            hip_ok = count > 0
            results.append(check("HIP Python binding", hip_ok,
                                 f"{count} device(s)"))
        except Exception as e:
            results.append(check("HIP Python binding", False, str(e)))
    else:
        print("  [SKIP] HIP Python binding -- not needed on NVIDIA/CUDA")

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
    # 6. PCF Compile + Walk (correct conventions)
    # ---------------------------------------------------------------
    section("6. PCF Compile + Walk Smoke Test (v0.2.0)")

    try:
        from dreams_rocm.cmf_compile import compile_pcf_from_strings, pcf_initial_values
        from dreams_rocm.runner import run_pcf_walk, verify_pcf
        import math

        # Compile PCF(2, n**2) — famous Brouncker CF for pi
        program = compile_pcf_from_strings("2", "n**2")
        results.append(check("Compile PCF(2, n**2)",
                             program is not None and len(program.instructions) > 0,
                             f"{len(program.instructions)} instructions"))

        # Check initial values
        a0 = pcf_initial_values("2")
        results.append(check("pcf_initial_values('2') == 2", a0 == 2))

        # Small RNS walk (depth=200, K=8)
        t0 = time.time()
        res = run_pcf_walk(program, a0, depth=200, K=8)
        dt = time.time() - t0

        p_f = res['p_float']
        q_f = res['q_float']
        est = p_f / q_f if abs(q_f) > 1e-300 else float('nan')
        target = 2.0 / (4.0 - 3.141592653589793)  # 2/(4-pi)
        close = abs(est - target) < 0.1

        results.append(check("RNS walk (depth=200, K=8)",
                             close,
                             f"est={est:.6f}, target={target:.6f}, dt={dt:.3f}s"))

        # Full verify_pcf
        t0 = time.time()
        vr = verify_pcf("2", "n**2", "2/(4 - pi)", depth=500, K=16, dps=100)
        dt = time.time() - t0

        if vr is not None:
            d_exact = vr['delta_exact']
            results.append(check("verify_pcf exact delta",
                                 d_exact is not None and math.isfinite(d_exact),
                                 f"delta_exact={d_exact:.6f}, dt={dt:.3f}s"))

            limit_match = abs(vr['est_float'] - vr['target']) < 0.01
            results.append(check("verify_pcf limit match", limit_match,
                                 f"est={vr['est_float']:.6f}, target={vr['target']:.6f}"))
        else:
            results.append(check("verify_pcf", False, "returned None"))

    except Exception as e:
        results.append(check("PCF smoke test", False, str(e)))
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

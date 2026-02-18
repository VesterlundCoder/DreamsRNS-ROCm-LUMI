#!/usr/bin/env python3
"""
Euler2AI PCF Verification — correct RNS pipeline for LUMI.

Loads PCFs from cmf_pcfs.json (JSONL), expands per-source tasks (5192 total),
runs the correct companion-matrix walk with auto-K and CRT overflow detection,
and verifies against stated limits.

Usage:
    python scripts/euler2ai_verify.py --input cmf_pcfs.json --depth 2000 --max-tasks 50
    python scripts/euler2ai_verify.py --input cmf_pcfs.json --depth 2000

On LUMI (inside Singularity container):
    singularity exec $CONTAINER python scripts/euler2ai_verify.py --input $DATA/cmf_pcfs.json
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.runner import (
    run_pcf_walk, crt_reconstruct, centered,
    compute_dreams_delta_exact, compute_dreams_delta_float,
)
from dreams_rocm.cmf_compile import compile_pcf_from_strings, pcf_initial_values


# ── Auto-K estimation ────────────────────────────────────────────────────

def estimate_K(a_str: str, b_str: str, depth: int, safety: float = 1.5) -> int:
    """Estimate K primes needed for exact CRT reconstruction."""
    try:
        n = sp.Symbol('n')
        a_expr = sp.sympify(a_str)
        b_expr = sp.sympify(b_str)
        deg_a = sp.degree(a_expr, n) if a_expr.has(n) else 0
        deg_b = sp.degree(b_expr, n) if b_expr.has(n) else 0
        max_deg = max(int(deg_a), int(deg_b), 1)

        max_coeff = 1
        for expr in [a_expr, b_expr]:
            try:
                poly = sp.Poly(expr, n)
                for c in poly.all_coeffs():
                    max_coeff = max(max_coeff, abs(int(c)))
            except Exception:
                max_coeff = max(max_coeff, 100)

        if depth > 1:
            log2_fact = depth * math.log2(depth / math.e) + 0.5 * math.log2(2 * math.pi * depth)
        else:
            log2_fact = 1

        coeff_bits = math.log2(max_coeff + 1) if max_coeff > 1 else 0
        bits_needed = max_deg * log2_fact + depth * coeff_bits
        bits_needed = int(bits_needed * safety) + 100
        return max(bits_needed // 31 + 1, 64)
    except Exception:
        log2_fact = depth * math.log2(max(depth, 2) / math.e)
        return max(int(4 * log2_fact * safety / 31) + 1, 1000)


# ── CRT overflow detection ───────────────────────────────────────────────

def crt_overflowed(p_big, q_big, est_float):
    """Detect CRT overflow by comparing CRT ratio with float shadow."""
    if q_big == 0:
        return True
    try:
        crt_ratio = float(mp.mpf(p_big) / mp.mpf(q_big))
        if not math.isfinite(crt_ratio) or not math.isfinite(est_float):
            return True
        if abs(est_float) > 1e-10:
            return abs(crt_ratio - est_float) / abs(est_float) > 0.01
        else:
            return abs(crt_ratio - est_float) > 0.01
    except Exception:
        return True


# ── Load and expand sources ──────────────────────────────────────────────

def load_and_expand(path: str):
    """Load cmf_pcfs.json, expand per-source → ~5192 tasks."""
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            items = json.load(f)
        else:
            items = [json.loads(l) for l in f if l.strip()]

    tasks = []
    for pi, rec in enumerate(items):
        if 'a' not in rec or 'b' not in rec:
            continue
        sources = rec.get('sources', [None])
        if not sources:
            sources = [None]
        for si, src in enumerate(sources):
            tasks.append({
                'pcf_idx': pi, 'source_idx': si,
                'a': str(rec['a']), 'b': str(rec['b']),
                'limit': rec.get('limit', ''),
                'delta_ref': rec.get('delta', None),
                'conv_rate': rec.get('convergence_rate', None),
                'trajectory': src[0] if src and isinstance(src, list) else None,
                'shift': src[1] if src and isinstance(src, list) and len(src) > 1 else None,
            })
    return tasks


# ── Parse limit ──────────────────────────────────────────────────────────

def parse_limit(limit_str, dps=200):
    mp.mp.dps = dps
    try:
        target_expr = sp.sympify(limit_str, locals={
            "pi": sp.pi, "E": sp.E, "EulerGamma": sp.EulerGamma,
            "I": sp.I, "sqrt": sp.sqrt, "log": sp.log,
            "Catalan": sp.Catalan, "GoldenRatio": sp.GoldenRatio,
        })
        if target_expr.has(sp.I):
            return None
        return mp.mpf(str(sp.N(target_expr, dps)))
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Euler2AI PCF Verification (ROCm)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to cmf_pcfs.json or pcfs.json")
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--K", type=int, default=0,
                        help="Number of RNS primes (0=auto per PCF)")
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all")
    parser.add_argument("--output", type=str, default="verify_report.csv")
    parser.add_argument("--dps", type=int, default=200)
    parser.add_argument("--delta-tol", type=float, default=0.05)
    args = parser.parse_args()

    mp.mp.dps = args.dps

    tasks = load_and_expand(args.input)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    n_unique = len(set((t['a'], t['b']) for t in tasks))
    print(f"Loaded {len(tasks)} tasks ({n_unique} unique PCFs)")
    print(f"Depth={args.depth}, K={'auto' if args.K == 0 else args.K}")
    print(f"Output: {args.output}")
    print(f"{'='*80}")

    fieldnames = [
        'task_idx', 'pcf_idx', 'source_idx', 'a', 'b', 'limit',
        'trajectory', 'shift',
        'delta_ref', 'delta_rns', 'delta_method', 'est_float',
        'delta_diff', 'match', 'p_bits', 'K',
    ]

    n_match = 0
    n_total = 0
    n_skip = 0
    n_error = 0
    walk_cache = {}
    t_global = time.time()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ti, task in enumerate(tasks):
            a_str, b_str = task['a'], task['b']
            limit_str = task['limit']
            delta_ref = task['delta_ref']
            pcf_key = (a_str, b_str)
            traj_str = str(task['trajectory'] or '')
            shift_str = str(task['shift'] or '')

            # Use cached walk result
            if pcf_key in walk_cache:
                cached = walk_cache[pcf_key]
                if cached is None:
                    n_skip += 1
                    writer.writerow({
                        'task_idx': ti, 'pcf_idx': task['pcf_idx'],
                        'source_idx': task['source_idx'],
                        'a': a_str, 'b': b_str, 'limit': limit_str,
                        'trajectory': traj_str, 'shift': shift_str,
                        'delta_ref': delta_ref, 'delta_rns': 'SKIP',
                        'delta_method': '', 'est_float': '',
                        'delta_diff': '', 'match': 'SKIP',
                        'p_bits': '', 'K': '',
                    })
                    continue
            else:
                # Compile + walk
                try:
                    program = compile_pcf_from_strings(a_str, b_str)
                    if program is None:
                        walk_cache[pcf_key] = None
                        n_skip += 1
                        continue
                    a0 = pcf_initial_values(a_str)
                except Exception as e:
                    walk_cache[pcf_key] = None
                    n_error += 1
                    print(f"  [{ti+1:>5}/{len(tasks)}] COMPILE ERROR: {e}")
                    continue

                try:
                    K_use = args.K if args.K > 0 else estimate_K(a_str, b_str, args.depth)
                    res = run_pcf_walk(program, a0, args.depth, K_use)

                    primes = [int(p) for p in res['primes']]
                    p_big, Mp = crt_reconstruct(
                        [int(r) for r in res['p_residues']], primes)
                    q_big, _ = crt_reconstruct(
                        [int(r) for r in res['q_residues']], primes)
                    p_big = centered(p_big, Mp)
                    q_big = centered(q_big, Mp)

                    est = (res['p_float'] / res['q_float']
                           if abs(res['q_float']) > 1e-300 else float('nan'))

                    target_mp = parse_limit(limit_str, args.dps) if limit_str else None
                    overflow = crt_overflowed(p_big, q_big, est)

                    if target_mp is not None:
                        if overflow:
                            delta_rns, _ = compute_dreams_delta_float(
                                res['p_float'], res['q_float'],
                                res.get('log_scale', 0.0), float(target_mp))
                            delta_method = 'float'
                        else:
                            delta_rns = compute_dreams_delta_exact(
                                p_big, q_big, target_mp, args.dps)
                            delta_method = 'crt'
                    else:
                        delta_rns = float('nan')
                        delta_method = 'none'

                    # Adaptive tolerance
                    if delta_ref is not None and math.isfinite(delta_rns):
                        diff = abs(delta_rns - delta_ref)
                        tol = args.delta_tol
                        if delta_ref < -0.5:
                            tol = max(tol, 1.0)
                        elif delta_ref < 0:
                            tol = max(tol, 0.2)
                        is_match = diff < tol
                    else:
                        diff = float('nan')
                        is_match = False

                    walk_cache[pcf_key] = {
                        'est': est, 'delta_rns': delta_rns,
                        'delta_method': delta_method, 'p_bits': Mp.bit_length(),
                        'K_use': K_use, 'diff': diff, 'is_match': is_match,
                    }
                except Exception as e:
                    walk_cache[pcf_key] = None
                    n_error += 1
                    print(f"  [{ti+1:>5}/{len(tasks)}] WALK ERROR: {e}")
                    continue

            wc = walk_cache[pcf_key]
            n_total += 1
            if wc['is_match']:
                n_match += 1

            writer.writerow({
                'task_idx': ti,
                'pcf_idx': task['pcf_idx'],
                'source_idx': task['source_idx'],
                'a': a_str, 'b': b_str, 'limit': limit_str,
                'trajectory': traj_str, 'shift': shift_str,
                'delta_ref': f"{delta_ref:.5f}" if delta_ref is not None else '',
                'delta_rns': f"{wc['delta_rns']:.5f}" if math.isfinite(wc['delta_rns']) else str(wc['delta_rns']),
                'delta_method': wc['delta_method'],
                'est_float': f"{wc['est']:.10f}" if math.isfinite(wc['est']) else str(wc['est']),
                'delta_diff': f"{wc['diff']:.6f}" if math.isfinite(wc['diff']) else '',
                'match': 'YES' if wc['is_match'] else 'NO',
                'p_bits': wc['p_bits'],
                'K': wc['K_use'],
            })

            # Progress — print for first source of each PCF
            if pcf_key not in {(t['a'], t['b']) for t in tasks[:ti]}:
                elapsed = time.time() - t_global
                status = "MATCH" if wc['is_match'] else "MISS"
                n_src = sum(1 for t in tasks if (t['a'], t['b']) == pcf_key)
                d_show = f"{wc['delta_rns']:.5f}" if math.isfinite(wc['delta_rns']) else str(wc['delta_rns'])
                ref_show = f"{delta_ref:.5f}" if delta_ref is not None else "N/A"
                print(f"  [{ti+1:>5}/{len(tasks)}] {status} "
                      f"δ={d_show:>10} ref={ref_show:>10} "
                      f"[{wc['delta_method']}] K={wc['K_use']} "
                      f"{n_src}src a={a_str[:40]}")

    elapsed_total = time.time() - t_global
    print(f"\n{'='*80}")
    print(f"VERIFICATION COMPLETE")
    print(f"  Tasks:    {len(tasks)} ({n_unique} unique PCFs)")
    print(f"  Verified: {n_total}")
    print(f"  Matches:  {n_match}/{n_total} ({100*n_match/max(n_total,1):.1f}%)")
    print(f"  Skipped:  {n_skip}, Errors: {n_error}")
    print(f"  Time:     {elapsed_total:.1f}s")
    print(f"  CSV:      {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

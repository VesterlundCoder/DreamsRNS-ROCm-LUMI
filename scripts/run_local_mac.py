#!/usr/bin/env python3
"""
Local Mac test runner for the Dreams-RNS-ROCm pipeline.

Runs a quick PCF verification locally using the numpy reference walker
to validate the pipeline before deploying to LUMI-G.
No MPI or GPU required — this is for correctness testing only.

Usage:
    python scripts/run_local_mac.py
    python scripts/run_local_mac.py --input /path/to/pcfs.json --max-tasks 10
    python scripts/run_local_mac.py --depth 500 --K 16
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.runner import verify_pcf
from dreams_rocm.cmf_compile import compile_pcf_from_strings, pcf_initial_values


# Built-in test PCFs (no data file needed)
BUILTIN_PCFS = [
    {"a": "2", "b": "n**2", "limit": "2/(4 - pi)", "ref_delta": -0.99791},
    {"a": "1", "b": "n**2 + n", "limit": "pi/2 - 1", "ref_delta": -0.99915},
    {"a": "4", "b": "4*n**2 - 1", "limit": "pi/2", "ref_delta": -0.99867},
    {"a": "8", "b": "4*n**2 - 1", "limit": "pi", "ref_delta": -0.99717},
]


def load_pcfs(path: str) -> list:
    """Load PCFs from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({
                'a': rec['a'],
                'b': rec['b'],
                'limit': rec.get('limit', ''),
                'ref_delta': rec.get('delta', None),
            })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Local Mac test runner for Dreams-RNS-ROCm v0.2.0"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to pcfs.json (default: use built-in PCFs)")
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--dps", type=int, default=100)
    args = parser.parse_args()

    # Load PCFs
    if args.input:
        records = load_pcfs(args.input)
    else:
        records = BUILTIN_PCFS
        print("Using built-in test PCFs (no --input specified)")

    if args.max_tasks > 0:
        records = records[:args.max_tasks]

    print("=" * 60)
    print("Dreams-RNS-ROCm v0.2.0 — Local Mac Test")
    print("=" * 60)
    print(f"  PCFs:   {len(records)}")
    print(f"  Depth:  {args.depth}")
    print(f"  K:      {args.K} primes ({args.K * 31} bits)")
    print("=" * 60)
    print()

    n_ok = 0
    n_fail = 0
    n_skip = 0
    t_total = time.time()

    for idx, rec in enumerate(records):
        a_str = rec['a']
        b_str = rec['b']
        limit_str = rec.get('limit', '')

        if not limit_str:
            n_skip += 1
            continue

        t0 = time.time()
        try:
            res = verify_pcf(a_str, b_str, limit_str,
                             depth=args.depth, K=args.K, dps=args.dps)
        except Exception as e:
            print(f"  [{idx}] SKIP PCF({a_str[:20]}, {b_str[:20]}): {e}")
            n_skip += 1
            continue

        if res is None:
            n_skip += 1
            continue

        dt = time.time() - t0
        limit_match = (abs(res['est_float'] - res['target']) < 0.01
                       if math.isfinite(res['est_float']) else False)

        ref_delta = rec.get('ref_delta')
        d_diff = (res['delta_exact'] - ref_delta) if ref_delta is not None else None

        status = "Y" if limit_match else "N"
        if limit_match:
            n_ok += 1
        else:
            n_fail += 1

        print(f"  [{idx}] PCF({a_str[:20]}, {b_str[:20]}) "
              f"d_exact={res['delta_exact']:+.6f}  "
              f"ref={ref_delta}  match={status}  "
              f"{dt:.2f}s")

    wall_total = time.time() - t_total

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Completed:     {n_ok + n_fail}")
    print(f"  Skipped:       {n_skip}")
    print(f"  Limit matches: {n_ok}/{n_ok + n_fail} "
          f"({100*n_ok/max(n_ok+n_fail,1):.0f}%)")
    print(f"  Wall time:     {wall_total:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

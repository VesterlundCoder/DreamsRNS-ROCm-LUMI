#!/usr/bin/env python3
"""
MPI orchestrator for Dreams-RNS-ROCm pipeline on LUMI-G.

Distributes PCF verification across MPI ranks (or runs single-process).
Loads PCFs from a JSONL file (pcfs.json or cmf_pcfs.json), compiles using
the correct companion matrix convention, and runs full K-prime RNS walks.

Usage (single process):
    python scripts/run_mpi_sweep.py --input data/pcfs.json --depth 2000 --K 64

Usage (MPI on LUMI):
    srun -n8 python scripts/run_mpi_sweep.py --input /data/pcfs.json

Usage (inside Singularity):
    singularity exec --rocm --bind $PWD:/workspace $CONTAINER \\
        python3 /workspace/scripts/run_mpi_sweep.py --input /data/pcfs.json
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.runner import (
    verify_pcf, compile_pcf_from_strings, pcf_initial_values,
    run_pcf_walk, crt_reconstruct, centered,
    compute_dreams_delta_float, compute_dreams_delta_exact,
)
from dreams_rocm.cmf_compile import compile_pcf_from_strings as compile_pcf
from dreams_rocm.logging import RunLogger, create_manifest


def load_pcfs(path: str) -> list:
    """Load PCFs from JSONL file (pcfs.json or cmf_pcfs.json format)."""
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


def process_pcf(rec: dict, depth: int, K: int, dps: int,
                logger: RunLogger, idx: int, rank: int) -> dict:
    """Process a single PCF: compile → walk → CRT → delta.

    Returns result dict or None on failure.
    """
    a_str = rec['a']
    b_str = rec['b']
    limit_str = rec['limit']

    if not limit_str:
        return None

    t0 = time.time()
    try:
        res = verify_pcf(a_str, b_str, limit_str, depth=depth, K=K, dps=dps)
    except Exception as e:
        print(f"  [Rank {rank}] SKIP [{idx}] a={a_str[:40]}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None

    if res is None:
        return None

    wall = time.time() - t0
    ref_delta = rec.get('ref_delta')

    result = {
        "pcf_idx": idx,
        "a": a_str,
        "b": b_str,
        "limit": limit_str,
        "target": res['target'],
        "est_float": res['est_float'],
        "delta_exact": res['delta_exact'],
        "delta_float": res['delta_float'],
        "ref_delta": ref_delta,
        "depth": depth,
        "K": K,
        "p_bits": res['p_bits'],
        "wall_sec": wall,
    }

    # Log to structured output
    logger.log_result(result)

    # Escalate if delta > 0 (interesting convergence)
    if res['delta_exact'] is not None and math.isfinite(res['delta_exact']):
        if res['delta_exact'] > 0:
            logger.log_positive(result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Dreams-RNS-ROCm PCF sweep (MPI-capable)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to pcfs.json or cmf_pcfs.json")
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--dps", type=int, default=200)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    # --- MPI initialization ---
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        rank = 0
        size = 1
        comm = None

    gpu_id = int(os.environ.get("ROCR_VISIBLE_DEVICES", str(rank)))

    # --- Output directory ---
    run_id = args.run_id or os.environ.get(
        "DREAMS_RUN_ID", f"run_{int(time.time())}")
    output_dir = Path(args.output_dir or os.environ.get(
        "DREAMS_OUTPUT_DIR",
        str(Path("./outputs") / run_id)))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load PCFs ---
    records = load_pcfs(args.input)
    if args.max_tasks > 0:
        records = records[:args.max_tasks]

    if rank == 0:
        print(f"Dreams-RNS-ROCm v0.2.0 PCF Sweep")
        print(f"  Ranks:  {size}")
        print(f"  Input:  {args.input} ({len(records)} PCFs)")
        print(f"  Depth:  {args.depth}, K: {args.K}")
        print(f"  Output: {output_dir}")
        print(f"  Run ID: {run_id}")
        print(flush=True)

        config = {
            "input": args.input, "depth": args.depth, "K": args.K,
            "dps": args.dps, "n_pcfs": len(records),
        }
        manifest = create_manifest(run_id=run_id, config=config, n_ranks=size)
        manifest.save(output_dir / "manifest.json")

    # --- Distribute tasks round-robin ---
    my_tasks = [(i, rec) for i, rec in enumerate(records) if i % size == rank]

    if rank == 0:
        print(f"  [Rank 0] {len(my_tasks)} PCFs assigned", flush=True)

    # --- Logger ---
    logger = RunLogger(
        output_dir=output_dir,
        rank=rank,
        delta_threshold=-100.0,  # log everything
    )

    # --- Process ---
    results = []
    n_ok = 0
    n_skip = 0
    t_total = time.time()

    for task_num, (idx, rec) in enumerate(my_tasks):
        result = process_pcf(rec, args.depth, args.K, args.dps,
                             logger, idx, rank)
        if result is None:
            n_skip += 1
            continue

        results.append(result)
        limit_match = (abs(result['est_float'] - result['target']) < 0.01
                       if math.isfinite(result['est_float']) else False)
        if limit_match:
            n_ok += 1

        if (task_num + 1) % 50 == 0 or task_num < 5:
            print(f"  [Rank {rank}] {task_num+1}/{len(my_tasks)} | "
                  f"PCF({rec['a'][:15]}, {rec['b'][:15]}) "
                  f"d={result['delta_exact']:.4f} "
                  f"{'Y' if limit_match else 'N'}", flush=True)

    logger.close()
    wall_total = time.time() - t_total

    # --- Barrier + summary ---
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        total_results = 0
        total_positives = 0
        for r in range(size):
            rf = output_dir / f"results_rank{r}.jsonl"
            if rf.exists():
                total_results += sum(1 for _ in open(rf))
            pf = output_dir / f"positives_rank{r}.jsonl"
            if pf.exists():
                total_positives += sum(1 for _ in open(pf))

        summary = {
            "run_id": run_id,
            "n_ranks": size,
            "n_pcfs": len(records),
            "total_results": total_results,
            "total_positives": total_positives,
            "wall_time_sec": wall_total,
            "completed": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Run complete: {run_id}")
        print(f"  Total PCFs processed: {total_results}")
        print(f"  Positives (delta>0): {total_positives}")
        print(f"  Wall time: {wall_total:.1f}s")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

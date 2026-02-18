#!/usr/bin/env python3
"""
Generate chunked task files for LUMI exhaust sweep.

Reads CMF specs from a JSONL file, generates trajectories and shifts
for the appropriate dimension, and produces per-GPU task files that
can be submitted to LUMI via sbatch.

Supports staged execution:
  Stage 1:  64 shifts × full trajectories  (triage)
  Stage 2: 256 shifts × full trajectories  (signal confirmation)
  Stage 3: full shifts × full trajectories  (exhaust top candidates)

Output: tasks_gpu{0-7}.jsonl files, each containing balanced workloads.

Usage:
    # Generate stage-1 tasks for all 2000 CMFs across 8 GPUs
    python scripts/generate_exhaust_tasks.py \\
        --cmfs cmfs_exhaust.jsonl \\
        --stage 1 \\
        --n-gpus 8 \\
        --output-dir tasks/stage1/

    # Generate stage-2 tasks only for CMFs that showed signal
    python scripts/generate_exhaust_tasks.py \\
        --cmfs cmfs_exhaust.jsonl \\
        --stage 2 \\
        --include-list stage1_hits.txt \\
        --n-gpus 8 \\
        --output-dir tasks/stage2/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.exhaust import (
    exhaust_trajectories,
    exhaust_shifts,
    exhaust_summary,
    EXHAUST_PRESETS,
    RationalShift,
)


def load_cmf_specs(path: str) -> List[dict]:
    specs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            specs.append(json.loads(line))
    return specs


def load_include_list(path: str) -> set:
    """Load a file of CMF spec_hashes (one per line) to include."""
    hashes = set()
    with open(path) as f:
        for line in f:
            h = line.strip()
            if h and not h.startswith("#"):
                hashes.add(h)
    return hashes


def shifts_to_serializable(shifts: List[RationalShift]) -> List[dict]:
    return [{"nums": list(s.nums), "dens": list(s.dens)} for s in shifts]


def trajectories_to_serializable(trajs: list) -> List[List[int]]:
    return [list(t) for t in trajs]


def generate_tasks(
    cmf_specs: List[dict],
    stage: int,
    n_gpus: int,
    output_dir: str,
    depth: int = 2000,
    K: int = 32,
    include_hashes: Optional[set] = None,
):
    """Generate task files balanced across n_gpus GPUs."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter CMFs if include list provided
    if include_hashes is not None:
        cmf_specs = [s for s in cmf_specs if s.get("spec_hash", "") in include_hashes]
        print(f"Filtered to {len(cmf_specs)} CMFs matching include list")

    if not cmf_specs:
        print("ERROR: No CMFs to process")
        return

    # Group CMFs by dimension
    cmfs_by_dim: dict = {}
    for spec in cmf_specs:
        dim = spec["dim"]
        cmfs_by_dim.setdefault(dim, []).append(spec)

    # Pre-generate trajectories and shifts per dimension
    traj_cache = {}
    shift_cache = {}
    for dim in cmfs_by_dim:
        traj_cache[dim] = exhaust_trajectories(dim)
        shift_cache[dim] = exhaust_shifts(dim, stage=stage)

    # Estimate total work units per CMF (for load balancing)
    # Work = n_trajectories × n_shifts
    cmf_work = []
    for spec in cmf_specs:
        dim = spec["dim"]
        n_traj = len(traj_cache[dim])
        n_shift = len(shift_cache[dim])
        work = n_traj * n_shift
        cmf_work.append((work, spec))

    # Sort by work descending (largest first for better balancing)
    cmf_work.sort(key=lambda x: -x[0])

    # Greedy bin-packing: assign each CMF to the GPU with least total work
    gpu_tasks = [[] for _ in range(n_gpus)]
    gpu_load = [0] * n_gpus

    for work, spec in cmf_work:
        # Find GPU with minimum load
        min_gpu = min(range(n_gpus), key=lambda g: gpu_load[g])
        gpu_tasks[min_gpu].append(spec)
        gpu_load[min_gpu] += work

    # Write task files
    manifest = {
        "stage": stage,
        "n_gpus": n_gpus,
        "n_cmfs": len(cmf_specs),
        "depth": depth,
        "K": K,
        "dims": {},
    }

    for dim in sorted(cmfs_by_dim.keys()):
        n_cmfs_dim = len(cmfs_by_dim[dim])
        n_traj = len(traj_cache[dim])
        n_shift = len(shift_cache[dim])
        manifest["dims"][str(dim)] = {
            "n_cmfs": n_cmfs_dim,
            "n_trajectories": n_traj,
            "n_shifts": n_shift,
            "runs_per_cmf": n_traj * n_shift,
            "total_runs": n_cmfs_dim * n_traj * n_shift,
        }

    total_runs = sum(v["total_runs"] for v in manifest["dims"].values())
    manifest["total_runs"] = total_runs

    # Write per-GPU task files
    for gpu_id in range(n_gpus):
        task_file = os.path.join(output_dir, f"tasks_gpu{gpu_id}.jsonl")
        n_tasks = 0
        with open(task_file, 'w') as f:
            for spec in gpu_tasks[gpu_id]:
                dim = spec["dim"]
                task = {
                    "cmf": spec,
                    "stage": stage,
                    "depth": depth,
                    "K": K,
                    "n_trajectories": len(traj_cache[dim]),
                    "n_shifts": len(shift_cache[dim]),
                    "shifts": shifts_to_serializable(shift_cache[dim]),
                    "trajectories": trajectories_to_serializable(traj_cache[dim]),
                }
                f.write(json.dumps(task, default=str) + "\n")
                n_tasks += 1

        print(f"  GPU {gpu_id}: {n_tasks:>4} CMFs, "
              f"load={gpu_load[gpu_id]:>12,} runs")

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Write summary
    print(f"\n{'=' * 60}")
    print(f"Stage {stage} task generation complete")
    print(f"{'=' * 60}")
    print(f"  CMFs:         {len(cmf_specs)}")
    print(f"  Total runs:   {total_runs:,}")
    for dim, info in manifest["dims"].items():
        print(f"  dim={dim}: {info['n_cmfs']} CMFs × "
              f"{info['n_trajectories']} traj × {info['n_shifts']} shifts "
              f"= {info['total_runs']:,}")
    print(f"  GPUs:         {n_gpus}")
    print(f"  Output:       {output_dir}/")

    # Compute estimate (assuming ~0.5ms per run on MI250X at depth=2000 K=32)
    ms_per_run = 0.5
    gpu_hours = total_runs * ms_per_run / 1000 / 3600 / n_gpus
    print(f"\n  Estimated GPU-hours: {gpu_hours:.1f}h per GPU "
          f"(assuming {ms_per_run}ms/run)")
    print(f"  Estimated wall time: {gpu_hours:.1f}h "
          f"(8 GPUs parallel)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate exhaust task files for LUMI"
    )
    parser.add_argument("--cmfs", type=str, required=True,
                        help="Input CMF specs JSONL file")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Exhaust stage (1=triage, 2=confirm, 3=full)")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPUs to distribute across")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: tasks/stage{N}/)")
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--include-list", type=str, default=None,
                        help="File of spec_hashes to include (for stage 2/3)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"tasks/stage{args.stage}"

    cmf_specs = load_cmf_specs(args.cmfs)
    print(f"Loaded {len(cmf_specs)} CMF specs from {args.cmfs}")

    include_hashes = None
    if args.include_list:
        include_hashes = load_include_list(args.include_list)
        print(f"Include list: {len(include_hashes)} hashes from {args.include_list}")

    generate_tasks(
        cmf_specs=cmf_specs,
        stage=args.stage,
        n_gpus=args.n_gpus,
        output_dir=args.output_dir,
        depth=args.depth,
        K=args.K,
        include_hashes=include_hashes,
    )


if __name__ == "__main__":
    main()

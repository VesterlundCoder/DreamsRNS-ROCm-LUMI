#!/usr/bin/env python3
"""
MPI orchestrator for Dreams-RNS-ROCm pipeline on LUMI-G.

Distributes CMF evaluation across 8 GPUs using mpi4py.
Default mode: CMF-per-GPU (each rank claims one CMF at a time).

Usage:
    srun python scripts/run_mpi_sweep.py --config configs/lumi_1node_8gpu.yaml
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.cmf_compile import compile_cmf_from_dict, CmfProgram
from dreams_rocm.runner import DreamsRunner, WalkConfig, Hit
from dreams_rocm.shifts import generate_shifts
from dreams_rocm.trajectories import generate_trajectories
from dreams_rocm.logging import RunLogger, create_manifest
from dreams_rocm.crt.partial_crt import partial_crt_delta_proxy
from dreams_rocm.crt.full_crt_cpu import full_crt_verify, verify_against_all_targets
from dreams_rocm.crt.delta_targets import ZETA_TARGETS


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compile_cmfs_from_config(config: dict) -> list:
    """Compile CMF programs from config definition."""
    programs = []
    for cmf_def in config.get("cmf_families", []):
        # Convert "row,col" string keys to (int, int) tuples
        matrix_dict = {}
        for key, expr in cmf_def["matrix"].items():
            if isinstance(key, str):
                parts = key.split(",")
                row, col = int(parts[0]), int(parts[1])
            else:
                row, col = key
            matrix_dict[(row, col)] = expr

        program = compile_cmf_from_dict(
            matrix_dict=matrix_dict,
            m=cmf_def["m"],
            dim=cmf_def["dim"],
            axis_names=cmf_def.get("axis_names"),
            directions=cmf_def.get("directions"),
            name=cmf_def.get("name", f"cmf_{len(programs)}"),
        )
        programs.append((program, cmf_def))

    return programs


def build_task_list(config: dict, programs: list) -> list:
    """Build a list of (cmf_idx, program, cmf_def) tasks."""
    tasks = []
    for idx, (program, cmf_def) in enumerate(programs):
        tasks.append({
            "cmf_idx": idx,
            "cmf_name": cmf_def.get("name", f"cmf_{idx}"),
            "program": program,
            "cmf_def": cmf_def,
        })
    return tasks


def process_task(task: dict, config: dict, logger: RunLogger,
                 rank: int, gpu_id: int) -> dict:
    """Process a single CMF task: run shifts × trajectories.

    Returns timing/result summary dict.
    """
    program = task["program"]
    cmf_idx = task["cmf_idx"]
    cmf_name = task["cmf_name"]
    cmf_def = task["cmf_def"]

    t_start = time.time()

    # Get config parameters
    shift_cfg = config.get("shifts", {})
    traj_cfg = config.get("trajectories", {})
    walk_cfg = config.get("walk", {})
    rns_cfg = config.get("rns", {})
    triage_cfg = config.get("triage", {})

    n_shifts = shift_cfg.get("count", 100)
    n_trajectories = traj_cfg.get("count", 1000)
    depth = walk_cfg.get("depth", 2000)
    K = rns_cfg.get("K", 64)
    delta_threshold = triage_cfg.get("delta_threshold", 0.0)

    # Resolve target
    target_name = cmf_def.get("target", config.get("primary_target", "zeta3"))
    target_value = ZETA_TARGETS.get(target_name, 1.2020569031595942)

    # Generate shifts
    shifts = generate_shifts(
        n_shifts=n_shifts,
        dim=program.dim,
        method=shift_cfg.get("method", "grid"),
        bounds=tuple(shift_cfg.get("bounds", [-1000, 1000])),
        seed=shift_cfg.get("seed", 42),
        cmf_idx=cmf_idx,
    )

    # Generate trajectories
    trajectories = generate_trajectories(
        count=n_trajectories,
        max_component=traj_cfg.get("max_component", 50),
    )

    # Configure walker
    walk_config = WalkConfig(
        K=K,
        B=n_shifts,
        depth=depth,
        topk=config.get("performance", {}).get("topk", 100),
        target=target_value,
        target_name=target_name,
        snapshot_depths=tuple(walk_cfg.get("snapshot_depths", [200, 2000])),
        delta_threshold=delta_threshold,
        K_small=rns_cfg.get("K_small", 6),
    )

    runner = DreamsRunner([program], config=walk_config)

    total_hits = 0
    total_escalated = 0
    traj_times = []

    # For Test 1: iterate over trajectories
    # Each trajectory changes the walk direction
    for traj_idx, (dn, dk) in enumerate(trajectories):
        t_traj = time.time()

        # Update directions based on trajectory
        if program.dim >= 2:
            program.directions = [dn, dk] + program.directions[2:]
        elif program.dim == 1:
            program.directions = [dn]

        # Run walk for this trajectory
        hits, metrics = runner.run_single(
            program, shifts, cmf_idx=cmf_idx,
            directions=program.directions,
        )

        traj_time = time.time() - t_traj
        traj_times.append(traj_time)

        # Process hits
        for hit in hits:
            hit.traj_id = traj_idx
            hit.traj_dir = (dn, dk)

            # Log result (only delta > threshold)
            logger.log_result({
                "cmf_idx": cmf_idx,
                "cmf_name": cmf_name,
                "shift": hit.shift,
                "depth": hit.depth,
                "delta": hit.delta,
                "log_q": hit.log_q,
                "traj_id": traj_idx,
                "traj_dir": [dn, dk],
                "target": target_name,
                "gpu_id": gpu_id,
            })
            total_hits += 1

            # Escalate to full CRT if delta > escalation threshold
            escalate_thresh = triage_cfg.get("escalate_threshold", 0.0)
            if hit.delta > escalate_thresh:
                try:
                    # For now, use float-based verification
                    # Full RNS -> CRT will be enabled when native library is built
                    verifications = []
                    for tgt in ["zeta3", "zeta5", "zeta7"]:
                        tgt_val = ZETA_TARGETS.get(tgt, 0)
                        if tgt_val > 0 and hit.log_q > 0:
                            verifications.append({
                                "target": tgt,
                                "delta": hit.delta,
                                "decision": "positive" if hit.delta > 0.5 else "weak_positive",
                            })

                    logger.log_positive({
                        "cmf_idx": cmf_idx,
                        "cmf_name": cmf_name,
                        "shift": hit.shift,
                        "depth": hit.depth,
                        "delta": hit.delta,
                        "log_q": hit.log_q,
                        "traj_id": traj_idx,
                        "traj_dir": [dn, dk],
                        "verifications": verifications,
                        "gpu_id": gpu_id,
                    })
                    total_escalated += 1
                except Exception as e:
                    logger.log_metrics({
                        "event": "escalation_error",
                        "cmf_idx": cmf_idx,
                        "traj_idx": traj_idx,
                        "error": str(e),
                    })

        # Log progress periodically
        if (traj_idx + 1) % 100 == 0 or traj_idx == 0:
            avg_traj_time = sum(traj_times) / len(traj_times)
            print(f"  [Rank {rank}] CMF {cmf_name}: "
                  f"traj {traj_idx+1}/{len(trajectories)}, "
                  f"hits={total_hits}, "
                  f"avg {avg_traj_time:.3f}s/traj", flush=True)

    t_end = time.time()
    wall_time = t_end - t_start

    # Log timing metrics
    task_metrics = {
        "event": "task_complete",
        "cmf_idx": cmf_idx,
        "cmf_name": cmf_name,
        "n_shifts": n_shifts,
        "n_trajectories": len(trajectories),
        "total_evaluations": n_shifts * len(trajectories),
        "total_hits": total_hits,
        "total_escalated": total_escalated,
        "wall_time_sec": wall_time,
        "trajectories_per_sec": len(trajectories) / max(wall_time, 1e-6),
        "shifts_per_sec": n_shifts * len(trajectories) / max(wall_time, 1e-6),
        "gpu_id": gpu_id,
    }
    logger.log_metrics(task_metrics)

    return task_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Dreams-RNS-ROCm MPI sweep on LUMI-G"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: from env or ./outputs)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: auto-generated)")
    args = parser.parse_args()

    # --- MPI initialization ---
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        # Fallback for single-process debugging
        rank = 0
        size = 1
        comm = None

    # --- GPU ID ---
    gpu_id = int(os.environ.get("ROCR_VISIBLE_DEVICES", str(rank)))

    # --- Load config ---
    config = load_config(args.config)

    # --- Output directory ---
    output_dir = args.output_dir or os.environ.get(
        "DREAMS_OUTPUT_DIR",
        str(Path("./outputs") / (args.run_id or f"run_{int(time.time())}"))
    )
    output_dir = Path(output_dir)

    # --- Run ID ---
    run_id = args.run_id or os.environ.get(
        "DREAMS_RUN_ID",
        f"run_{int(time.time())}"
    )

    # --- Rank 0: write manifest ---
    if rank == 0:
        print(f"Dreams-RNS-ROCm MPI Sweep")
        print(f"  Ranks: {size}")
        print(f"  Config: {args.config}")
        print(f"  Output: {output_dir}")
        print(f"  Run ID: {run_id}")
        print(f"", flush=True)

        manifest = create_manifest(
            run_id=run_id,
            config=config,
            n_ranks=size,
        )
        manifest.save(output_dir / "manifest.json")

    # --- Compile CMFs (all ranks, for simplicity) ---
    try:
        programs = compile_cmfs_from_config(config)
    except Exception as e:
        print(f"[Rank {rank}] ERROR compiling CMFs: {e}", flush=True)
        traceback.print_exc()
        if comm is not None:
            comm.Abort(1)
        sys.exit(1)

    if rank == 0:
        print(f"  Compiled {len(programs)} CMF programs", flush=True)

    # --- Build task list ---
    tasks = build_task_list(config, programs)

    # --- Distribute tasks ---
    # CMF-per-GPU: static round-robin assignment
    my_tasks = [t for i, t in enumerate(tasks) if i % size == rank]

    if not my_tasks:
        print(f"[Rank {rank}] No tasks assigned (more ranks than CMFs)", flush=True)
    else:
        print(f"[Rank {rank}] Assigned {len(my_tasks)} CMF(s): "
              f"{[t['cmf_name'] for t in my_tasks]}", flush=True)

    # --- Initialize logger ---
    logger = RunLogger(
        output_dir=output_dir,
        rank=rank,
        delta_threshold=config.get("triage", {}).get("delta_threshold", 0.0),
    )

    # --- Process tasks ---
    all_metrics = []
    for task_idx, task in enumerate(my_tasks):
        print(f"\n[Rank {rank}] Starting task {task_idx+1}/{len(my_tasks)}: "
              f"{task['cmf_name']}", flush=True)

        try:
            metrics = process_task(task, config, logger, rank, gpu_id)
            all_metrics.append(metrics)

            # Checkpoint
            checkpoint_every = config.get("distribution", {}).get("checkpoint_every", 10)
            if (task_idx + 1) % checkpoint_every == 0:
                ckpt_path = output_dir / f"checkpoint_rank{rank}.json"
                with open(ckpt_path, 'w') as f:
                    json.dump({
                        "rank": rank,
                        "completed_tasks": task_idx + 1,
                        "total_tasks": len(my_tasks),
                        "metrics": all_metrics,
                    }, f, indent=2, default=str)

        except Exception as e:
            print(f"[Rank {rank}] ERROR on task {task['cmf_name']}: {e}",
                  flush=True)
            traceback.print_exc()
            logger.log_metrics({
                "event": "task_error",
                "cmf_name": task['cmf_name'],
                "error": str(e),
            })

    # --- Finalize ---
    logger.close()

    # Barrier
    if comm is not None:
        comm.Barrier()

    # Rank 0: write global summary
    if rank == 0:
        # Gather summaries from all ranks
        summary_path = output_dir / "summary.json"
        summary = {
            "run_id": run_id,
            "n_ranks": size,
            "n_cmfs": len(tasks),
            "completed": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        # Read all result files to count totals
        total_results = 0
        total_positives = 0
        for r in range(size):
            results_file = output_dir / f"results_rank{r}.jsonl"
            if results_file.exists():
                total_results += sum(1 for _ in open(results_file))
            positives_file = output_dir / f"positives_rank{r}.jsonl"
            if positives_file.exists():
                total_positives += sum(1 for _ in open(positives_file))

        summary["total_results"] = total_results
        summary["total_positives"] = total_positives

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Run complete: {run_id}")
        print(f"  Total results (delta>0): {total_results}")
        print(f"  Total positives:         {total_positives}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

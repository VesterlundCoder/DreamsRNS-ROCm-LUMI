#!/usr/bin/env python3
"""
Local Mac test runner for the Dreams-RNS-ROCm pipeline.

Runs a single-process, CPU-only sweep with reduced parameters
(1 CMF at a time, 10 shifts, 100 trajectories) for quick local testing.

No MPI or GPU required.

Usage:
    python scripts/run_local_mac.py
    python scripts/run_local_mac.py --config configs/mac_local_test.yaml
    python scripts/run_local_mac.py --cmf 0          # run only CMF index 0
    python scripts/run_local_mac.py --list-cmfs       # show available CMFs
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


DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "configs" / "mac_local_test.yaml"
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compile_cmfs_from_config(config: dict) -> list:
    programs = []
    for cmf_def in config.get("cmf_families", []):
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


def process_cmf(program, cmf_def, cmf_idx, config, logger):
    """Process a single CMF: shifts x trajectories walk."""
    cmf_name = cmf_def.get("name", f"cmf_{cmf_idx}")
    t_start = time.time()

    shift_cfg = config.get("shifts", {})
    traj_cfg = config.get("trajectories", {})
    walk_cfg = config.get("walk", {})
    rns_cfg = config.get("rns", {})
    triage_cfg = config.get("triage", {})

    n_shifts = shift_cfg.get("count", 10)
    n_trajectories = traj_cfg.get("count", 100)
    depth = walk_cfg.get("depth", 200)
    K = rns_cfg.get("K", 16)
    delta_threshold = triage_cfg.get("delta_threshold", -10.0)

    target_name = cmf_def.get("target", config.get("primary_target", "zeta3"))
    target_value = ZETA_TARGETS.get(target_name, 1.2020569031595942)

    # Generate shifts
    shifts = generate_shifts(
        n_shifts=n_shifts,
        dim=program.dim,
        method=shift_cfg.get("method", "grid"),
        bounds=tuple(shift_cfg.get("bounds", [-100, 100])),
        seed=shift_cfg.get("seed", 42),
        cmf_idx=cmf_idx,
    )

    # Generate trajectories
    trajectories = generate_trajectories(
        count=n_trajectories,
        max_component=traj_cfg.get("max_component", 20),
    )

    # Configure walker
    walk_config = WalkConfig(
        K=K,
        B=n_shifts,
        depth=depth,
        topk=config.get("performance", {}).get("topk", 10),
        target=target_value,
        target_name=target_name,
        snapshot_depths=tuple(walk_cfg.get("snapshot_depths", [50, 200])),
        delta_threshold=delta_threshold,
        K_small=rns_cfg.get("K_small", 4),
    )

    runner = DreamsRunner([program], config=walk_config)

    total_hits = 0
    total_escalated = 0
    best_delta = float('-inf')
    best_hit = None

    print(f"\n  Walking {n_trajectories} trajectories x {n_shifts} shifts "
          f"(depth={depth}, K={K})...")

    for traj_idx, (dn, dk) in enumerate(trajectories):
        # Update directions based on trajectory
        if program.dim >= 2:
            dirs = [dn, dk] + program.directions[2:]
        elif program.dim == 1:
            dirs = [dn]
        else:
            dirs = program.directions

        # Run walk
        hits, metrics = runner.run_single(
            program, shifts, cmf_idx=cmf_idx,
            directions=dirs,
        )

        for hit in hits:
            hit.traj_id = traj_idx
            hit.traj_dir = (dn, dk)

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
                "gpu_id": -1,
            })
            total_hits += 1

            if hit.delta > best_delta:
                best_delta = hit.delta
                best_hit = hit

            # Escalate
            escalate_thresh = triage_cfg.get("escalate_threshold", 0.0)
            if hit.delta > escalate_thresh:
                logger.log_positive({
                    "cmf_idx": cmf_idx,
                    "cmf_name": cmf_name,
                    "shift": hit.shift,
                    "depth": hit.depth,
                    "delta": hit.delta,
                    "log_q": hit.log_q,
                    "traj_id": traj_idx,
                    "traj_dir": [dn, dk],
                    "verifications": [{"target": target_name,
                                       "delta": hit.delta,
                                       "decision": "positive" if hit.delta > 0.5
                                       else "weak_positive"}],
                    "gpu_id": -1,
                })
                total_escalated += 1

        # Progress every 10 trajectories
        if (traj_idx + 1) % 10 == 0 or traj_idx == len(trajectories) - 1:
            elapsed = time.time() - t_start
            rate = (traj_idx + 1) / max(elapsed, 1e-6)
            print(f"    traj {traj_idx+1:4d}/{len(trajectories)} | "
                  f"hits={total_hits:4d} | "
                  f"best_delta={best_delta:+.4f} | "
                  f"{rate:.1f} traj/s", end='\r', flush=True)

    wall = time.time() - t_start
    print()  # newline after \r progress
    print(f"  Done in {wall:.1f}s — "
          f"{total_hits} hits, {total_escalated} escalated, "
          f"best delta = {best_delta:+.4f}")

    if best_hit is not None:
        print(f"  Best hit: shift={best_hit.shift}, "
              f"traj=({best_hit.traj_dir}), "
              f"depth={best_hit.depth}, log_q={best_hit.log_q:.2f}")

    return {
        "cmf_name": cmf_name,
        "wall_time": wall,
        "total_hits": total_hits,
        "total_escalated": total_escalated,
        "best_delta": best_delta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Local Mac test runner for Dreams-RNS-ROCm"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--cmf", type=int, default=None,
                        help="Run only this CMF index (default: all)")
    parser.add_argument("--list-cmfs", action="store_true",
                        help="List available CMFs and exit")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./outputs/local_<timestamp>)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Compile CMFs
    programs = compile_cmfs_from_config(config)

    if args.list_cmfs:
        print("Available CMFs:")
        for i, (prog, cdef) in enumerate(programs):
            print(f"  [{i}] {cdef.get('name', f'cmf_{i}')} "
                  f"({prog.m}x{prog.m}, dim={prog.dim}, "
                  f"target={cdef.get('target', '?')})")
        return

    # Output directory
    output_dir = Path(args.output_dir or
                      f"./outputs/local_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"local_{int(time.time())}"

    print("=" * 60)
    print("Dreams-RNS-ROCm — Local Mac Test")
    print("=" * 60)
    print(f"  Config:  {args.config}")
    print(f"  Output:  {output_dir}")
    print(f"  Run ID:  {run_id}")
    print(f"  CMFs:    {len(programs)}")
    print(f"  Shifts:  {config.get('shifts', {}).get('count', 10)}")
    print(f"  Trajs:   {config.get('trajectories', {}).get('count', 100)}")
    print(f"  Depth:   {config.get('walk', {}).get('depth', 200)}")
    print(f"  K:       {config.get('rns', {}).get('K', 16)}")
    print("=" * 60)

    # Write manifest
    manifest = create_manifest(run_id=run_id, config=config, n_ranks=1)
    manifest.save(output_dir / "manifest.json")

    # Logger
    logger = RunLogger(
        output_dir=output_dir,
        rank=0,
        delta_threshold=config.get("triage", {}).get("delta_threshold", -10.0),
    )

    # Select CMFs
    if args.cmf is not None:
        if args.cmf >= len(programs):
            print(f"ERROR: CMF index {args.cmf} out of range "
                  f"(have {len(programs)} CMFs)")
            sys.exit(1)
        selected = [(args.cmf, *programs[args.cmf])]
    else:
        selected = [(i, prog, cdef) for i, (prog, cdef) in enumerate(programs)]

    # Run
    all_metrics = []
    t_total = time.time()

    for cmf_idx, program, cmf_def in selected:
        cmf_name = cmf_def.get("name", f"cmf_{cmf_idx}")
        print(f"\n{'─' * 60}")
        print(f"CMF [{cmf_idx}]: {cmf_name} "
              f"({program.m}x{program.m}, dim={program.dim})")
        print(f"{'─' * 60}")

        try:
            metrics = process_cmf(program, cmf_def, cmf_idx, config, logger)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    logger.close()

    # Summary
    total_wall = time.time() - t_total
    total_hits = sum(m["total_hits"] for m in all_metrics)
    total_esc = sum(m["total_escalated"] for m in all_metrics)

    summary = {
        "run_id": run_id,
        "n_ranks": 1,
        "n_cmfs": len(selected),
        "total_results": total_hits,
        "total_positives": total_esc,
        "wall_time_sec": total_wall,
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Run complete: {run_id}")
    print(f"  Wall time:  {total_wall:.1f}s")
    print(f"  Hits:       {total_hits}")
    print(f"  Escalated:  {total_esc}")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Task list generator for Dreams-RNS-ROCm pipeline.

Generates a CSV/JSON task list describing CMFs, shifts, and trajectory sets
for the MPI sweep. Useful for:
  - Pre-computing task assignments
  - Resuming interrupted runs
  - Inspecting the workload before submission

Usage:
    python scripts/make_tasks.py --config configs/lumi_1node_8gpu.yaml
    python scripts/make_tasks.py --config configs/lumi_1node_8gpu.yaml --format csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.cmf_compile import compile_cmf_from_dict
from dreams_rocm.trajectories import generate_trajectories
from dreams_rocm.shifts import generate_shifts
from dreams_rocm.crt.delta_targets import ZETA_TARGETS


def main():
    parser = argparse.ArgumentParser(description="Generate task list")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--n-ranks", type=int, default=8,
                        help="Number of MPI ranks for assignment preview")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    shift_cfg = config.get("shifts", {})
    traj_cfg = config.get("trajectories", {})

    # Generate trajectories (shared across all CMFs)
    trajectories = generate_trajectories(
        count=traj_cfg.get("count", 1000),
        max_component=traj_cfg.get("max_component", 50),
    )

    tasks = []
    for idx, cmf_def in enumerate(config.get("cmf_families", [])):
        # Compile to validate
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
            name=cmf_def.get("name", f"cmf_{idx}"),
        )

        # Generate shifts for this CMF
        shifts = generate_shifts(
            n_shifts=shift_cfg.get("count", 100),
            dim=cmf_def["dim"],
            method=shift_cfg.get("method", "grid"),
            bounds=tuple(shift_cfg.get("bounds", [-1000, 1000])),
            seed=shift_cfg.get("seed", 42),
            cmf_idx=idx,
        )

        target_name = cmf_def.get("target", config.get("primary_target", "zeta3"))

        task = {
            "cmf_idx": idx,
            "cmf_name": cmf_def.get("name", f"cmf_{idx}"),
            "m": cmf_def["m"],
            "dim": cmf_def["dim"],
            "n_instructions": len(program.instructions),
            "n_constants": len(program.constants),
            "n_shifts": len(shifts),
            "n_trajectories": len(trajectories),
            "total_evaluations": len(shifts) * len(trajectories),
            "target": target_name,
            "assigned_rank": idx % args.n_ranks,
        }
        tasks.append(task)

    # Summary
    total_evals = sum(t["total_evaluations"] for t in tasks)
    summary = {
        "n_cmfs": len(tasks),
        "n_trajectories": len(trajectories),
        "total_evaluations": total_evals,
        "n_ranks": args.n_ranks,
    }

    # Output
    if args.format == "json":
        output = {"summary": summary, "tasks": tasks}
        text = json.dumps(output, indent=2)
    else:
        # CSV
        import io
        sio = io.StringIO()
        if tasks:
            writer = csv.DictWriter(sio, fieldnames=tasks[0].keys())
            writer.writeheader()
            writer.writerows(tasks)
        text = sio.getvalue()

    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {len(tasks)} tasks to {args.output}")
    else:
        print(text)

    # Print summary to stderr
    print(f"\nSummary:", file=sys.stderr)
    print(f"  CMFs: {summary['n_cmfs']}", file=sys.stderr)
    print(f"  Trajectories: {summary['n_trajectories']}", file=sys.stderr)
    print(f"  Total evaluations: {summary['total_evaluations']:,}", file=sys.stderr)
    print(f"  Ranks: {summary['n_ranks']}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate pre-computed trajectory and shift files for each dimension.

Produces JSON files that can be loaded directly by the LUMI runner
without re-computing at job time.

Output structure:
  sweep_data/
    trajectories/
      dim4_trajectories.json    (1,120 entries)
      dim5_trajectories.json    (8,161 entries)
      dim6_trajectories.json    (7,448 entries)
      dim7_trajectories.json    (37,969 entries)
      dim9_trajectories.json    (9,841 entries)
    shifts/
      dim4_shifts.json          (512 entries)
      dim5_shifts.json          (512 entries)
      dim6_shifts.json          (1,024 entries)
      dim7_shifts.json          (1,024 entries)
      dim9_shifts.json          (1,024 entries)

Usage:
    python scripts/generate_sweep_data.py --output-dir sweep_data
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreams_rocm.exhaust import (
    exhaust_trajectories, exhaust_shifts, exhaust_summary, EXHAUST_PRESETS,
)
from dreams_rocm.cmf_generator import FAMILY_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory + shift files")
    parser.add_argument("--output-dir", type=str, default="sweep_data",
                        help="Base output directory")
    args = parser.parse_args()

    traj_dir = os.path.join(args.output_dir, "trajectories")
    shift_dir = os.path.join(args.output_dir, "shifts")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(shift_dir, exist_ok=True)

    dims_seen = set()
    total_traj = 0
    total_shifts = 0

    print(f"{'='*70}")
    print(f"Generating trajectories and shifts for all pFq families")
    print(f"{'='*70}")

    for fam, cfg in FAMILY_CONFIG.items():
        dim = cfg["dim"]
        if dim in dims_seen:
            continue
        dims_seen.add(dim)

        print(f"\n--- dim={dim} ({fam}: {cfg['p']}F{cfg['q']}, rank={cfg['rank']}) ---")

        # Trajectories
        t0 = time.time()
        trajs = exhaust_trajectories(dim)
        t_traj = time.time() - t0

        traj_path = os.path.join(traj_dir, f"dim{dim}_trajectories.json")
        with open(traj_path, 'w') as f:
            json.dump({
                "dim": dim,
                "k_max": EXHAUST_PRESETS[dim]["k_max"],
                "count": len(trajs),
                "trajectories": [list(t) for t in trajs],
            }, f)

        traj_size_mb = os.path.getsize(traj_path) / 1024 / 1024
        print(f"  Trajectories: {len(trajs):>8,}  ({t_traj:.1f}s, {traj_size_mb:.1f} MB)")
        print(f"    → {traj_path}")
        total_traj += len(trajs)

        # Shifts
        t0 = time.time()
        shifts = exhaust_shifts(dim, seed=0)
        t_shift = time.time() - t0

        shift_path = os.path.join(shift_dir, f"dim{dim}_shifts.json")
        shift_data = []
        for s in shifts:
            shift_data.append({
                "nums": list(s.nums),
                "dens": list(s.dens),
                "tokens": s.as_tokens(),
            })

        with open(shift_path, 'w') as f:
            json.dump({
                "dim": dim,
                "count": len(shifts),
                "shifts": shift_data,
            }, f)

        shift_size_mb = os.path.getsize(shift_path) / 1024 / 1024
        print(f"  Shifts:       {len(shifts):>8,}  ({t_shift:.1f}s, {shift_size_mb:.2f} MB)")
        print(f"    → {shift_path}")
        total_shifts += len(shifts)

    # Summary manifest
    manifest = {"families": {}, "totals": {}}
    for fam, cfg in FAMILY_CONFIG.items():
        dim = cfg["dim"]
        s = exhaust_summary(dim)
        manifest["families"][fam] = {
            "p": cfg["p"], "q": cfg["q"], "rank": cfg["rank"], "dim": dim,
            "n_trajectories": s["n_trajectories"],
            "n_shifts": s["n_shifts"],
            "runs_per_cmf": s["runs_per_cmf"],
            "runs_per_1000_cmfs": s["runs_per_cmf"] * 1000,
            "runs_per_10000_cmfs": s["runs_per_cmf"] * 10000,
        }

    grand_total_runs = sum(
        v["runs_per_10000_cmfs"] for v in manifest["families"].values()
    )
    manifest["totals"] = {
        "n_families": len(FAMILY_CONFIG),
        "n_cmfs_per_family": 10000,
        "n_cmfs_total": 50000,
        "grand_total_runs": grand_total_runs,
    }

    manifest_path = os.path.join(args.output_dir, "sweep_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total trajectories generated: {total_traj:,}")
    print(f"  Total shifts generated:       {total_shifts:,}")
    print(f"\n  Per-family compute (10,000 CMFs each):")

    for fam, info in manifest["families"].items():
        print(f"    {fam} (dim={info['dim']}): "
              f"{info['n_trajectories']:>6,} traj × {info['n_shifts']:>5,} shifts "
              f"× 10,000 CMFs = {info['runs_per_10000_cmfs']:>15,} runs")

    print(f"\n  GRAND TOTAL: {grand_total_runs:,} runs across 50,000 CMFs")

    # Compute time estimate
    ms_per_run = 0.5  # conservative for MI250X at depth=2000 K=32
    gpu_hours_total = grand_total_runs * ms_per_run / 1000 / 3600
    node_hours = gpu_hours_total / 8  # 8 GPUs per node
    print(f"\n  Estimated compute (at {ms_per_run}ms/run on MI250X):")
    print(f"    GPU-hours:  {gpu_hours_total:>12,.0f}")
    print(f"    Node-hours: {node_hours:>12,.0f} (8 GPU/node)")
    print(f"    Wall days:  {node_hours / 24:>12,.1f}")

    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

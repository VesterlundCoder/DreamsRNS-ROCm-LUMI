#!/usr/bin/env python3
"""
worker_level2_from_jobs.py — Level-2 worker for 6F5 CMF sweep.

Reads a single job from level2_jobs.jsonl (indexed by SLURM_ARRAY_TASK_ID
or --job_index), expands the shard range into (s_idx, d_idx) pairs,
runs CMF walks at the escalated depth, and matches only against
the job's selected target constants.

Modes:
  - full_shard_expansion: enumerate all (s_idx in range) × (d_idx in range)
  - local_neighborhood: enumerate around center ± radius (wrap mod N)

Outputs:
  results/hits_depth{DEPTH}_job{JOB_IDX:06d}.jsonl
  progress/pairs_tested_L2_job{JOB_IDX:06d}.jsonl

Hit schema matches Level-1 (with level="level2"):
  {cmf_id, level, depth, pair:{s_idx,d_idx,s_shard,d_shard,shard_id},
   target_family, target_const, score, residual,
   shift:{nums,dens}, trajectory:{v}}
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

# Re-use core functions from Level-1 worker
# In production, these would be in a shared module
AXES_11 = [f"x{i}" for i in range(6)] + [f"y{j}" for j in range(5)]


def load_target_constants() -> Dict[str, float]:
    try:
        import mpmath as mp
        mp.mp.dps = 50
        return {
            "zeta3": float(mp.zeta(3)),
            "zeta5": float(mp.zeta(5)),
            "zeta7": float(mp.zeta(7)),
            "zeta9": float(mp.zeta(9)),
            "catalan": float(mp.catalan),
            "pi": float(mp.pi),
        }
    except ImportError:
        return {
            "zeta3": 1.2020569031595942,
            "zeta5": 1.0369277551433699,
            "zeta7": 1.0083492773819228,
            "zeta9": 1.0020083928260822,
            "catalan": 0.9159655941772190,
            "pi": 3.141592653589793,
        }


def match_against_targets(
    estimate: float,
    targets: Dict[str, float],
    threshold: float = 1e-8,
) -> List[Dict[str, Any]]:
    hits = []
    if not math.isfinite(estimate) or estimate == 0.0:
        return hits
    for name, val in targets.items():
        residual = abs(estimate - val)
        if residual == 0.0:
            score = 300.0
        elif residual < threshold:
            score = -math.log10(residual)
            hits.append({
                "target_family": "zeta_basis",
                "target_const": name,
                "score": round(score, 2),
                "residual": f"{residual:.6e}",
            })
        if abs(estimate) > 0.5 and abs(val) > 0.5:
            ratio_est = estimate - 1.0
            ratio_val = val - 1.0
            if abs(ratio_val) > 1e-15:
                ratio_res = abs(ratio_est - ratio_val)
                if ratio_res < threshold and ratio_res < residual:
                    ratio_score = -math.log10(ratio_res)
                    hits.append({
                        "target_family": "zeta_basis_ratio",
                        "target_const": name,
                        "score": round(ratio_score, 2),
                        "residual": f"{ratio_res:.6e}",
                    })
    return hits


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_shard(s_idx: int, d_idx: int, S_BIN: int, D_BIN: int, N_DIRS: int) -> Tuple[int, int, int]:
    s_shard = s_idx // S_BIN
    d_shard = d_idx // D_BIN
    N_DIR_SHARDS = (N_DIRS + D_BIN - 1) // D_BIN
    shard_id = s_shard * N_DIR_SHARDS + d_shard
    return s_shard, d_shard, shard_id


def compute_cmf_walk(cmf, shift, direction, depth) -> Optional[float]:
    """
    TODO: Replace with your Dreams RNS walk backend.
    See worker_variantB_skeleton.py for the ramanujantools fallback.
    """
    return None


def enumerate_pairs_full_shard(ranges: dict) -> List[Tuple[int, int]]:
    """Enumerate all (s_idx, d_idx) in the shard range."""
    s0, s1 = ranges["s_idx"]
    d0, d1 = ranges["d_idx"]
    pairs = []
    for s in range(s0, s1 + 1):
        for d in range(d0, d1 + 1):
            pairs.append((s, d))
    return pairs


def enumerate_pairs_neighborhood(
    center: dict, radius: dict, N_SHIFTS: int, N_DIRS: int
) -> List[Tuple[int, int]]:
    """Enumerate (s_idx, d_idx) around center ± radius, wrapping."""
    cs = center["s_idx"]
    cd = center["d_idx"]
    rs = radius["rs"]
    rd = radius["rd"]
    pairs = []
    for ds in range(-rs, rs + 1):
        s = (cs + ds) % N_SHIFTS
        for dd in range(-rd, rd + 1):
            d = (cd + dd) % N_DIRS
            pairs.append((s, d))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Level-2 worker (job array)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--job_json", default=None,
                    help="Path to level2_jobs.jsonl (default: run_dir/results/level2_jobs.jsonl)")
    ap.add_argument("--job_index", type=int, default=None,
                    help="Job index (default: SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--shifts_jsonl", default=None)
    ap.add_argument("--dirs_jsonl", default=None)
    ap.add_argument("--cmfs_jsonl", default=None)
    ap.add_argument("--pre_score", type=float, default=1e-8)
    ap.add_argument("--S_BIN", type=int, default=64)
    ap.add_argument("--D_BIN", type=int, default=1000)
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")

    # Determine job index
    job_idx = args.job_index
    if job_idx is None:
        job_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Load job definition
    jobs_path = args.job_json or f"{run_dir}/results/level2_jobs.jsonl"
    jobs = load_jsonl(jobs_path)
    if job_idx >= len(jobs):
        print(f"[L2] job_index={job_idx} >= {len(jobs)} jobs. Nothing to do.")
        return
    job = jobs[job_idx]

    cmf_id = job["cmf_id"]
    depth = job["depth"]
    mode = job["mode"]
    job_targets = job.get("targets", [])

    print(f"[L2] job_index={job_idx} cmf={cmf_id} depth={depth} mode={mode} "
          f"targets={job_targets}", flush=True)

    # Load manifest for N_SHIFTS, N_DIRS
    manifest_path = f"{run_dir}/manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    N_SHIFTS = manifest["pairing_rule"]["n_shifts"]
    N_DIRS = manifest["pairing_rule"]["n_dirs"]
    S_BIN = manifest["sharding"]["S_BIN"]
    D_BIN = manifest["sharding"]["D_BIN"]

    # Load shifts + dirs
    shifts_path = args.shifts_jsonl
    if shifts_path is None:
        # Try to find in run_dir or parent
        for candidate in [f"{run_dir}/tasks_6f5_zeta5_shifts.jsonl",
                          f"{run_dir}/../tasks_6f5_zeta5_shifts.jsonl"]:
            if os.path.exists(candidate):
                shifts_path = candidate
                break
    if shifts_path is None:
        raise SystemExit("Cannot find shifts JSONL")

    dirs_path = args.dirs_jsonl
    if dirs_path is None:
        for candidate in [f"{run_dir}/tasks_6f5_zeta5_dirs.jsonl",
                          f"{run_dir}/../tasks_6f5_zeta5_dirs.jsonl"]:
            if os.path.exists(candidate):
                dirs_path = candidate
                break
    if dirs_path is None:
        raise SystemExit("Cannot find dirs JSONL")

    shifts = load_jsonl(shifts_path)
    dirs = load_jsonl(dirs_path)

    # Load CMF (find by cmf_id)
    cmfs_path = args.cmfs_jsonl
    if cmfs_path is None:
        for candidate in [f"{run_dir}/cmfs_level1.jsonl",
                          f"{run_dir}/../cmfs_level1.jsonl"]:
            if os.path.exists(candidate):
                cmfs_path = candidate
                break
    cmfs = load_jsonl(cmfs_path) if cmfs_path else []
    cmf = next((c for c in cmfs if c.get("cmf_id") == cmf_id), cmfs[0] if cmfs else {})

    # Load only requested target constants
    all_targets = load_target_constants()
    targets = {k: v for k, v in all_targets.items() if k in job_targets}
    if not targets:
        targets = all_targets  # fallback to all

    # Enumerate pairs based on mode
    if mode == "full_shard_expansion":
        pairs = enumerate_pairs_full_shard(job["ranges"])
    elif mode == "local_neighborhood":
        pairs = enumerate_pairs_neighborhood(
            job["center"], job["radius"], N_SHIFTS, N_DIRS)
    else:
        print(f"[L2] Unknown mode: {mode}")
        return

    print(f"[L2] Enumerating {len(pairs)} pairs", flush=True)

    # Output files
    os.makedirs(f"{run_dir}/results", exist_ok=True)
    os.makedirs(f"{run_dir}/progress", exist_ok=True)
    hits_path = f"{run_dir}/results/hits_depth{depth}_job{job_idx:06d}.jsonl"
    pt_path = f"{run_dir}/progress/pairs_tested_L2_job{job_idx:06d}.jsonl"

    hits_f = open(hits_path, "w")
    shard_pairs = defaultdict(int)
    n_hits = 0
    t0 = time.time()

    for pi, (s_idx, d_idx) in enumerate(pairs):
        shift = shifts[s_idx]
        direction = dirs[d_idx]["v"]
        s_shard, d_shard, shard_id = compute_shard(s_idx, d_idx, S_BIN, D_BIN, N_DIRS)
        shard_pairs[(cmf_id, shard_id)] += 1

        # ── COMPUTE WALK ──
        estimate = compute_cmf_walk(cmf, shift, direction, depth)
        if estimate is None:
            continue

        # ── MATCH AGAINST TARGETS ──
        hit_list = match_against_targets(estimate, targets, args.pre_score)
        for hit in hit_list:
            n_hits += 1
            hr = {
                "cmf_id": cmf_id,
                "level": "level2",
                "depth": depth,
                "pair": {
                    "s_idx": s_idx,
                    "d_idx": d_idx,
                    "s_shard": s_shard,
                    "d_shard": d_shard,
                    "shard_id": shard_id,
                },
                "target_family": hit["target_family"],
                "target_const": hit["target_const"],
                "score": hit["score"],
                "residual": hit["residual"],
                "shift": {"nums": shift["nums"], "dens": shift["dens"]},
                "trajectory": {"v": direction},
            }
            hits_f.write(json.dumps(hr, separators=(",", ":")) + "\n")
            hits_f.flush()

        if (pi + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (pi + 1) / max(elapsed, 1e-9)
            print(f"[L2] pairs={pi+1}/{len(pairs)} hits={n_hits} "
                  f"rate={rate:.0f}/s", flush=True)

    hits_f.close()

    # Flush pairs_tested
    with open(pt_path, "w") as f:
        for (cid, sid), count in sorted(shard_pairs.items()):
            row = {"cmf_id": cid, "shard_id": sid, "pairs_tested": count}
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    elapsed = time.time() - t0
    print(f"[L2] Done job={job_idx} pairs={len(pairs)} hits={n_hits} "
          f"wall={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
worker_variantB_skeleton.py — Level-1 Variant-B worker for 6F5 CMF sweep.

Each rank processes a slice of shifts. For each shift s_idx, it generates
dirs_per_shift direction indices via the deterministic pairing rule, runs
CMF walks at the specified depth, and matches against target constants.

Pairing rule (Variant B):
  base = (mix_a * s_idx + mix_b) % N_DIRS
  for k in 0..dirs_per_shift-1:
      d_idx = (base + k * stride) % N_DIRS

Outputs (per rank):
  results/hits_level1_rank{RANK:05d}.jsonl
  progress/pairs_tested_rank{RANK:05d}.jsonl

Hit schema per row:
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
from fractions import Fraction
from typing import List, Dict, Tuple, Optional, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS / TARGET BANK
# ═══════════════════════════════════════════════════════════════════════════

AXES_11 = [f"x{i}" for i in range(6)] + [f"y{j}" for j in range(5)]

def load_target_constants() -> Dict[str, float]:
    """Return target constants to match against."""
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
        # Fallback hardcoded values (50-digit precision truncated to float64)
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
    """
    Match an estimate against target constants.
    Returns list of hits above threshold.
    Score = -log10(|estimate - target|) (higher = better match).
    """
    hits = []
    if not math.isfinite(estimate) or estimate == 0.0:
        return hits
    for name, val in targets.items():
        residual = abs(estimate - val)
        if residual == 0.0:
            score = 300.0  # perfect match
        elif residual < threshold:
            score = -math.log10(residual)
            hits.append({
                "target_family": "zeta_basis",
                "target_const": name,
                "score": round(score, 2),
                "residual": f"{residual:.6e}",
            })
        # Also check ratio matching for values near 1.0
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_shifts(path: str) -> List[dict]:
    """Load shifts JSONL. Each row: {idx, nums:[11], dens:[11]}"""
    return load_jsonl(path)


def load_dirs(path: str) -> List[dict]:
    """Load dirs JSONL. Each row: {idx, v:[11 ints]}"""
    return load_jsonl(path)


def load_cmfs(path: str) -> List[dict]:
    """Load CMFs JSONL."""
    return load_jsonl(path)


# ═══════════════════════════════════════════════════════════════════════════
# CMF WALK COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_cmf_walk(
    cmf: dict,
    shift: dict,
    direction: List[int],
    depth: int,
) -> Optional[float]:
    """
    Run a CMF walk and return the limit estimate.

    This is the computational core — replace with your Dreams RNS backend
    for GPU acceleration.

    Args:
        cmf: CMF definition dict with 'matrices', 'axes', 'rank', etc.
        shift: {'nums': [11], 'dens': [11]} rational shift
        direction: [11 ints] trajectory direction
        depth: number of walk steps

    Returns:
        float estimate of the limit, or None if computation fails.
    """
    # ──────────────────────────────────────────────────────────────────
    # TODO: Replace this with your actual Dreams RNS walk.
    #
    # For the real implementation, you would:
    # 1. Build the start position: base_params + shift
    #    start = {axes[i]: upper/lower[i] + shift_nums[i]/shift_dens[i]}
    # 2. Build the trajectory: {axes[i]: direction[i]}
    # 3. Compile the 5×5 trajectory matrix M(n) from the CMF axis matrices
    # 4. Compute P(N) = M(0) · M(1) · ... · M(N-1) via RNS modular walk
    # 5. Extract limit from P: estimate = P[0,rank-1] / P[1,rank-1]
    #
    # For now, this returns None (placeholder).
    # ──────────────────────────────────────────────────────────────────
    return None


def compute_cmf_walk_ramanujantools(
    cmf: dict,
    shift: dict,
    direction: List[int],
    depth: int,
) -> Optional[float]:
    """
    Fallback: use ramanujantools pFq.limit() for exact symbolic computation.
    SLOW — only for validation, not for 800M sweep.
    """
    try:
        import sympy as sp
        rmtools_path = os.environ.get("RAMANUJANTOOLS_PATH", "")
        if rmtools_path and rmtools_path not in sys.path:
            sys.path.insert(0, rmtools_path)
        from ramanujantools import Position
        from ramanujantools.cmf.pfq import pFq

        p, q, z_val = cmf["p"], cmf["q"], int(cmf["z"])
        pfq = pFq(p, q, sp.S(z_val))
        axes = sorted(list(pfq.axes()), key=str)

        # Build start position: base params + shift
        upper = [sp.Rational(u) for u in cmf["upper_params"]]
        lower = [sp.Rational(l) for l in cmf["lower_params"]]
        base = {}
        for i in range(p):
            base[axes[i]] = upper[i]
        for j in range(q):
            base[axes[p + j]] = lower[j]

        # Apply shift
        for k in range(len(axes)):
            s_num = shift["nums"][k]
            s_den = shift["dens"][k]
            base[axes[k]] += sp.Rational(s_num, s_den)

        start = Position(base)
        trajectory = Position({axes[k]: direction[k] for k in range(len(axes))})

        # Check trajectory is not all zeros
        if all(direction[k] == 0 for k in range(len(axes))):
            return None

        lim = pfq.limit(dict(trajectory), depth, dict(start))
        return float(lim.as_float())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT B PAIRING RULE
# ═══════════════════════════════════════════════════════════════════════════

def variant_b_dirs_for_shift(
    s_idx: int,
    n_dirs: int,
    dirs_per_shift: int,
    stride: int,
    mix_a: int,
    mix_b: int,
) -> List[int]:
    """Deterministic pairing: return list of d_idx for a given s_idx."""
    base = (mix_a * s_idx + mix_b) % n_dirs
    return [(base + k * stride) % n_dirs for k in range(dirs_per_shift)]


# ═══════════════════════════════════════════════════════════════════════════
# SHARDING
# ═══════════════════════════════════════════════════════════════════════════

def compute_shard(s_idx: int, d_idx: int, S_BIN: int, D_BIN: int, N_DIRS: int) -> Tuple[int, int, int]:
    """Return (s_shard, d_shard, shard_id)."""
    s_shard = s_idx // S_BIN
    d_shard = d_idx // D_BIN
    N_DIR_SHARDS = (N_DIRS + D_BIN - 1) // D_BIN
    shard_id = s_shard * N_DIR_SHARDS + d_shard
    return s_shard, d_shard, shard_id


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WORKER LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Level-1 Variant-B CMF sweep worker")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--cmfs_jsonl", required=True)
    ap.add_argument("--shifts_jsonl", required=True)
    ap.add_argument("--dirs_jsonl", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--depth", type=int, default=2000)
    ap.add_argument("--dirs_per_shift", type=int, default=4883)
    ap.add_argument("--stride", type=int, default=99991)
    ap.add_argument("--mix_a", type=int, default=104729)
    ap.add_argument("--mix_b", type=int, default=12345)
    ap.add_argument("--S_BIN", type=int, default=64)
    ap.add_argument("--D_BIN", type=int, default=1000)
    ap.add_argument("--pre_score", type=float, default=1e-8,
                    help="Threshold for pre-screening hits")
    ap.add_argument("--use_ramanujantools", action="store_true",
                    help="Use ramanujantools (slow) instead of Dreams RNS")
    args = ap.parse_args()

    rank = args.rank
    world = args.world
    run_dir = args.run_dir.rstrip("/")
    os.makedirs(f"{run_dir}/results", exist_ok=True)
    os.makedirs(f"{run_dir}/progress", exist_ok=True)
    os.makedirs(f"{run_dir}/logs", exist_ok=True)

    # Setup logging
    log_path = f"{run_dir}/logs/rank{rank:05d}.log"
    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [rank{rank:05d}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting Level-1 worker: rank={rank}/{world} depth={args.depth}")

    # Load data
    log(f"Loading shifts from {args.shifts_jsonl}")
    shifts = load_shifts(args.shifts_jsonl)
    N_SHIFTS = len(shifts)
    log(f"  {N_SHIFTS} shifts loaded")

    log(f"Loading dirs from {args.dirs_jsonl}")
    dirs = load_dirs(args.dirs_jsonl)
    N_DIRS = len(dirs)
    log(f"  {N_DIRS} dirs loaded")

    log(f"Loading CMFs from {args.cmfs_jsonl}")
    cmfs = load_cmfs(args.cmfs_jsonl)
    log(f"  {len(cmfs)} CMFs loaded")

    # Load target constants
    targets = load_target_constants()
    log(f"Target constants: {list(targets.keys())}")

    # Write manifest (rank 0 only)
    if rank == 0:
        import datetime
        manifest = {
            "run_id": os.path.basename(run_dir),
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "variant": "B",
            "cmf_family": "6F5_zeta5",
            "depths": {"level1": args.depth, "level2": 2000},
            "pairing_rule": {
                "n_shifts": N_SHIFTS,
                "n_dirs": N_DIRS,
                "dirs_per_shift": args.dirs_per_shift,
                "stride": args.stride,
                "mix_a": args.mix_a,
                "mix_b": args.mix_b,
            },
            "sharding": {"S_BIN": args.S_BIN, "D_BIN": args.D_BIN},
            "targets_level1": list(targets.keys()),
            "thresholds": {"pre_score": args.pre_score, "hit_residual": 1e-20},
            "repro": {"config_hash": "95fe7e107776"},
        }
        with open(f"{run_dir}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        coverage = {
            "n_shifts": N_SHIFTS,
            "n_dirs": N_DIRS,
            "dirs_per_shift": args.dirs_per_shift,
            "dir_index_rule": "d = ((a*s + b) mod n_dirs + k*stride) mod n_dirs, k=0..dirs_per_shift-1",
            "a": args.mix_a,
            "b": args.mix_b,
            "stride": args.stride,
            "s_interval": [0, N_SHIFTS - 1],
            "d_interval": [0, N_DIRS - 1],
        }
        with open(f"{run_dir}/coverage.json", "w") as f:
            json.dump(coverage, f, indent=2)
        log("Wrote manifest.json + coverage.json")

    # Partition shifts across ranks (round-robin)
    my_shift_indices = list(range(rank, N_SHIFTS, world))
    total_pairs = len(my_shift_indices) * args.dirs_per_shift
    log(f"This rank handles {len(my_shift_indices)} shifts, {total_pairs} pairs")

    # Open output files
    hits_path = f"{run_dir}/results/hits_level1_rank{rank:05d}.jsonl"
    pairs_tested_path = f"{run_dir}/progress/pairs_tested_rank{rank:05d}.jsonl"
    hits_f = open(hits_path, "w")
    pairs_tested_f = open(pairs_tested_path, "w")

    # Shard tracking
    shard_pairs: Dict[Tuple[str, int], int] = defaultdict(int)
    n_hits_total = 0
    n_pairs_total = 0
    t0 = time.time()

    # Select walk function
    walk_fn = compute_cmf_walk_ramanujantools if args.use_ramanujantools else compute_cmf_walk

    for cmf in cmfs:
        cmf_id = cmf.get("cmf_id", "cmf_unknown")
        log(f"Processing CMF: {cmf_id}")

        for si_count, s_idx in enumerate(my_shift_indices):
            shift = shifts[s_idx]
            d_indices = variant_b_dirs_for_shift(
                s_idx, N_DIRS, args.dirs_per_shift,
                args.stride, args.mix_a, args.mix_b,
            )

            for d_idx in d_indices:
                direction = dirs[d_idx]["v"]
                s_shard, d_shard, shard_id = compute_shard(
                    s_idx, d_idx, args.S_BIN, args.D_BIN, N_DIRS,
                )

                # Track coverage
                shard_pairs[(cmf_id, shard_id)] += 1
                n_pairs_total += 1

                # ── COMPUTE WALK ──
                estimate = walk_fn(cmf, shift, direction, args.depth)
                if estimate is None:
                    continue

                # ── MATCH AGAINST TARGETS ──
                hit_list = match_against_targets(estimate, targets, args.pre_score)
                for hit in hit_list:
                    n_hits_total += 1
                    hr = {
                        "cmf_id": cmf_id,
                        "level": "level1",
                        "depth": args.depth,
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

            # Progress logging every 100 shifts
            if (si_count + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = n_pairs_total / max(elapsed, 1e-9)
                log(f"  shifts={si_count+1}/{len(my_shift_indices)} "
                    f"pairs={n_pairs_total} hits={n_hits_total} "
                    f"rate={rate:.0f} pairs/s")

        # Flush shard pairs_tested for this CMF
        for (cid, sid), count in sorted(shard_pairs.items()):
            if cid == cmf_id:
                row = {"cmf_id": cid, "shard_id": sid, "pairs_tested": count}
                pairs_tested_f.write(json.dumps(row, separators=(",", ":")) + "\n")
        pairs_tested_f.flush()

    hits_f.close()
    pairs_tested_f.close()

    elapsed = time.time() - t0
    log(f"Done. pairs={n_pairs_total} hits={n_hits_total} wall={elapsed:.1f}s")
    log(f"  Hits: {hits_path}")
    log(f"  Pairs tested: {pairs_tested_path}")


if __name__ == "__main__":
    main()

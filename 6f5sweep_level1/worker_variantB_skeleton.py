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

# Scalar multipliers applied to each base constant.
# For base constant c, we test: n*c for each n in SCALAR_MULTIPLIERS.
# Negative variants are handled automatically in match_against_targets.
SCALAR_MULTIPLIERS = [
    ("1",    1),
    ("2",    2),
    ("3",    3),
    ("4",    4),
    ("6",    6),
    ("1/2",  0.5),
    ("1/3",  1.0/3),
    ("1/4",  0.25),
    ("1/6",  1.0/6),
    ("1/12", 1.0/12),
]


def load_target_constants() -> Dict[str, float]:
    """Return an expanded bank of target constants.

    Base constants (20+ irrationals/transcendentals):
      pi, e, ln2, ln3, ln5, ln10,
      euler_gamma, catalan,
      zeta(2)..zeta(9),
      sqrt2, sqrt3, sqrt5, golden_ratio,
      pi^2, 1/pi,
      Apery = zeta(3) (included via zeta series)

    Each base constant is multiplied by the SCALAR_MULTIPLIERS list,
    giving ~200 distinct target values.
    """
    try:
        import mpmath as mp
        mp.mp.dps = 50
        base = {
            # --- Transcendentals ---
            "pi":            float(mp.pi),
            "e":             float(mp.e),
            "ln2":           float(mp.log(2)),
            "ln3":           float(mp.log(3)),
            "ln5":           float(mp.log(5)),
            "ln10":          float(mp.log(10)),
            "euler_gamma":   float(mp.euler),
            "catalan":       float(mp.catalan),
            # --- Zeta values ---
            "zeta2":         float(mp.zeta(2)),    # pi^2/6
            "zeta3":         float(mp.zeta(3)),    # Apery
            "zeta4":         float(mp.zeta(4)),    # pi^4/90
            "zeta5":         float(mp.zeta(5)),
            "zeta6":         float(mp.zeta(6)),
            "zeta7":         float(mp.zeta(7)),
            "zeta8":         float(mp.zeta(8)),
            "zeta9":         float(mp.zeta(9)),
            # --- Algebraic irrationals ---
            "sqrt2":         float(mp.sqrt(2)),
            "sqrt3":         float(mp.sqrt(3)),
            "sqrt5":         float(mp.sqrt(5)),
            "phi":           float((1 + mp.sqrt(5)) / 2),  # golden ratio
            # --- Derived transcendentals ---
            "pi_sq":         float(mp.pi ** 2),
            "1/pi":          float(1 / mp.pi),
            "pi/4":          float(mp.pi / 4),     # arctan(1)
        }
    except ImportError:
        # Hardcoded fallback (float64 precision)
        import math as _m
        base = {
            "pi":            _m.pi,
            "e":             _m.e,
            "ln2":           _m.log(2),
            "ln3":           _m.log(3),
            "ln5":           _m.log(5),
            "ln10":          _m.log(10),
            "euler_gamma":   0.5772156649015329,
            "catalan":       0.9159655941772190,
            "zeta2":         1.6449340668482264,
            "zeta3":         1.2020569031595942,
            "zeta4":         1.0823232337111382,
            "zeta5":         1.0369277551433699,
            "zeta6":         1.0173430619844491,
            "zeta7":         1.0083492773819228,
            "zeta8":         1.0040773561979443,
            "zeta9":         1.0020083928260822,
            "sqrt2":         _m.sqrt(2),
            "sqrt3":         _m.sqrt(3),
            "sqrt5":         _m.sqrt(5),
            "phi":           (1 + _m.sqrt(5)) / 2,
            "pi_sq":         _m.pi ** 2,
            "1/pi":          1.0 / _m.pi,
            "pi/4":          _m.pi / 4,
        }

    # Expand: base × scalar multipliers
    bank: Dict[str, float] = {}
    for bname, bval in base.items():
        for slabel, smul in SCALAR_MULTIPLIERS:
            if slabel == "1":
                key = bname
            else:
                key = f"{slabel}*{bname}"
            bank[key] = bval * smul
    return bank


def match_against_targets(
    estimate: float,
    targets: Dict[str, float],
    threshold: float = 1e-3,
) -> List[Dict[str, Any]]:
    """Match an estimate against the target constant bank.

    For Level-1 scouting we use a generous threshold (default 1e-3).
    We test the estimate directly, its negative, and its reciprocal
    against every target value.

    Score = -log10(|estimate - target|) (higher = better match).
    """
    hits = []
    if not math.isfinite(estimate) or estimate == 0.0:
        return hits

    # Build candidate values to test: est, -est, 1/est, -1/est
    candidates = [("direct", estimate)]
    candidates.append(("neg", -estimate))
    if abs(estimate) > 1e-15:
        candidates.append(("recip", 1.0 / estimate))
        candidates.append(("neg_recip", -1.0 / estimate))

    seen = set()  # avoid duplicate (target, transform) hits
    for transform, cval in candidates:
        if not math.isfinite(cval):
            continue
        for name, tval in targets.items():
            residual = abs(cval - tval)
            if residual < threshold:
                hit_key = (name, transform)
                if hit_key in seen:
                    continue
                seen.add(hit_key)
                if residual == 0.0:
                    score = 300.0
                else:
                    score = -math.log10(residual)
                hits.append({
                    "target_family": "constant_bank",
                    "target_const": name,
                    "score": round(score, 2),
                    "residual": f"{residual:.6e}",
                    "transform": transform,
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
    program,
    lcm: int,
    shift: dict,
    direction: List[int],
    depth: int,
) -> Optional[Tuple[float, bool]]:
    """
    Run a single CMF walk on GPU via the RNS bytecode engine.

    Args:
        program: Compiled CmfProgram (rationalized, GPU-ready).
        lcm: LCM used for shift denominator clearing.
        shift: {'nums': [dim], 'dens': [dim]} rational shift.
        direction: [dim ints] trajectory direction.
        depth: number of walk steps.

    Returns:
        (estimate, confident) or None if walk diverged.
    """
    from cmf_walk_engine import walk_single
    return walk_single(program, lcm, shift["nums"], shift["dens"], direction, depth)


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
    ap.add_argument("--pre_score", type=float, default=5e-3,
                    help="Threshold for pre-screening hits (5e-3 = wide catch band for Level-1)")
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

    # Compile CMF to GPU-ready bytecode (sympy at startup only)
    from cmf_walk_engine import compile_6f5
    compiled_programs = {}  # cmf_id -> (CmfProgram, lcm)
    for cmf in cmfs:
        cid = cmf.get("cmf_id", "cmf_unknown")
        log(f"  Compiling CMF {cid} to GPU bytecode...")
        prog, lcm = compile_6f5(cmf)
        compiled_programs[cid] = (prog, lcm)
        log(f"    rank={prog.m} dim={prog.dim} instrs={len(prog.instructions)} "
            f"regs={prog.n_reg} consts={len(prog.constants)} lcm={lcm}")

    # Load target constants (expanded bank: 23 base × 10 scalars = ~230 targets)
    targets = load_target_constants()
    log(f"Target constants: {len(targets)} values ({len(targets)//10} base × 10 scalars)"
        f" + neg/recip transforms at threshold={args.pre_score}")

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

    for cmf in cmfs:
        cmf_id = cmf.get("cmf_id", "cmf_unknown")
        program, lcm = compiled_programs[cmf_id]
        log(f"Processing CMF: {cmf_id} (GPU RNS engine, depth={args.depth})")

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

                # ── COMPUTE WALK (GPU RNS kernel) ──
                result = compute_cmf_walk(program, lcm, shift, direction, args.depth)
                if result is None:
                    continue
                estimate, confident = result

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
                        "confident": confident,
                        "shift": {"nums": shift["nums"], "dens": shift["dens"]},
                        "trajectory": {"v": direction},
                    }
                    hits_f.write(json.dumps(hr, separators=(",",":")) + "\n")
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

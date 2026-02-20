
"""
make_level2_tasks_from_shard_hitlist.py

Turn Level-1 shard scouting output into compact Level-2 (depth=2000) job definitions.

Input (under run_dir):
  - results/shard_hitlist.jsonl   (required)
  - results/shard_summary.jsonl   (optional but recommended, used to get best center pairs)

Output:
  - results/level2_jobs.jsonl     (compact job specs; workers enumerate pairs)
  - results/level2_jobs_summary.json (counts by cmf and by target)

Supports two escalation modes:
  1) full_shard_expansion (default)
     - enumerate all (s_idx in shard shift-bin) x (d_idx in shard dir-bin)
  2) local_neighborhood
     - around best (s_idx,d_idx) centers per target inside shard, generate a neighborhood window

Variant-B friendly: no gigantic pair files; workers generate pairs deterministically from the compact job specs.

Usage:
  python make_level2_tasks_from_shard_hitlist.py --run_dir runs/<run_id> \
     --mode full_shard_expansion --depth 2000

Neighborhood mode:
  python make_level2_tasks_from_shard_hitlist.py --run_dir runs/<run_id> \
     --mode local_neighborhood --depth 2000 --rs 128 --rd 2000

Notes:
- Shards are in index-space. Shard geometry is read from manifest.json:
    sharding: S_BIN, D_BIN
    pairing_rule: n_shifts, n_dirs
- shard_id -> (s_shard, d_shard) via N_DIR_SHARDS = N_DIRS//D_BIN
"""

from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows: List[dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",",":")) + "\n")

def shard_to_bins(shard_id: int, N_DIR_SHARDS: int) -> Tuple[int,int]:
    s_shard = shard_id // N_DIR_SHARDS
    d_shard = shard_id % N_DIR_SHARDS
    return s_shard, d_shard

def shard_ranges(s_shard: int, d_shard: int, S_BIN: int, D_BIN: int, N_SHIFTS: int, N_DIRS: int) -> Tuple[Tuple[int,int],Tuple[int,int]]:
    s0 = s_shard * S_BIN
    s1 = min(s0 + S_BIN - 1, N_SHIFTS - 1)
    d0 = d_shard * D_BIN
    d1 = min(d0 + D_BIN - 1, N_DIRS - 1)
    return (s0,s1), (d0,d1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--hitlist", default="results/shard_hitlist.jsonl")
    ap.add_argument("--shard_summary", default="results/shard_summary.jsonl")
    ap.add_argument("--out_jobs", default="results/level2_jobs.jsonl")
    ap.add_argument("--out_summary", default="results/level2_jobs_summary.json")
    ap.add_argument("--mode", choices=["full_shard_expansion","local_neighborhood"], default="full_shard_expansion")
    ap.add_argument("--depth", type=int, default=2000)
    ap.add_argument("--rs", type=int, default=128, help="Neighborhood radius in shift-index (local_neighborhood)")
    ap.add_argument("--rd", type=int, default=2000, help="Neighborhood radius in dir-index (local_neighborhood)")
    ap.add_argument("--max_centers_per_shard", type=int, default=3, help="Use up to this many best centers per shard.")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"Missing manifest.json at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    S_BIN = int(manifest["sharding"]["S_BIN"])
    D_BIN = int(manifest["sharding"]["D_BIN"])
    N_SHIFTS = int(manifest["pairing_rule"]["n_shifts"])
    N_DIRS = int(manifest["pairing_rule"]["n_dirs"])
    N_DIR_SHARDS = N_DIRS // D_BIN

    hitlist_path = os.path.join(run_dir, args.hitlist)
    if not os.path.exists(hitlist_path):
        raise SystemExit(f"Missing shard_hitlist at {hitlist_path}")

    # Optional: load shard_summary best centers (best_by_target has s_idx,d_idx)
    best_centers: Dict[Tuple[str,int,str], Tuple[int,int,float]] = {}
    summary_path = os.path.join(run_dir, args.shard_summary)
    if os.path.exists(summary_path):
        for row in iter_jsonl(summary_path):
            cmf_id = row["cmf_id"]
            shard_id = int(row["shard"]["shard_id"])
            bb = row.get("level1", {}).get("best_by_target", {})
            for tconst, info in bb.items():
                s_idx = info.get("s_idx")
                d_idx = info.get("d_idx")
                score = info.get("score")
                if s_idx is None or d_idx is None:
                    continue
                best_centers[(cmf_id, shard_id, tconst)] = (int(s_idx), int(d_idx), float(score) if score is not None else 0.0)

    jobs = []
    by_cmf = defaultdict(int)
    by_const = defaultdict(int)

    for row in iter_jsonl(hitlist_path):
        cmf_id = row["cmf_id"]
        shard_id = int(row["shard_id"])
        targets = row.get("selected_targets", [])
        s_shard, d_shard = shard_to_bins(shard_id, N_DIR_SHARDS)
        (s0,s1), (d0,d1) = shard_ranges(s_shard, d_shard, S_BIN, D_BIN, N_SHIFTS, N_DIRS)

        if args.mode == "full_shard_expansion":
            job = {
                "cmf_id": cmf_id,
                "level": "level2",
                "depth": args.depth,
                "mode": "full_shard_expansion",
                "shard": {"shard_id": shard_id, "s_shard": s_shard, "d_shard": d_shard},
                "ranges": {"s_idx": [s0, s1], "d_idx": [d0, d1]},
                "targets": targets
            }
            jobs.append(job)
            by_cmf[cmf_id] += 1
            for t in targets:
                by_const[t] += 1
            continue

        # local neighborhood mode: choose up to max_centers_per_shard centers from shard_summary
        centers = []
        for t in targets:
            k = (cmf_id, shard_id, t)
            if k in best_centers:
                centers.append((t, best_centers[k][0], best_centers[k][1], best_centers[k][2]))
        # If no centers found, fallback to shard center
        if not centers:
            centers = [(targets[0] if targets else "unknown", (s0+s1)//2, (d0+d1)//2, 0.0)]
        # sort by score desc
        centers.sort(key=lambda x: -x[3])
        centers = centers[:max(1, args.max_centers_per_shard)]

        # produce one job per center
        for (t, cs, cd, score) in centers:
            job = {
                "cmf_id": cmf_id,
                "level": "level2",
                "depth": args.depth,
                "mode": "local_neighborhood",
                "shard": {"shard_id": shard_id, "s_shard": s_shard, "d_shard": d_shard},
                "center": {"target_const": t, "s_idx": int(cs), "d_idx": int(cd), "score": float(score)},
                "radius": {"rs": int(args.rs), "rd": int(args.rd)},
                "targets": targets
            }
            jobs.append(job)
            by_cmf[cmf_id] += 1
            for tt in targets:
                by_const[tt] += 1

    out_jobs = os.path.join(run_dir, args.out_jobs)
    write_jsonl(out_jobs, jobs)

    summary = {
        "mode": args.mode,
        "depth": args.depth,
        "jobs": len(jobs),
        "by_cmf_jobs": dict(sorted(by_cmf.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_const_jobs": dict(sorted(by_const.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    out_sum = os.path.join(run_dir, args.out_summary)
    ensure_dir(os.path.dirname(out_sum) or ".")
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[level2] wrote", out_jobs, "jobs=", len(jobs))
    print("[level2] wrote", out_sum)

if __name__ == "__main__":
    main()

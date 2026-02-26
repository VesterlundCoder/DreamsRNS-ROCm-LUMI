
"""
postprocess_level1.py

Merge per-rank outputs from Variant-B Level-1 runs and produce:
  - results/hits_level1.jsonl            (merged)
  - results/shard_summary.jsonl          (aggregated)
  - results/miss_intervals.jsonl         (complete-miss shards)
  - results/shard_hitlist.jsonl          (escalation plan: which shards + which consts)

Inputs expected under run_dir:
  results/hits_level1_rank*.jsonl
  progress/shard_stats_rank*.jsonl   (optional; we can compute shard pairs from hits only, but better to have stats)

Note:
- If shard_stats files are sparse snapshots, we don't rely on them for exact pair counts.
- Prefer exact pair counts from a separate "pairs_tested_rank*.jsonl" if you add it later.
- For now: we estimate pairs_tested per shard from (hits + optional stats). You can also pass a required
  pairs_tested file via --pairs_tested_glob.

Recommended practice:
- In workers, maintain exact pairs_tested[(cmf_id, shard_id)] and flush at end of CMF into:
    progress/pairs_tested_rankXXXXX.jsonl
  This postprocess script supports that via --pairs_tested_glob.

Usage:
  python postprocess_level1.py --run_dir runs/<run_id> \
    --min_pairs_per_shard 200 --score_thresh 20.0 \
    --top_shards_per_const 200 --max_consts_per_shard 6

"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Any

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

def concat_jsonl(out_path: str, in_paths: List[str]) -> int:
    ensure_dir(os.path.dirname(out_path) or ".")
    n=0
    with open(out_path, "w") as out:
        for p in sorted(in_paths):
            with open(p, "r") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line+"\n")
                        n+=1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--hits_glob", default="results/hits_level1_rank*.jsonl")
    ap.add_argument("--pairs_tested_glob", default="progress/pairs_tested_rank*.jsonl",
                    help="Optional exact per-shard pair counts dumped by workers at end.")
    ap.add_argument("--min_pairs_per_shard", type=int, default=200)
    ap.add_argument("--score_thresh", type=float, default=20.0,
                    help="Minimum score to consider a hit 'real' for shard selection.")
    ap.add_argument("--top_shards_per_const", type=int, default=200)
    ap.add_argument("--max_consts_per_shard", type=int, default=6)
    ap.add_argument("--escalate_depth", type=int, default=5000)
    ap.add_argument("--escalate_mode", choices=["full_shard_expansion","local_neighborhood"], default="full_shard_expansion")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    results_dir = os.path.join(run_dir, "results")
    progress_dir = os.path.join(run_dir, "progress")
    ensure_dir(results_dir)

    hit_paths = glob.glob(os.path.join(run_dir, args.hits_glob))
    if not hit_paths:
        raise SystemExit(f"No hits files found under {run_dir}/{args.hits_glob}")

    # 1) Merge hits
    merged_hits_path = os.path.join(results_dir, "hits_level1.jsonl")
    n_hits = concat_jsonl(merged_hits_path, hit_paths)

    # 2) Load hits into aggregators
    # Per (cmf_id, shard_id):
    pairs_tested: Dict[Tuple[str,int], int] = defaultdict(int)   # may be overridden by exact file
    targets_run: Dict[Tuple[str,int], set] = defaultdict(set)
    hits_by_target: Dict[Tuple[str,int], Dict[str,int]] = defaultdict(lambda: defaultdict(int))
    best_by_target: Dict[Tuple[str,int], Dict[str,dict]] = defaultdict(dict)

    # Track which shards have any hits >= score_thresh for each const
    shards_by_const: Dict[str, Dict[Tuple[str,int], float]] = defaultdict(dict)  # key=(cmf_id, shard_id)->best_score

    # hits contribute to pairs_tested lower bound (not exact)
    for hr in iter_jsonl(merged_hits_path):
        cmf_id = hr.get("cmf_id")
        pair = hr.get("pair", {})
        shard_id = int(pair.get("shard_id"))
        key = (cmf_id, shard_id)

        # lower bound: at least the number of hits observed
        pairs_tested[key] += 0

        tfam = hr.get("target_family","?")
        tconst = hr.get("target_const","?")
        targets_run[key].add(tfam)
        hits_by_target[key][tconst] += 1

        score = float(hr.get("score", 0.0))
        residual = hr.get("residual", None)

        prev = best_by_target[key].get(tconst)
        if (prev is None) or (score > float(prev.get("score", -1e300))):
            best_by_target[key][tconst] = {
                "score": score,
                "residual": residual,
                "s_idx": pair.get("s_idx"),
                "d_idx": pair.get("d_idx")
            }

        if score >= args.score_thresh:
            prev_best = shards_by_const[tconst].get(key)
            if (prev_best is None) or (score > prev_best):
                shards_by_const[tconst][key] = score

    # 3) Optional: load exact pairs_tested if available
    pt_paths = glob.glob(os.path.join(run_dir, args.pairs_tested_glob))
    if pt_paths:
        exact = defaultdict(int)
        for p in sorted(pt_paths):
            for row in iter_jsonl(p):
                cmf_id = row["cmf_id"]
                shard_id = int(row["shard_id"])
                exact[(cmf_id, shard_id)] += int(row["pairs_tested"])
        pairs_tested = exact  # override with exact counts

    # 4) Build shard_summary + miss_intervals
    shard_summary_rows = []
    miss_rows = []
    # gather all observed keys (from pairs_tested if exact, otherwise from hits only)
    all_keys = set(pairs_tested.keys()) | set(hits_by_target.keys())

    for (cmf_id, shard_id) in sorted(all_keys, key=lambda x: (x[0], x[1])):
        pt = int(pairs_tested.get((cmf_id, shard_id), 0))
        hb = hits_by_target.get((cmf_id, shard_id), {})
        bb = best_by_target.get((cmf_id, shard_id), {})
        tr = sorted(list(targets_run.get((cmf_id, shard_id), set())))

        is_miss = (pt >= args.min_pairs_per_shard) and (sum(hb.values()) == 0)

        row = {
            "cmf_id": cmf_id,
            "shard": {"shard_id": shard_id},
            "coverage": {"pairs_tested": pt},
            "level1": {
                "targets_run": tr,
                "hits_by_target": dict(hb),
                "best_by_target": dict(bb)
            },
            "miss": {
                "is_complete_miss": bool(is_miss),
                "reason": "no hits above threshold" if is_miss else None,
                "targets_tested": len(tr)
            }
        }
        shard_summary_rows.append(row)

        if is_miss:
            miss_rows.append({
                "cmf_id": cmf_id,
                "level": "level1",
                "miss_type": "shard",
                "shard_id": shard_id,
                "targets": tr,
                "pairs_tested": pt
            })

    write_jsonl(os.path.join(results_dir, "shard_summary.jsonl"), shard_summary_rows)
    write_jsonl(os.path.join(results_dir, "miss_intervals.jsonl"), miss_rows)

    # 5) Build shard_hitlist (escalation plan)
    # Select top shards per constant, then within each shard select up to max_consts_per_shard constants.
    selected_shards = defaultdict(lambda: defaultdict(float))  # (cmf_id, shard_id)-> tconst->best_score

    for tconst, shard_map in shards_by_const.items():
        # shard_map: (cmf_id, shard_id)->best_score
        items = sorted(shard_map.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        for (cmf_id, shard_id), best_score in items[:args.top_shards_per_const]:
            selected_shards[(cmf_id, shard_id)][tconst] = best_score

    hitlist_rows = []
    for (cmf_id, shard_id), tmap in sorted(selected_shards.items(), key=lambda x: (x[0][0], x[0][1])):
        # pick top constants within shard by score
        consts_sorted = sorted(tmap.items(), key=lambda kv: -kv[1])[:args.max_consts_per_shard]
        selected_consts = [c for c,_ in consts_sorted]
        best_score = consts_sorted[0][1] if consts_sorted else None

        hitlist_rows.append({
            "cmf_id": cmf_id,
            "shard_id": shard_id,
            "selected_targets": selected_consts,
            "reason": {"best_score": best_score, "scores": {c:s for c,s in consts_sorted}},
            "next": {"level": "level2", "depth": args.escalate_depth, "mode": args.escalate_mode}
        })

    write_jsonl(os.path.join(results_dir, "shard_hitlist.jsonl"), hitlist_rows)

    print(f"[postprocess] merged hits: {n_hits} -> {merged_hits_path}")
    print(f"[postprocess] shard_summary rows: {len(shard_summary_rows)}")
    print(f"[postprocess] complete-miss shards: {len(miss_rows)}")
    print(f"[postprocess] shard_hitlist rows: {len(hitlist_rows)}")

    if not pt_paths:
        print("[postprocess] NOTE: no exact pairs_tested files found. "
              "For exact miss intervals, add worker flush to progress/pairs_tested_rankXXXXX.jsonl")

if __name__ == "__main__":
    main()

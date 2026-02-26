# 6F5 Level-2 Sweep — LUMI Container Deployment

## Overview

Level-2 deep sweep for the **6F5(1,1,1,1,1,1; 2,2,2,2,2; 1)** CMF.
Takes the `shard_hitlist.jsonl` from Level-1 and runs full shard expansion
at depth 5000 on the most promising shards.

**Prerequisites:** Level-1 must have completed and you must have analyzed
the results before running Level-2.

## Files

| File | Description |
|------|-------------|
| `worker_level2_from_jobs.py` | Level-2 worker (one job per array task) |
| `make_level2_tasks_from_shard_hitlist.py` | Generate Level-2 job specs from Level-1 hitlist |
| `submit_level2.sh` | Submit script (C→D Slurm chain) |
| `sbatch_make_level2_jobs.sh` | Sbatch for job generation |
| `sbatch_launch_level2_array.sh` | Sbatch for array launch |
| `sbatch_level2_worker.sh` | Sbatch for individual array workers |

## Step-by-Step

### 1. Upload scripts to LUMI

```bash
rsync -avz 6f5sweep_level2/ \
  lumi:/projappl/project_465002669/$(whoami)/6f5sweep_level2/
```

### 2. Submit (after Level-1 analysis)

```bash
ssh lumi
chmod +x /projappl/project_465002669/$(whoami)/6f5sweep_level2/*.sh

# Point L1_RUN_DIR to your completed Level-1 run
L1_RUN_DIR=/scratch/project_465002669/$(whoami)/runs/<LEVEL1_RUN_ID> \
  bash /projappl/project_465002669/$(whoami)/6f5sweep_level2/submit_level2.sh
```

### 3. Monitor

```bash
squeue -u $(whoami)
```

### 4. Download Level-2 results

```bash
scp -r lumi:/scratch/project_465002669/$(whoami)/runs/<RUN_ID>/results/hits_depth*.jsonl ./level2_results/
```

## Escalation Modes

- **full_shard_expansion** (default): enumerate all (s_idx, d_idx) in the shard bin
- **local_neighborhood**: enumerate around the best center ± radius

The mode is set in `submit_level2.sh` → `sbatch_make_level2_jobs.sh`.

## Dependency on Level-1

Level-2 reads these files from the Level-1 run directory:
- `results/shard_hitlist.jsonl` — which shards to escalate
- `results/shard_summary.jsonl` — best centers per shard (for neighborhood mode)
- `manifest.json` — pairing rule + sharding parameters
- Data files (shifts, dirs, cmfs) — auto-discovered from parent directory

## Container

Uses the **same container** as Level-1 (`6f5-level1.sif`). No separate
container build needed.

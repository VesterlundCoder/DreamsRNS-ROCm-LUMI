# 6F5 Level-1 Sweep — LUMI Container Deployment

## Overview

Level-1 Variant-B shard sweep for the **6F5(1,1,1,1,1,1; 2,2,2,2,2; 1)** CMF.
Searches for ζ(3), ζ(5), ζ(7), ζ(9), Catalan's G, and π.

- **8,001 shifts × 4,883 dirs/shift = ~39M pairs** at depth 2000
- 1 node, 8 GPUs (MI250X), 1 rank per GPU
- Output: per-rank hit files + merged shard summary + escalation hitlist

## Files

| File | Description |
|------|-------------|
| `worker_variantB_skeleton.py` | Level-1 worker (one rank per GPU) |
| `postprocess_level1.py` | Merge + analyze results → shard_hitlist |
| `submit_level1.sh` | Submit script (A→B Slurm chain) |
| `sbatch_level1.sh` | Sbatch for the sweep |
| `sbatch_postprocess.sh` | Sbatch for postprocessing |
| `6f5_level1.def` | Singularity container definition |
| `cmfs_level1.jsonl` | CMF definition (1 line) |
| `tasks_6f5_zeta5_shifts.jsonl` | 8,001 shifts |
| `tasks_6f5_zeta5_dirs.jsonl` | 100,011 directions |
| `cmf_6f5.json` | Full CMF definition (reference) |
| `sweep_meta.json` | Sweep configuration metadata |

## Step-by-Step: Build, Upload, Run

### 1. Build the container (on a machine with Singularity/Apptainer)

```bash
# On your local machine or a build server (NOT on LUMI login nodes)
cd 6f5sweep_level1/
singularity build 6f5-level1.sif 6f5_level1.def
```

The resulting `.sif` file is ~400–600 MB.

### 2. Upload to LUMI

```bash
# Upload container
scp 6f5-level1.sif lumi:/projappl/project_465002669/containers/

# Upload scripts + data
rsync -avz --exclude='*.sif' 6f5sweep_level1/ \
  lumi:/projappl/project_465002669/$(whoami)/6f5sweep_level1/
```

### 3. Submit the sweep

```bash
ssh lumi

# Make scripts executable
chmod +x /projappl/project_465002669/$(whoami)/6f5sweep_level1/*.sh

# Submit (uses defaults from the script)
cd /projappl/project_465002669/$(whoami)/6f5sweep_level1
bash submit_level1.sh
```

To override defaults:

```bash
ACCOUNT=project_465002669 \
PARTITION=standard-g \
L1_TIME=04:00:00 \
bash submit_level1.sh
```

### 4. Monitor

```bash
squeue -u $(whoami)
# When done, check output:
ls /scratch/project_465002669/$(whoami)/runs/*/results/
```

### 5. Download results for analysis

```bash
# On your local machine
scp -r lumi:/scratch/project_465002669/$(whoami)/runs/<RUN_ID>/results/ ./level1_results/
```

Key output files:
- `results/hits_level1.jsonl` — all hits (merged from per-rank files)
- `results/shard_summary.jsonl` — per-shard statistics
- `results/shard_hitlist.jsonl` — **input for Level-2** (shards to escalate)
- `results/miss_intervals.jsonl` — shards with zero hits

## After Analysis → Level-2

Once you've analyzed the Level-1 results and are satisfied, proceed to
`6f5sweep_level2/` which reads the `shard_hitlist.jsonl` and runs full
shard expansion at depth 5000.

## Variant B Pairing Rule

```
dirs_per_shift = 4883
stride = 99991
mix_a = 104729, mix_b = 12345

For shift s:
  base = (104729 * s + 12345) % 100011
  d[k] = (base + k * 99991) % 100011,  k = 0..4882
```

Coverage: ~4.88% of the full 800M pair space.
Sharding: S_BIN=64, D_BIN=1000 → 12,726 shards.

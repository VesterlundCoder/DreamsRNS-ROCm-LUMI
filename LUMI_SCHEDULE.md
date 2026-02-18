# LUMI Exhaust Sweep Schedule

Full-dimension CMF sweep with all trajectories and shifts per CMF.
No staging — run everything. 50,000 CMFs across 5 pFq families.

---

## CMF Inventory

| Family | Rank | Matrix | Dim | Axes | Count | Files |
|--------|------|--------|-----|------|-------|-------|
| 2F2 | 3 | 3×3 | 4 | x0,x1,y0,y1 | 10,000 | `sweep_data/2F2/2F2_part{00-09}.jsonl` |
| 3F2 | 4 | 4×4 | 5 | x0,x1,x2,y0,y1 | 10,000 | `sweep_data/3F2/3F2_part{00-09}.jsonl` |
| 3F3 | 4 | 4×4 | 6 | x0,x1,x2,y0,y1,y2 | 10,000 | `sweep_data/3F3/3F3_part{00-09}.jsonl` |
| 4F3 | 5 | 5×5 | 7 | x0,x1,x2,x3,y0,y1,y2 | 10,000 | `sweep_data/4F3/4F3_part{00-09}.jsonl` |
| 5F4 | 6 | 6×6 | 9 | x0..x4,y0..y3 | 10,000 | `sweep_data/5F4/5F4_part{00-09}.jsonl` |
| **Total** | | | | | **50,000** | **50 files** |

Each file contains 1,000 CMFs. Parameter pool: integers ±5, halves ±11/2,
thirds ±8/3, quarters ±10/4, sixths ±2. All unique by SHA-256 hash on
sorted canonical parameters.

---

## Trajectory Coverage

Primitive integer vectors in Z^d with L∞ ≤ k_max, canonical normalization
(gcd-reduced, first nonzero positive).

| Family | Dim | k_max | Unique trajectories | File |
|--------|-----|-------|---------------------|------|
| 2F2 | 4 | 3 | **1,120** | `sweep_data/trajectories/dim4_trajectories.json` |
| 3F2 | 5 | 3 | **8,161** | `sweep_data/trajectories/dim5_trajectories.json` |
| 3F3 | 6 | 2 | **7,448** | `sweep_data/trajectories/dim6_trajectories.json` |
| 4F3 | 7 | 2 | **37,969** | `sweep_data/trajectories/dim7_trajectories.json` |
| 5F4 | 9 | 1 | **9,841** | `sweep_data/trajectories/dim9_trajectories.json` |

---

## Shift Coverage

Sobol low-discrepancy sequence quantized to rationals with denominators
∈ {2, 3, 4, 6, 8}, centered in [-1/2, 1/2)^d, deduplicated.

| Family | Dim | Shifts | File |
|--------|-----|--------|------|
| 2F2 | 4 | **512** | `sweep_data/shifts/dim4_shifts.json` |
| 3F2 | 5 | **512** | `sweep_data/shifts/dim5_shifts.json` |
| 3F3 | 6 | **1,024** | `sweep_data/shifts/dim6_shifts.json` |
| 4F3 | 7 | **1,024** | `sweep_data/shifts/dim7_shifts.json` |
| 5F4 | 9 | **1,024** | `sweep_data/shifts/dim9_shifts.json` |

---

## Compute Budget (Full Exhaust)

| Family | CMFs | Traj | Shifts | Runs/CMF | Total runs | GPU-hours |
|--------|------|------|--------|----------|------------|-----------|
| 2F2 | 10,000 | 1,120 | 512 | 573,440 | 5.7B | 797 |
| 3F2 | 10,000 | 8,161 | 512 | 4,178,432 | 41.8B | 5,804 |
| 3F3 | 10,000 | 7,448 | 1,024 | 7,626,752 | 76.3B | 10,593 |
| 4F3 | 10,000 | 37,969 | 1,024 | 38,880,256 | 388.8B | 53,945 |
| 5F4 | 10,000 | 9,841 | 1,024 | 10,077,184 | 100.8B | 13,996 |
| **Total** | **50,000** | | | | **613.4B** | **85,135** |

At 0.5 ms/run on MI250X, depth=2000, K=32:
- **85,135 GPU-hours total**
- **10,642 node-hours** (8 GPUs/node)
- **~444 wall-days** on 1 node

### Practical scheduling (multi-node)

| Nodes | Wall time |
|-------|-----------|
| 1 | ~444 days |
| 10 | ~44 days |
| 50 | ~9 days |
| 100 | ~4.5 days |

Recommended: **run by family, smallest first**.

### Priority order (by compute cost)

| Priority | Family | GPU-hours | % of total |
|----------|--------|-----------|------------|
| 1 | 2F2 | 797 | 0.9% |
| 2 | 3F2 | 5,804 | 6.8% |
| 3 | 3F3 | 10,593 | 12.4% |
| 4 | 5F4 | 13,996 | 16.4% |
| 5 | 4F3 | 53,945 | 63.4% |

4F3 dominates due to 37,969 trajectories in dim=7 (k_max=2).

### Per-file compute (1,000 CMFs per file)

| Family | Runs per file | GPU-hours per file | Wall hours (8 GPU) |
|--------|---------------|--------------------|--------------------|
| 2F2 | 573M | 80 | 10 |
| 3F2 | 4.2B | 580 | 73 |
| 3F3 | 7.6B | 1,059 | 132 |
| 4F3 | 38.9B | 5,395 | 674 |
| 5F4 | 10.1B | 1,400 | 175 |

Each `.jsonl` file = 1 independent job. Submit in parallel across nodes.

---

## LUMI Job Configuration

```
Partition:    small-g (or standard-g for multi-node)
GPUs/node:    8 (MI250X)
CPUs/GPU:     7
Memory:       480 GB
Container:    dreams_rocm.sif (Singularity)
```

---

## RNS Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| K | 32 | 31-bit primes → 992-bit capacity |
| depth | 2000 | Walk steps |
| delta_threshold | −2.0 | Reporting cutoff |

---

## File Layout

```
DreamsRNS-ROCm-LUMI/
├── sweep_data/
│   ├── sweep_manifest.json             # Master manifest with all counts
│   ├── 2F2/
│   │   ├── 2F2_part00.jsonl            # 1,000 CMF specs
│   │   ├── ...
│   │   └── 2F2_part09.jsonl
│   ├── 3F2/
│   │   └── 3F2_part{00-09}.jsonl
│   ├── 3F3/
│   │   └── 3F3_part{00-09}.jsonl
│   ├── 4F3/
│   │   └── 4F3_part{00-09}.jsonl
│   ├── 5F4/
│   │   └── 5F4_part{00-09}.jsonl
│   ├── trajectories/
│   │   ├── dim4_trajectories.json      # 1,120 vectors
│   │   ├── dim5_trajectories.json      # 8,161 vectors
│   │   ├── dim6_trajectories.json      # 7,448 vectors
│   │   ├── dim7_trajectories.json      # 37,969 vectors
│   │   └── dim9_trajectories.json      # 9,841 vectors
│   └── shifts/
│       ├── dim4_shifts.json            # 512 rational shifts
│       ├── dim5_shifts.json            # 512 rational shifts
│       ├── dim6_shifts.json            # 1,024 rational shifts
│       ├── dim7_shifts.json            # 1,024 rational shifts
│       └── dim9_shifts.json            # 1,024 rational shifts
├── results/
│   └── {family}/{part}/                # Per-job output
└── dreams_rocm/
    ├── cmf_generator.py                # Generic pFq generator
    ├── exhaust.py                      # Trajectory + shift generators
    └── ...
```

---

## Workflow

```bash
# 1. Generate all CMFs (one-time, on login node — ~30s)
python -m dreams_rocm.cmf_generator --all --count 10000 -o sweep_data

# 2. Generate trajectories + shifts (one-time — ~1s)
python scripts/generate_sweep_data.py --output-dir sweep_data

# 3. Submit jobs per file (each file = 1 independent job)
for part in sweep_data/2F2/*.jsonl; do
    sbatch scripts/sbatch_1node_8gpu.sh --input "$part" --dim 4
done

# 4. Collect results
python scripts/collect_results.py results/ > full_report.csv
```

---

## Scaling Notes

- **4F3 dominates**: 63% of total compute due to 37,969 trajectories in dim=7
- **Each file is independent**: trivially parallelizable across nodes
- **2F2 completes fast**: 10 hours per file on 1 node — good for early validation
- **If budget is tight**: run 2F2 + 3F2 + 3F3 first (20% of total compute, 30,000 CMFs)
- **Multi-node**: Use standard-g partition, 1 file per node, up to 50 simultaneous jobs

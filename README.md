# Dreams-RNS-ROCm-LUMI

**GPU-accelerated Ramanujan Dreams pipeline** using RNS (Residue Number System) arithmetic on **AMD MI250X GPUs** via ROCm/HIP on the **LUMI-G supercomputer**.

Runs **8 GPUs simultaneously** on a single LUMI-G node with 1 MPI rank per GPU.

## Overview

This pipeline searches for exceptional rational approximations to mathematical constants (ζ(3), ζ(5), ζ(7), π, etc.) using Continued Matrix Fractions (CMFs). The computation is done entirely in **exact integer arithmetic** via the Residue Number System, avoiding floating-point precision loss.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LUMI-G Node (1 node, 8 GPUs)                 │
│                                                                 │
│  Rank 0 ──► GPU 0 ──► CMF family A                             │
│  Rank 1 ──► GPU 1 ──► CMF family B                             │
│  Rank 2 ──► GPU 2 ──► CMF family C                             │
│  ...                                                            │
│  Rank 7 ──► GPU 7 ──► CMF family H                             │
│                                                                 │
│  Each rank:                                                     │
│    for each CMF assigned to this rank:                          │
│      for each trajectory (1000):                                │
│        for each shift (100):                                    │
│          ┌─────────┐    ┌──────┐    ┌───────┐    ┌──────────┐  │
│          │ COMPILE │───►│ EVAL │───►│ WALK  │───►│  SCORE   │  │
│          │ CMF→BC  │    │ Axes │    │ P@M_t │    │  Delta   │  │
│          └─────────┘    └──────┘    └───────┘    └──────────┘  │
│                                                    │            │
│                                        delta > 0 ──► LOG       │
│                                        delta > T ──► ESCALATE  │
│                                                      to CPU    │
│                                                      Full CRT  │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **CMF Compile** — Symbolic matrix → bytecode (CPU, offline)
2. **Axis Evaluation** — Evaluate matrix entries mod each prime (GPU/CPU)
3. **Walk** — Iterated matrix product P = P @ M_t in RNS (GPU/CPU)
4. **Delta Proxy** — Partial CRT → float64 ratio → Dreams delta
5. **Triage** — delta > 0 → log; delta > threshold → escalate to CPU full CRT

### Test Campaign 1 Configuration

| Parameter | Value |
|-----------|-------|
| GPUs | 8 (1 LUMI-G node) |
| Shifts per CMF | 100 |
| Trajectories per shift | 1,000 |
| Total per CMF | 100,000 evaluations |
| RNS primes (K) | 64 (1,984 bits) |
| Walk depth | 2,000 steps |
| Targets | ζ(3), ζ(5), ζ(7) |

---

## Quick Start on LUMI

### 1. Clone and set up environment

```bash
# On LUMI login node
cd /projappl/$LUMI_PROJECT
git clone <your-repo-url> Dreams-RNS-ROCm-LUMI
cd Dreams-RNS-ROCm-LUMI

# Load modules
source env/lumi_modules.sh

# Create virtual environment
python3 -m venv --system-site-packages $HOME/dreams-venv
source $HOME/dreams-venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r env/requirements.txt
```

### 2. Validate the installation

```bash
# On login node (GPU checks will be skipped)
python scripts/validate_rocm_rns.py

# On a compute node (full validation including GPU)
srun --partition=small-g --nodes=1 --ntasks=1 --gpus=1 \
     --time=00:10:00 --account=project_XXXXXXX \
     python scripts/validate_rocm_rns.py
```

### 3. Preview the task list

```bash
python scripts/make_tasks.py --config configs/lumi_1node_8gpu.yaml
```

### 4. Submit the job

```bash
# Edit the account in the sbatch script first:
# Change --account=project_465001234 to your project ID

sbatch scripts/sbatch_1node_8gpu.sh
```

### 5. Check results

```bash
# Job output
cat dreams-rns_<JOBID>.out

# Results directory
ls /scratch/$LUMI_PROJECT/dreams_runs/run_<timestamp>_<JOBID>/

# Files produced:
#   manifest.json              - Run metadata
#   results_rank{0-7}.jsonl    - Per-rank results (delta > 0 only)
#   positives_rank{0-7}.jsonl  - Escalated positive hits
#   metrics_rank{0-7}.jsonl    - Timing and performance data
#   summary.json               - Global summary
```

---

## Repository Structure

```
Dreams-RNS-ROCm-LUMI/
├── README.md                           # This file
├── pyproject.toml                      # Python package config
├── env/
│   ├── lumi_modules.sh                 # LUMI module loads for ROCm + Python
│   ├── requirements.txt                # Core Python dependencies
│   └── requirements-rocm.txt           # Optional ROCm-specific deps
├── configs/
│   ├── lumi_1node_8gpu.yaml            # Default 1-node/8-GPU config
│   └── triage.yaml                     # Delta triage thresholds
├── scripts/
│   ├── sbatch_1node_8gpu.sh            # SLURM batch script
│   ├── run_mpi_sweep.py                # MPI orchestrator (main entry)
│   ├── make_tasks.py                   # Task list generator
│   └── validate_rocm_rns.py            # Installation validation
├── dreams_rocm/
│   ├── __init__.py
│   ├── runner.py                       # CPU walk execution (fallback)
│   ├── gpu_runner.py                   # GPU walk via native RNS-ROCm library
│   ├── cmf_compile.py                  # CMF → bytecode compiler
│   ├── trajectories.py                 # Trajectory generation + normalization
│   ├── shifts.py                       # Shift generation
│   ├── logging.py                      # Structured JSONL logging
│   ├── crt/
│   │   ├── __init__.py
│   │   ├── partial_crt.py              # Fast partial CRT for delta proxy
│   │   ├── full_crt_cpu.py             # High-precision CPU CRT verification
│   │   └── delta_targets.py            # Target constants (ζ(3), ζ(5), ζ(7)...)
│   └── rns/
│       ├── __init__.py
│       ├── reference.py                # Pure-Python RNS reference
│       ├── bindings.py                 # ctypes bindings to RNS-ROCm C++ lib
│       └── tests/
│           ├── test_rns_ops.py         # RNS arithmetic vs big-int reference
│           ├── test_partial_crt.py     # Partial CRT correctness
│           └── test_gpu_smoke.py       # GPU smoke test
├── rns_rocm_lib/                       # RNS-ROCm C++ library (with bug fixes)
│   ├── CMakeLists.txt                  # CMake build (targets gfx90a)
│   ├── include/rns/                    # C++ headers
│   │   ├── barrett.h                   # Barrett reduction (FIXED for HIP)
│   │   ├── config.h, modops.h, ...     # Core headers
│   │   ├── rns_walk.h                  # Fused walk kernel interface
│   │   └── rns_eval.h                  # Bytecode evaluator interface
│   ├── src/                            # C++ source
│   │   ├── hip/                        # HIP GPU kernels (.hip files)
│   │   │   ├── rns_walk_fused.hip      # Main fused walk kernel
│   │   │   ├── rns_eval_bytecode.hip   # GPU bytecode evaluator
│   │   │   ├── rns_kernels.hip         # Elementwise RNS ops
│   │   │   └── rns_matmul_m{4,6,8,10}.hip  # Specialized GEMM kernels
│   │   ├── c_api.cpp                   # C ABI wrapper for Python ctypes
│   │   ├── crt.cpp                     # CRT + BigInt implementation
│   │   └── ...
│   └── python/                         # pybind11 bindings (alternative)
└── outputs/                            # Created at runtime on /scratch
```

---

## Detailed Setup Guide

### LUMI Account and Project

You need:
- A LUMI account with GPU allocation
- A project ID (e.g., `project_465001234`)
- Access to `/scratch/<project>/` for output

Set your project in the sbatch script:
```bash
#SBATCH --account=project_XXXXXXX
```

### Module Environment

The `env/lumi_modules.sh` script loads:

| Module | Purpose |
|--------|---------|
| `LUMI/23.09` | LUMI software stack |
| `partition/G` | GPU partition |
| `PrgEnv-cray` | Cray programming environment |
| `craype-accel-amd-gfx90a` | MI250X target |
| `rocm/5.6.1` | ROCm runtime + hipcc |
| `cray-python/3.10.10` | Python with numpy, scipy, mpi4py |

### Building the RNS-ROCm Native Library (Optional)

For maximum performance, build the C++ RNS-ROCm library:

```bash
# On a LUMI-G compute node (needs hipcc)
srun --partition=small-g --nodes=1 --ntasks=1 --gpus=1 \
     --time=00:30:00 --account=project_XXXXXXX bash << 'EOF'

source env/lumi_modules.sh

# Clone RNS-ROCm if not already present
# cp -r /path/to/RNS-ROCm-main ./rns_rocm_lib

mkdir -p rns_rocm_lib/build && cd rns_rocm_lib/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DRNS_ENABLE_GPU=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_HIP_COMPILER=hipcc
cmake --build . -j 16
ctest --output-on-failure

EOF
```

Point the Python bindings to the built library:
```bash
export RNS_ROCM_LIB=/path/to/rns_rocm_lib/build
```

### Partitions

| Partition | Nodes | Billing | Use case |
|-----------|-------|---------|----------|
| `small-g` | 1-4 | Per-GPU | Development and test runs |
| `standard-g` | 4+ | Full node | Production campaigns |

For Test Campaign 1 (1 node), use `small-g`.

---

## Configuration

### `configs/lumi_1node_8gpu.yaml`

Key sections:

```yaml
# RNS configuration
rns:
  K: 64           # 64 primes × 31 bits = 1984-bit capacity
  K_small: 6      # Primes for fast partial CRT

# Walk parameters
walk:
  depth: 2000     # Walk steps per evaluation
  snapshot_depths: [200, 2000]

# Test 1 sweep
shifts:
  count: 100      # Shifts per CMF
trajectories:
  count: 1000     # Unique normalized trajectories

# Triage
triage:
  delta_threshold: 0.0   # Log when delta > 0
```

### Adding New CMF Families

Add entries to `cmf_families` in the config:

```yaml
cmf_families:
  - name: "My_Custom_CMF"
    matrix:
      "0,0": "n**2 + 1"
      "0,1": "n"
      "1,0": "-(n+1)**4"
      "1,1": "6*n**2 + 6*n + 1"
    m: 2
    dim: 1
    axis_names: ["n"]
    directions: [1]
    target: "zeta3"
```

---

## Output Schema

### `manifest.json`
```json
{
  "run_id": "run_20260206_143000_12345",
  "timestamp": "2026-02-06T14:30:00+0100",
  "git_commit": "abc123...",
  "config_hash": "f3a1b2c3d4e5f6a7",
  "node_name": "nid001234",
  "job_id": "12345",
  "rocm_version": "5.6.1",
  "n_ranks": 8,
  "config": { ... }
}
```

### `results_rank{N}.jsonl` (delta > 0 only)
```json
{"cmf_idx": 0, "cmf_name": "Apery_Zeta3_2x2", "shift": [42], "depth": 2000,
 "delta": 1.234, "log_q": 456.7, "traj_id": 55, "traj_dir": [1, 3],
 "target": "zeta3", "gpu_id": 0, "rank": 0, "timestamp": 1707225600.0}
```

### `positives_rank{N}.jsonl` (escalated)
```json
{"cmf_idx": 0, "cmf_name": "Apery_Zeta3_2x2", "shift": [42], "depth": 2000,
 "delta": 2.5, "log_q": 456.7, "traj_id": 55, "traj_dir": [1, 3],
 "verifications": [{"target": "zeta3", "delta": 2.5, "decision": "positive"}],
 "gpu_id": 0, "rank": 0, "timestamp": 1707225600.0}
```

---

## Implementation Notes

### RNS (Residue Number System)

All integer computations use RNS representation with K = 64 coprime 31-bit primes, giving **1,984-bit integer capacity**. Operations per prime are independent — ideal for GPU parallelism.

### Trajectory Normalization

Trajectories (dn, dk) are normalized to avoid duplicates:
1. Divide by gcd(|dn|, |dk|)
2. Prefer dn > 0; if dn = 0, prefer dk > 0
3. Uniqueness enforced via hash set

### Two-Stage CRT

1. **Partial CRT** (K_small = 6 primes): Quick float64 delta estimate
2. **Full CRT** (all K primes): Exact big-int reconstruction + mpmath verification

Only positives (delta > threshold) are escalated to the expensive full CRT stage.

### Dreams Delta Formula

```
delta = -(1 + log|err| / log|q|)
```
where `err = |p/q - target|`. Higher delta = better convergence rate. Delta > 0 means the approximation converges faster than expected.

---

## Running Tests

```bash
# Unit tests (no GPU required)
python -m pytest dreams_rocm/rns/tests/ -v

# Individual test suites
python -m pytest dreams_rocm/rns/tests/test_rns_ops.py -v
python -m pytest dreams_rocm/rns/tests/test_partial_crt.py -v

# GPU smoke test (on compute node)
srun -n1 --gpus=1 python -m pytest dreams_rocm/rns/tests/test_gpu_smoke.py -v

# Full validation
python scripts/validate_rocm_rns.py
```

---

## Troubleshooting

### Job never starts
- Check `--cpus-per-task` does not exceed 56/8 = 7 per rank
- Verify project has GPU allocation: `lumi-allocations`

### "No module named mpi4py"
- Ensure `cray-python` is loaded: `module load cray-python/3.10.10`
- Or install in venv: `pip install mpi4py`

### GPU not detected
- Verify ROCm modules are loaded: `module list | grep rocm`
- Check `ROCR_VISIBLE_DEVICES` is set (the sbatch script handles this)
- Run `rocm-smi` on compute node to verify GPU access

### Out of memory on GPU
- Reduce `K` (primes) or `B` (batch size) in config
- Reduce `walk.depth`

### Slow performance
- Build the native RNS-ROCm library (see Building section)
- The pure-Python CPU fallback is ~1000x slower than GPU
- Use `standard-g` partition for full-node allocation (better NUMA binding)

---

## RNS-ROCm Library Bug Fixes

The bundled `rns_rocm_lib/` includes the following fixes applied to the original RNS-ROCm library:

### 1. Barrett Reduction — `__int128` not supported in HIP device code

**File**: `rns_rocm_lib/include/rns/barrett.h`

**Problem**: The original `barrett_reduce_u64()` function used `unsigned __int128` for the 128-bit multiply `x * mu`. This type is **not supported in HIP device code** on AMD GPUs — compilation would fail with `hipcc` when targeting gfx90a.

**Fix**: Introduced a `mulhi64()` helper that dispatches to `__umul64hi()` (HIP device intrinsic) on the GPU and `unsigned __int128` on the host:

```cpp
RNS_HOST_DEVICE inline u64 mulhi64(u64 a, u64 b) {
#if defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__)
  return __umul64hi(a, b);  // AMD GPU intrinsic
#else
  unsigned __int128 prod = (unsigned __int128)a * b;
  return (u64)(prod >> 64);
#endif
}
```

### 2. GPU Architecture Target

**File**: `rns_rocm_lib/CMakeLists.txt`

**Change**: Default `HIP_ARCHITECTURES` changed from `"gfx90a;gfx908;gfx1030"` to `"gfx90a"` only (LUMI-G MI250X). Overridable via `-DCMAKE_HIP_ARCHITECTURES=...`.

### 3. C ABI Wrapper

**File**: `rns_rocm_lib/src/c_api.cpp` (new)

Added a flat C API wrapper (`extern "C"`) for all key library functions, enabling direct `ctypes` access from Python without requiring pybind11 at build time. Functions include `rns_create_context`, `rns_walk_fused`, `rns_crt_reconstruct`, etc.

---

## Roadmap

- [x] Pure-Python CPU pipeline with RNS arithmetic
- [x] MPI 1-node/8-GPU orchestration
- [x] Structured JSONL logging with triage
- [x] Trajectory normalization and deduplication
- [x] Partial CRT delta proxy
- [x] Full CRT CPU verification with mpmath
- [x] RNS-ROCm library integrated with bug fixes (barrett.h, gfx90a)
- [x] C ABI wrapper + ctypes bindings for native library
- [x] Build script for LUMI (scripts/build_rns_lib.sh)
- [ ] Full GPU-accelerated walk via native library on LUMI
- [ ] Multi-node scaling (N nodes × 8 GPUs)
- [ ] Parquet output format
- [ ] Dynamic work-queue load balancing

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| mpi4py | ≥3.1 | MPI for multi-GPU |
| sympy | ≥1.12 | CMF symbolic compilation |
| pyyaml | ≥6.0 | Configuration files |
| mpmath | ≥1.3 | High-precision verification |

## License

MIT License

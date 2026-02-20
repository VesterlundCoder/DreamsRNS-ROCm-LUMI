#!/usr/bin/env bash
#SBATCH --job-name=cmf_L1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- LUMI-G typical mapping: 8 GPUs per node ---
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpu-bind=closest
#SBATCH --mem=0

# NOTE: account/partition/time passed from submit script; can be hardcoded here too.

set -euo pipefail

: "${RUN_DIR:?Need RUN_DIR}"
: "${SHIFTS_JSONL:?Need SHIFTS_JSONL}"
: "${DIRS_JSONL:?Need DIRS_JSONL}"
: "${CMFS_JSONL:?Need CMFS_JSONL}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/progress"

# ---- Modules / env (adjust to your LUMI software stack) ----
module purge
# Example; adjust if your site uses different module names/versions:
module load LUMI/23.09
module load cray-python
# If you use a venv:
# source $PWD/.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# ROCm/NCCL-ish stability knobs (harmless if unused)
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_USER_DB_PATH="${RUN_DIR}/miopen-db"
export MIOPEN_CACHE_DIR="${RUN_DIR}/miopen-cache"

echo "[L1] RUN_DIR=${RUN_DIR}"
echo "[L1] SLURM_NTASKS=${SLURM_NTASKS} GPUs/node=${SLURM_GPUS_PER_NODE}"

# ---- Your actual worker ----
# Replace with your integrated worker entrypoint (mpi4py or pure multi-proc).
# Pattern: one rank per GPU.
srun --ntasks="${SLURM_NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python worker_variantB_skeleton.py \
    --run_dir "${RUN_DIR}" \
    --cmfs_jsonl "${CMFS_JSONL}" \
    --shifts_jsonl "${SHIFTS_JSONL}" \
    --dirs_jsonl "${DIRS_JSONL}" \
    --rank "${SLURM_PROCID}" \
    --world "${SLURM_NTASKS}" \
    --depth 2000 \
    --dirs_per_shift 4883 \
    --stride 99991 \
    --mix_a 104729 \
    --mix_b 12345 \
    --S_BIN 64 --D_BIN 1000

#!/usr/bin/env bash
#SBATCH --job-name=cmf_L2
#SBATCH --output=%x_%j_%a.out
#SBATCH --error=%x_%j_%a.err

# --- LUMI-G: 1 GPU per array task ---
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0

# NOTE: account/partition/time/array passed from launch script

set -euo pipefail

: "${RUN_DIR:?Need RUN_DIR}"

module purge
module load LUMI/23.09
module load cray-python
# source $PWD/.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

JOB_IDX="${SLURM_ARRAY_TASK_ID}"
echo "[L2-worker] RUN_DIR=${RUN_DIR} JOB_IDX=${JOB_IDX}"

python worker_level2_from_jobs.py \
  --run_dir "${RUN_DIR}" \
  --job_index "${JOB_IDX}" \
  --S_BIN 64 --D_BIN 1000

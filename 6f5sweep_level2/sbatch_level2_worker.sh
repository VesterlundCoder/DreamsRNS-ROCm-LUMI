#!/bin/bash -l
#SBATCH --job-name=6f5_L2
#SBATCH --output=%x_%j_%a.out
#SBATCH --error=%x_%j_%a.err

# --- LUMI-G: 1 GPU per array task ---
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0

set -euo pipefail

: "${L1_RUN_DIR:?Need L1_RUN_DIR}"
: "${WORK_DIR:?Need WORK_DIR}"
: "${CONTAINER:?Need CONTAINER}"

# L1_DATA_DIR: where the level-1 data files live (shifts, dirs, cmfs)
# Defaults to the level-1 WORK_DIR sibling (same project directory)
L1_DATA_DIR="${L1_DATA_DIR:-/projappl/${SLURM_JOB_ACCOUNT}/$(whoami)/6f5sweep_level1}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

JOB_IDX="${SLURM_ARRAY_TASK_ID}"
echo "[L2-worker] L1_RUN_DIR=${L1_RUN_DIR} JOB_IDX=${JOB_IDX} L1_DATA_DIR=${L1_DATA_DIR}"

singularity exec -B "${WORK_DIR}" -B "${L1_RUN_DIR}" -B "${L1_DATA_DIR}" "${CONTAINER}" \
  python "${WORK_DIR}/worker_level2_from_jobs.py" \
    --run_dir "${L1_RUN_DIR}" \
    --data_dir "${L1_DATA_DIR}" \
    --job_index "${JOB_IDX}" \
    --S_BIN 64 --D_BIN 1000

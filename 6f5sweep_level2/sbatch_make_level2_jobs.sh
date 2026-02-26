#!/bin/bash -l
#SBATCH --job-name=6f5_make_L2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0

set -euo pipefail

: "${L1_RUN_DIR:?Need L1_RUN_DIR (path to Level-1 results)}"
: "${WORK_DIR:?Need WORK_DIR}"
: "${CONTAINER:?Need CONTAINER}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "[makeL2] L1_RUN_DIR=${L1_RUN_DIR}"
echo "[makeL2] WORK_DIR=${WORK_DIR}"

singularity exec -B "${WORK_DIR}" -B "${L1_RUN_DIR}" "${CONTAINER}" \
  python "${WORK_DIR}/make_level2_tasks_from_shard_hitlist.py" \
    --run_dir "${L1_RUN_DIR}" \
    --mode full_shard_expansion \
    --depth 5000

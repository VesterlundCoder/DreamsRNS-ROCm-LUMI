#!/bin/bash -l
#SBATCH --job-name=6f5_post_L1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --mem=64G

set -euo pipefail

: "${RUN_DIR:?Need RUN_DIR}"
: "${WORK_DIR:?Need WORK_DIR}"
: "${CONTAINER:?Need CONTAINER}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "[post] RUN_DIR=${RUN_DIR}"

singularity exec -B "${WORK_DIR}" -B "${RUN_DIR}" "${CONTAINER}" \
  python "${WORK_DIR}/postprocess_level1.py" \
    --run_dir "${RUN_DIR}" \
    --min_pairs_per_shard 200 \
    --score_thresh 20.0 \
    --top_shards_per_const 200 \
    --max_consts_per_shard 6 \
    --escalate_depth 5000 \
    --escalate_mode full_shard_expansion

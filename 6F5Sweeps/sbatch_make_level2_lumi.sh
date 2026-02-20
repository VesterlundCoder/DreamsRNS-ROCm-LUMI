#!/usr/bin/env bash
#SBATCH --job-name=cmf_make_L2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0

set -euo pipefail
: "${RUN_DIR:?Need RUN_DIR}"

module purge
module load LUMI/23.09
module load cray-python
# source $PWD/.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "[makeL2] RUN_DIR=${RUN_DIR}"

python make_level2_tasks_from_shard_hitlist.py \
  --run_dir "${RUN_DIR}" \
  --mode full_shard_expansion \
  --depth 2000

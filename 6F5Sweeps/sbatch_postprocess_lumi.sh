#!/usr/bin/env bash
#SBATCH --job-name=cmf_post_L1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0

set -euo pipefail
: "${RUN_DIR:?Need RUN_DIR}"

module purge
module load LUMI/23.09
module load cray-python
# source $PWD/.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "[post] RUN_DIR=${RUN_DIR}"

python postprocess_level1.py \
  --run_dir "${RUN_DIR}" \
  --min_pairs_per_shard 200 \
  --score_thresh 20.0 \
  --top_shards_per_const 200 \
  --max_consts_per_shard 6 \
  --escalate_depth 2000 \
  --escalate_mode full_shard_expansion

#!/bin/bash -l
#SBATCH --job-name=6f5_launch_L2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail

: "${L1_RUN_DIR:?Need L1_RUN_DIR (path to Level-1 results)}"
: "${WORK_DIR:?Need WORK_DIR}"
: "${CONTAINER:?Need CONTAINER}"
: "${L2_TIME:?Need L2_TIME}"
: "${L2_MAX_CONCURRENT:?Need L2_MAX_CONCURRENT}"

LEVEL2_JOBS="${L1_RUN_DIR}/results/level2_jobs.jsonl"
if [[ ! -f "${LEVEL2_JOBS}" ]]; then
    echo "[launchL2] Missing ${LEVEL2_JOBS}"
    echo "Run make_level2_tasks_from_shard_hitlist.py first."
    exit 1
fi

N=$(wc -l < "${LEVEL2_JOBS}")
if [[ "${N}" -le 0 ]]; then
    echo "[launchL2] No jobs in level2_jobs.jsonl"
    exit 1
fi

ARRAY_MAX=$((N-1))
echo "[launchL2] level2 jobs=${N} -> array=0-${ARRAY_MAX}%${L2_MAX_CONCURRENT}"
echo "[launchL2] L1_RUN_DIR=${L1_RUN_DIR}"

JOBID=$(sbatch \
  --account="${SLURM_JOB_ACCOUNT}" --partition="${SLURM_JOB_PARTITION}" \
  --export=ALL,L1_RUN_DIR="${L1_RUN_DIR}",WORK_DIR="${WORK_DIR}",CONTAINER="${CONTAINER}" \
  --time="${L2_TIME}" \
  --array=0-"${ARRAY_MAX}"%"${L2_MAX_CONCURRENT}" \
  "${WORK_DIR}/sbatch_level2_worker.sh" | awk '{print $4}')

echo "[launchL2] Submitted Level-2 array jobid=${JOBID}"
echo "${JOBID}" > "${L1_RUN_DIR}/results/level2_array_jobid.txt"

#!/usr/bin/env bash
#SBATCH --job-name=cmf_launch_L2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail
: "${RUN_DIR:?Need RUN_DIR}"
: "${L2_TIME:?Need L2_TIME}"
: "${L2_MAX_CONCURRENT:?Need L2_MAX_CONCURRENT}"

# NOTE: account/partition inherited from sbatch environment; if not, pass explicitly in master.
LEVEL2_JOBS="${RUN_DIR}/results/level2_jobs.jsonl"
if [[ ! -f "${LEVEL2_JOBS}" ]]; then
  echo "[launchL2] Missing ${LEVEL2_JOBS}"
  exit 1
fi

N=$(wc -l < "${LEVEL2_JOBS}")
if [[ "${N}" -le 0 ]]; then
  echo "[launchL2] No jobs in level2_jobs.jsonl"
  exit 1
fi

ARRAY_MAX=$((N-1))
echo "[launchL2] level2 jobs=${N} -> array=0-${ARRAY_MAX}%${L2_MAX_CONCURRENT}"
echo "[launchL2] RUN_DIR=${RUN_DIR}"

JOBID=$(sbatch \
  --export=ALL,RUN_DIR="${RUN_DIR}" \
  --time="${L2_TIME}" \
  --array=0-"${ARRAY_MAX}"%"${L2_MAX_CONCURRENT}" \
  sbatch_level2_array_worker_lumi.sh | awk '{print $4}')

echo "[launchL2] Submitted Level-2 array jobid=${JOBID}"
echo "${JOBID}" > "${RUN_DIR}/results/level2_array_jobid.txt"

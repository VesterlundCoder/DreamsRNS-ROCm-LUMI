#!/usr/bin/env bash
set -euo pipefail

# submit_pipeline_lumi.sh
# End-to-end Slurm chain (A→B→C→D) for:
#  A) Level-1 shard sweep (Variant B)
#  B) Postprocess Level-1 -> shard_hitlist
#  C) Build Level-2 jobs from shard_hitlist
#  D) Submit Level-2 job array sized from level2_jobs.jsonl

# ---- USER EDITS ----
ACCOUNT="${ACCOUNT:-YOUR_PROJECT_ACCOUNT}"
PARTITION="${PARTITION:-gpu}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%dT%H-%M-%SZ)_6f5_zeta5_L1}"
RUN_DIR="${RUN_DIR:-$PWD/runs/${RUN_ID}}"

# Inputs (generated earlier with generate_tasks_6f5.py --emit_mode compact etc.)
SHIFTS_JSONL="${SHIFTS_JSONL:-$PWD/tasks_6f5_zeta5_shifts.jsonl}"
DIRS_JSONL="${DIRS_JSONL:-$PWD/tasks_6f5_zeta5_dirs.jsonl}"

# CMFs list (one per line). For single CMF, make a 1-line file.
CMFS_JSONL="${CMFS_JSONL:-$PWD/cmfs_level1.jsonl}"

# Resources (tune)
L1_NODES="${L1_NODES:-1}"
L1_TIME="${L1_TIME:-02:00:00}"

POST_TIME="${POST_TIME:-00:20:00}"
MK2_TIME="${MK2_TIME:-00:10:00}"
LAUNCH_TIME="${LAUNCH_TIME:-00:05:00}"

# Level-2 array resources
L2_TIME="${L2_TIME:-04:00:00}"
L2_MAX_CONCURRENT="${L2_MAX_CONCURRENT:-256}"   # array throttle

# ---- derived ----
mkdir -p "${RUN_DIR}"

echo "[pipeline] RUN_DIR=${RUN_DIR}"
echo "[pipeline] Using shifts=${SHIFTS_JSONL}"
echo "[pipeline] Using dirs=${DIRS_JSONL}"
echo "[pipeline] Using cmfs=${CMFS_JSONL}"

# A) Level-1
JOB_A=$(sbatch \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",SHIFTS_JSONL="${SHIFTS_JSONL}",DIRS_JSONL="${DIRS_JSONL}",CMFS_JSONL="${CMFS_JSONL}" \
  --nodes="${L1_NODES}" --time="${L1_TIME}" \
  sbatch_level1_lumi.sh | awk '{print $4}')
echo "[pipeline] Submitted Level-1: ${JOB_A}"

# B) postprocess (afterok A)
JOB_B=$(sbatch \
  --dependency=afterok:"${JOB_A}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,RUN_DIR="${RUN_DIR}" \
  --time="${POST_TIME}" \
  sbatch_postprocess_lumi.sh | awk '{print $4}')
echo "[pipeline] Submitted postprocess: ${JOB_B} (afterok ${JOB_A})"

# C) make level2 jobs (afterok B)
JOB_C=$(sbatch \
  --dependency=afterok:"${JOB_B}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,RUN_DIR="${RUN_DIR}" \
  --time="${MK2_TIME}" \
  sbatch_make_level2_lumi.sh | awk '{print $4}')
echo "[pipeline] Submitted make_level2_jobs: ${JOB_C} (afterok ${JOB_B})"

# D) launch Level-2 array (afterok C)
JOB_D=$(sbatch \
  --dependency=afterok:"${JOB_C}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",L2_TIME="${L2_TIME}",L2_MAX_CONCURRENT="${L2_MAX_CONCURRENT}" \
  --time="${LAUNCH_TIME}" \
  sbatch_launch_level2_array_lumi.sh | awk '{print $4}')
echo "[pipeline] Submitted launch_level2_array: ${JOB_D} (afterok ${JOB_C})"

echo "[pipeline] Done."
echo "  Level-1 jobid: ${JOB_A}"
echo "  Postprocess:   ${JOB_B}"
echo "  Make L2 jobs:  ${JOB_C}"
echo "  Launch L2:     ${JOB_D}"

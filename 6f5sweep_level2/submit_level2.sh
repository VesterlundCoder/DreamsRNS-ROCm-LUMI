#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# submit_level2.sh — Submit Level-2 6F5 sweep on LUMI
#
# Prerequisites: Level-1 must have completed and produced:
#   <L1_RUN_DIR>/results/shard_hitlist.jsonl
#
# This submits a two-step Slurm chain:
#   C) Generate Level-2 job definitions from Level-1 shard_hitlist
#   D) Launch Level-2 job array (full shard expansion at depth 5000)
#
# You should have already analyzed the Level-1 results before running this.
# ============================================================================

# ---- USER EDITS ----
ACCOUNT="${ACCOUNT:-project_465002669}"
PARTITION="${PARTITION:-standard-g}"

# L1_RUN_DIR: path to the completed Level-1 run (must contain results/shard_hitlist.jsonl)
L1_RUN_DIR="${L1_RUN_DIR:?Set L1_RUN_DIR to the Level-1 run directory}"

# WORK_DIR: where the level-2 scripts live on LUMI
WORK_DIR="${WORK_DIR:-/projappl/${ACCOUNT}/$(whoami)/6f5sweep_level2}"

# CONTAINER: path to the Singularity .sif image (same as level-1)
CONTAINER="${CONTAINER:-/projappl/${ACCOUNT}/containers/6f5-level1.sif}"

# Resources
MK2_TIME="${MK2_TIME:-00:10:00}"
LAUNCH_TIME="${LAUNCH_TIME:-00:05:00}"
L2_TIME="${L2_TIME:-04:00:00}"
L2_MAX_CONCURRENT="${L2_MAX_CONCURRENT:-256}"

# ---- Verify Level-1 output exists ----
HITLIST="${L1_RUN_DIR}/results/shard_hitlist.jsonl"
if [[ ! -f "${HITLIST}" ]]; then
    echo "ERROR: Missing ${HITLIST}"
    echo "Level-1 must complete before running Level-2."
    echo "Check: ls ${L1_RUN_DIR}/results/"
    exit 1
fi

N_SHARDS=$(wc -l < "${HITLIST}")
echo "============================================"
echo " 6F5 Level-2 Sweep"
echo "============================================"
echo "  ACCOUNT:        ${ACCOUNT}"
echo "  PARTITION:       ${PARTITION}"
echo "  L1_RUN_DIR:      ${L1_RUN_DIR}"
echo "  WORK_DIR:        ${WORK_DIR}"
echo "  CONTAINER:       ${CONTAINER}"
echo "  Shard hitlist:   ${N_SHARDS} shards to escalate"
echo "  L2_TIME:         ${L2_TIME}"
echo "  L2_MAX_CONCURRENT: ${L2_MAX_CONCURRENT}"
echo "============================================"

# C) Generate Level-2 jobs from shard_hitlist
JOB_C=$(sbatch \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,L1_RUN_DIR="${L1_RUN_DIR}",WORK_DIR="${WORK_DIR}",CONTAINER="${CONTAINER}" \
  --time="${MK2_TIME}" \
  "${WORK_DIR}/sbatch_make_level2_jobs.sh" | awk '{print $4}')
echo "[submit] Make Level-2 jobs: job ${JOB_C}"

# D) Launch Level-2 array (afterok C)
JOB_D=$(sbatch \
  --dependency=afterok:"${JOB_C}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,L1_RUN_DIR="${L1_RUN_DIR}",WORK_DIR="${WORK_DIR}",CONTAINER="${CONTAINER}",L2_TIME="${L2_TIME}",L2_MAX_CONCURRENT="${L2_MAX_CONCURRENT}" \
  --time="${LAUNCH_TIME}" \
  "${WORK_DIR}/sbatch_launch_level2_array.sh" | awk '{print $4}')
echo "[submit] Launch Level-2 array: job ${JOB_D} (afterok ${JOB_C})"

echo ""
echo "Done. Monitor with:  squeue -u \$(whoami)"
echo ""
echo "After completion, Level-2 results at:"
echo "  ${L1_RUN_DIR}/results/hits_depth*_job*.jsonl   (per-job hits)"
echo "  ${L1_RUN_DIR}/progress/pairs_tested_L2_*.jsonl (coverage)"

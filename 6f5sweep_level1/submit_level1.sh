#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# submit_level1.sh — Submit Level-1 6F5 sweep + postprocess on LUMI
#
# This submits a two-step Slurm chain:
#   A) Level-1 shard sweep (Variant B pairing, ~39M pairs at depth 2000)
#   B) Postprocess → merged hits, shard_summary, shard_hitlist (for Level-2)
#
# Results are saved to RUN_DIR for offline analysis before running Level-2.
# ============================================================================

# ---- USER EDITS ----
ACCOUNT="${ACCOUNT:-project_465002669}"
PARTITION="${PARTITION:-small-g}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%dT%H-%M-%SZ)_6f5_L1}"

# WORK_DIR: where the scripts + data files live on LUMI
WORK_DIR="${WORK_DIR:-/projappl/${ACCOUNT}/$(whoami)/6f5sweep_level1}"

# RUN_DIR: where results will be written (use scratch for large output)
RUN_DIR="${RUN_DIR:-/scratch/${ACCOUNT}/$(whoami)/runs/${RUN_ID}}"

# CONTAINER: path to the Singularity .sif image
CONTAINER="${CONTAINER:-/projappl/${ACCOUNT}/containers/6f5-level1.sif}"

# Resources
L1_NODES="${L1_NODES:-1}"
L1_TIME="${L1_TIME:-2-00:00:00}"
POST_TIME="${POST_TIME:-00:20:00}"

# ---- Verify inputs exist ----
for F in "${WORK_DIR}/worker_variantB_skeleton.py" \
         "${WORK_DIR}/postprocess_level1.py" \
         "${WORK_DIR}/cmfs_level1.jsonl" \
         "${WORK_DIR}/tasks_6f5_zeta5_shifts.jsonl" \
         "${WORK_DIR}/tasks_6f5_zeta5_dirs.jsonl"; do
    if [[ ! -f "${F}" ]]; then
        echo "ERROR: Missing ${F}"
        echo "Upload 6f5sweep_level1/ contents to ${WORK_DIR}/ first."
        exit 1
    fi
done

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    echo "Build with: singularity build 6f5-level1.sif 6f5_level1.def"
    echo "Then upload to ${CONTAINER}"
    exit 1
fi

# ---- Create run directory ----
mkdir -p "${RUN_DIR}"

echo "============================================"
echo " 6F5 Level-1 Sweep"
echo "============================================"
echo "  ACCOUNT:   ${ACCOUNT}"
echo "  PARTITION:  ${PARTITION}"
echo "  WORK_DIR:   ${WORK_DIR}"
echo "  RUN_DIR:    ${RUN_DIR}"
echo "  CONTAINER:  ${CONTAINER}"
echo "  L1_NODES:   ${L1_NODES}"
echo "  L1_TIME:    ${L1_TIME}"
echo "  POST_TIME:  ${POST_TIME}"
echo "============================================"

# A) Level-1 sweep
JOB_A=$(sbatch \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",WORK_DIR="${WORK_DIR}",CONTAINER="${CONTAINER}" \
  --nodes="${L1_NODES}" --time="${L1_TIME}" \
  "${WORK_DIR}/sbatch_level1.sh" | awk '{print $4}')
echo "[submit] Level-1 sweep: job ${JOB_A}"

# B) Postprocess (afterok A)
JOB_B=$(sbatch \
  --dependency=afterok:"${JOB_A}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",WORK_DIR="${WORK_DIR}",CONTAINER="${CONTAINER}" \
  --time="${POST_TIME}" \
  "${WORK_DIR}/sbatch_postprocess.sh" | awk '{print $4}')
echo "[submit] Postprocess:   job ${JOB_B} (afterok ${JOB_A})"

echo ""
echo "Done. Monitor with:  squeue -u \$(whoami)"
echo ""
echo "After completion, results at:"
echo "  ${RUN_DIR}/results/hits_level1.jsonl        (merged hits)"
echo "  ${RUN_DIR}/results/shard_summary.jsonl       (per-shard stats)"
echo "  ${RUN_DIR}/results/shard_hitlist.jsonl        (escalation plan for Level-2)"
echo "  ${RUN_DIR}/results/miss_intervals.jsonl       (complete-miss shards)"
echo ""
echo "Download for analysis:  scp lumi:${RUN_DIR}/results/*.jsonl ."

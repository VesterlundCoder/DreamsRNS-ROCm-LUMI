#!/bin/bash -l
# =============================================================================
# Dreams-RNS-ROCm: LUMI-G small-g single-node SBATCH script
#
# Runs PCF verification inside a Singularity container on 1 LUMI-G node.
# Always uses small-g partition (1 job per node, up to 7 GCDs / 3 days).
#
# Prerequisites:
#   1. Build container: singularity build dreams_rocm.sif env/dreams_rocm.def
#   2. Upload data:     scp pcfs.json cmf_pcfs.json $SCRATCH/dreams_data/
#   3. Submit:          sbatch scripts/sbatch_1node_8gpu.sh
#
# Override defaults:
#   sbatch --account=project_XXXXXXX scripts/sbatch_1node_8gpu.sh
#   sbatch --time=3:00:00 scripts/sbatch_1node_8gpu.sh
# =============================================================================

#SBATCH --job-name=dreams-rns
#SBATCH --output=dreams-rns_%j.out
#SBATCH --error=dreams-rns_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --account=project_465001234

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

module --force purge
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1
module load singularity-bindings

# Project paths
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTAINER="${REPO_DIR}/dreams_rocm.sif"
DATA_DIR="${DREAMS_DATA_DIR:-${SCRATCH}/dreams_data}"

# Fallback: if no .sif, use system Python
if [ ! -f "${CONTAINER}" ]; then
    echo "WARNING: Container not found at ${CONTAINER}"
    echo "         Falling back to module-based Python environment"
    module load cray-python/3.10.10
    USE_CONTAINER=0
else
    USE_CONTAINER=1
fi

# Create unique run directory on /scratch
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUTPUT_DIR="${SCRATCH}/dreams_runs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Dreams-RNS-ROCm v0.2.0 Pipeline"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "GPUs:          ${SLURM_GPUS_PER_NODE:-1}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Container:     ${CONTAINER}"
echo "Data dir:      ${DATA_DIR}"
echo "Run ID:        ${RUN_ID}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# GPU binding
export ROCR_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
# Helper: run command inside or outside container
# ---------------------------------------------------------------------------
run_cmd() {
    if [ "${USE_CONTAINER}" = "1" ]; then
        singularity exec \
            --rocm \
            --bind "${REPO_DIR}:/workspace" \
            --bind "${DATA_DIR}:/data" \
            --bind "${OUTPUT_DIR}:/output" \
            "${CONTAINER}" \
            python3 "$@"
    else
        python3 "$@"
    fi
}

# ---------------------------------------------------------------------------
# 1. Smoke test (5 PCFs, quick)
# ---------------------------------------------------------------------------
echo "=== Smoke test (5 PCFs, depth=500, K=16) ==="

INPUT_FILE=""
if [ -f "${DATA_DIR}/pcfs.json" ]; then
    INPUT_FILE="/data/pcfs.json"
    INPUT_HOST="${DATA_DIR}/pcfs.json"
elif [ -f "${DATA_DIR}/cmf_pcfs.json" ]; then
    INPUT_FILE="/data/cmf_pcfs.json"
    INPUT_HOST="${DATA_DIR}/cmf_pcfs.json"
else
    echo "ERROR: No pcfs.json or cmf_pcfs.json found in ${DATA_DIR}"
    exit 1
fi

if [ "${USE_CONTAINER}" = "1" ]; then
    run_cmd /workspace/scripts/euler2ai_verify.py \
        --input "${INPUT_FILE}" \
        --depth 500 --K 16 --max-tasks 5 \
        --output /output/smoke_report.csv
else
    python3 "${REPO_DIR}/scripts/euler2ai_verify.py" \
        --input "${INPUT_HOST}" \
        --depth 500 --K 16 --max-tasks 5 \
        --output "${OUTPUT_DIR}/smoke_report.csv"
fi

SMOKE_EXIT=$?
if [ ${SMOKE_EXIT} -ne 0 ]; then
    echo "ERROR: Smoke test failed with exit code ${SMOKE_EXIT}"
    exit ${SMOKE_EXIT}
fi

# ---------------------------------------------------------------------------
# 2. Full verification run
# ---------------------------------------------------------------------------
echo ""
echo "=== Full run (depth=2000, K=32) ==="

if [ "${USE_CONTAINER}" = "1" ]; then
    run_cmd /workspace/scripts/euler2ai_verify.py \
        --input "${INPUT_FILE}" \
        --depth 2000 --K 32 --max-tasks 0 \
        --output /output/full_report.csv
else
    python3 "${REPO_DIR}/scripts/euler2ai_verify.py" \
        --input "${INPUT_HOST}" \
        --depth 2000 --K 32 --max-tasks 0 \
        --output "${OUTPUT_DIR}/full_report.csv"
fi

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "Results in: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"/*.csv 2>/dev/null || echo "(no CSV output)"
echo "=========================================="

exit ${EXIT_CODE}

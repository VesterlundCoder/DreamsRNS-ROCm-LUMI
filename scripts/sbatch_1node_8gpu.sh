#!/bin/bash -l
# =============================================================================
# Dreams-RNS-ROCm: LUMI-G 1-node / 8-GPU SBATCH script
#
# Usage:
#   sbatch scripts/sbatch_1node_8gpu.sh
#
# Override defaults:
#   sbatch --account=project_XXXXXXX scripts/sbatch_1node_8gpu.sh
#   sbatch --partition=standard-g scripts/sbatch_1node_8gpu.sh
#   sbatch --time=2:00:00 scripts/sbatch_1node_8gpu.sh
# =============================================================================

#SBATCH --job-name=dreams-rns
#SBATCH --output=dreams-rns_%j.out
#SBATCH --error=dreams-rns_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=1:00:00
#SBATCH --account=project_465001234

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load modules
module --force purge
module load LUMI/23.09
module load partition/G
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/5.6.1
module load cray-python/3.10.10

# Activate virtual environment
if [ -f "$HOME/dreams-venv/bin/activate" ]; then
    source "$HOME/dreams-venv/bin/activate"
fi

# Project paths
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LUMI_PROJECT="${LUMI_PROJECT:-$(echo $SBATCH_ACCOUNT | sed 's/project_//')}"

# Create unique run directory on /scratch
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUTPUT_DIR="/scratch/${LUMI_PROJECT}/dreams_runs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Dreams-RNS-ROCm Pipeline"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Tasks/node:    ${SLURM_NTASKS_PER_NODE}"
echo "GPUs/node:     ${SLURM_GPUS_PER_NODE:-8}"
echo "CPUs/task:     ${SLURM_CPUS_PER_TASK}"
echo "Run ID:        ${RUN_ID}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Repo dir:      ${REPO_DIR}"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# GPU binding wrapper
# Each MPI rank sees exactly one GPU via ROCR_VISIBLE_DEVICES
# ---------------------------------------------------------------------------

cat << 'EOF' > "${OUTPUT_DIR}/select_gpu.sh"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
exec "$@"
EOF
chmod +x "${OUTPUT_DIR}/select_gpu.sh"

# CPU-to-GPU binding for optimal NUMA affinity on LUMI-G
# Maps: rank 0->GPU4(core49), rank 1->GPU5(core57), rank 2->GPU2(core17),
#        rank 3->GPU3(core25), rank 4->GPU0(core1),  rank 5->GPU1(core9),
#        rank 6->GPU6(core33), rank 7->GPU7(core41)
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG="${REPO_DIR}/configs/lumi_1node_8gpu.yaml"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Export for use in Python
export DREAMS_OUTPUT_DIR="${OUTPUT_DIR}"
export DREAMS_RUN_ID="${RUN_ID}"
export DREAMS_CONFIG="${CONFIG}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

echo "Launching MPI sweep with 8 ranks..."
echo ""

srun --cpu-bind=${CPU_BIND} \
     "${OUTPUT_DIR}/select_gpu.sh" \
     python3 "${REPO_DIR}/scripts/run_mpi_sweep.py" \
         --config "${CONFIG}" \
         --output-dir "${OUTPUT_DIR}" \
         --run-id "${RUN_ID}"

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

rm -f "${OUTPUT_DIR}/select_gpu.sh"

echo ""
echo "=========================================="
echo "Job completed with exit code: ${EXIT_CODE}"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="

exit ${EXIT_CODE}

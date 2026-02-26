#!/bin/bash -l
#SBATCH --job-name=6f5_L1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- LUMI-G small-g partition (sub-node, up to 4 nodes, max 3 days) ---
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=2-00:00:00
#SBATCH --mem=480G

set -euo pipefail

: "${RUN_DIR:?Need RUN_DIR}"
: "${WORK_DIR:?Need WORK_DIR}"
: "${CONTAINER:?Need CONTAINER}"

SHIFTS_JSONL="${WORK_DIR}/tasks_6f5_zeta5_shifts.jsonl"
DIRS_JSONL="${WORK_DIR}/tasks_6f5_zeta5_dirs.jsonl"
CMFS_JSONL="${WORK_DIR}/cmfs_level1.jsonl"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/progress"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

echo "[L1] RUN_DIR=${RUN_DIR}"
echo "[L1] WORK_DIR=${WORK_DIR}"
echo "[L1] CONTAINER=${CONTAINER}"
echo "[L1] SLURM_NTASKS=${SLURM_NTASKS} GPUs/node=${SLURM_GPUS_PER_NODE}"

# GPU binding: set ROCR_VISIBLE_DEVICES per rank (LUMI recommended pattern)
cat << EOF > "${RUN_DIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x "${RUN_DIR}/select_gpu"

# CPU bind map for LUMI-G (maps rank to correct NUMA/CCD near its GPU)
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

# Run the worker inside the Singularity container.
# -B bind-mounts project + scratch so the container can access data and write results.
srun --cpu-bind=${CPU_BIND} \
  "${RUN_DIR}/select_gpu" \
  singularity exec -B "${WORK_DIR}" -B "${RUN_DIR}" "${CONTAINER}" \
  python3 "${WORK_DIR}/worker_variantB_skeleton.py" \
    --run_dir "${RUN_DIR}" \
    --cmfs_jsonl "${CMFS_JSONL}" \
    --shifts_jsonl "${SHIFTS_JSONL}" \
    --dirs_jsonl "${DIRS_JSONL}" \
    --rank "${SLURM_PROCID}" \
    --world "${SLURM_NTASKS}" \
    --depth 2000 \
    --dirs_per_shift 4883 \
    --stride 99991 \
    --mix_a 104729 \
    --mix_b 12345 \
    --S_BIN 64 --D_BIN 1000 \
    --pre_score 5e-3

rm -f "${RUN_DIR}/select_gpu"

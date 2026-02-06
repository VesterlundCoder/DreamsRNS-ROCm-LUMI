#!/bin/bash
# =============================================================================
# LUMI-G module environment for Dreams-RNS-ROCm pipeline
#
# Usage:
#   source env/lumi_modules.sh
#
# This script loads the recommended modules for ROCm + Python on LUMI-G.
# After sourcing, create a venv and install requirements.
# =============================================================================

set -e

echo "=== Loading LUMI-G modules for Dreams-RNS-ROCm ==="

# Reset module environment
module --force purge

# Load LUMI software stack (CrayPE + ROCm)
module load LUMI/23.09
module load partition/G

# Load ROCm programming environment
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/5.6.1

# Load Cray Python (includes numpy, scipy, mpi4py built against Cray MPICH)
module load cray-python/3.10.10

# Load CMake for building the RNS-ROCm C++ library
module load buildtools/23.09

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=0  # We don't need GPU-aware MPI for this pipeline

# Ensure hipcc is on PATH
export PATH=${ROCM_PATH}/bin:${PATH}

# Display loaded modules
echo ""
echo "=== Loaded modules ==="
module list

echo ""
echo "=== ROCm info ==="
echo "ROCM_PATH: ${ROCM_PATH:-not set}"
hipcc --version 2>/dev/null || echo "hipcc not found (expected on login node)"

echo ""
echo "=== Python info ==="
python3 --version
echo ""
echo "=== Next steps ==="
echo "1. Create a virtual environment:"
echo "   python3 -m venv --system-site-packages \$HOME/dreams-venv"
echo "   source \$HOME/dreams-venv/bin/activate"
echo ""
echo "2. Install requirements:"
echo "   pip install --upgrade pip"
echo "   pip install -r env/requirements.txt"
echo ""
echo "3. (Optional) Build the RNS-ROCm native library:"
echo "   See README.md section 'Building the RNS-ROCm Library'"
echo ""
echo "Environment ready."

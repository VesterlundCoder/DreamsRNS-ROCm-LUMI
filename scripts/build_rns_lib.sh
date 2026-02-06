#!/bin/bash
# =============================================================================
# Build the RNS-ROCm C++ library on LUMI-G
#
# Usage (on a compute node with GPU):
#   srun --partition=small-g --nodes=1 --ntasks=1 --gpus=1 \
#        --time=00:30:00 --account=project_XXXXXXX \
#        bash scripts/build_rns_lib.sh
#
# Or without GPU (CPU-only build, works on login node):
#   bash scripts/build_rns_lib.sh --cpu-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="${REPO_DIR}/rns_rocm_lib"
BUILD_DIR="${LIB_DIR}/build"

CPU_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --cpu-only) CPU_ONLY=true ;;
    esac
done

echo "=== Building RNS-ROCm Library ==="
echo "  Source:    ${LIB_DIR}"
echo "  Build:     ${BUILD_DIR}"
echo "  CPU-only:  ${CPU_ONLY}"
echo ""

# Load modules if on LUMI
if command -v module &>/dev/null; then
    module --force purge 2>/dev/null || true
    module load LUMI/23.09 2>/dev/null || true
    module load partition/G 2>/dev/null || true
    module load PrgEnv-cray 2>/dev/null || true
    module load craype-accel-amd-gfx90a 2>/dev/null || true
    module load rocm/5.6.1 2>/dev/null || true
    module load buildtools/23.09 2>/dev/null || true
    echo "Loaded LUMI modules"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Determine GPU flag
GPU_FLAG="ON"
if [ "$CPU_ONLY" = true ]; then
    GPU_FLAG="OFF"
fi

# Configure
echo ""
echo "=== CMake Configure ==="
cmake "${LIB_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRNS_ENABLE_GPU=${GPU_FLAG} \
    -DRNS_ENABLE_TESTS=ON \
    -DRNS_ENABLE_EXAMPLES=OFF \
    -DRNS_BUILD_PYTHON=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

# Build
echo ""
echo "=== CMake Build ==="
cmake --build . -j "$(nproc 2>/dev/null || echo 4)"

# Test
echo ""
echo "=== Running Tests ==="
ctest --output-on-failure || echo "WARNING: Some tests failed"

# Report
echo ""
echo "=== Build Complete ==="
LIB_PATH=$(find "${BUILD_DIR}" -name "librns_rocm*" -type f | head -1)
if [ -n "${LIB_PATH}" ]; then
    echo "  Library: ${LIB_PATH}"
    echo ""
    echo "  To use with Dreams pipeline, set:"
    echo "    export RNS_ROCM_LIB=${BUILD_DIR}"
else
    echo "  WARNING: Library file not found in ${BUILD_DIR}"
    echo "  Check build output for errors."
fi

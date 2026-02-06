#!/bin/bash
# Build and run all tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"

# Parse arguments
GPU_ENABLED=OFF
if [[ "$1" == "--gpu" ]]; then
  GPU_ENABLED=ON
fi

echo "=== RNS Test Runner ==="
echo "GPU enabled: $GPU_ENABLED"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring..."
cmake "$ROOT_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DRNS_ENABLE_GPU=$GPU_ENABLED \
  -DRNS_ENABLE_TESTS=ON

# Build
echo ""
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo ""
echo "Running tests..."
ctest --output-on-failure

echo ""
echo "=== All tests passed ==="

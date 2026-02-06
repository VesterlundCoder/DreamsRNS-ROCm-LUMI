#!/bin/bash
# Format all C++ source files using clang-format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Find and format all C++ files
find include src tests examples -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hip" \) | while read -r file; do
  echo "Formatting: $file"
  clang-format -i "$file"
done

echo "Done."

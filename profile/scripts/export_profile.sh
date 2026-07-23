#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# export_profile.sh — Package profiling results for transfer
# ============================================================
# Creates a tar.gz containing baseline results, nsys reports,
# and ncu reports for transfer to a Windows machine with
# Nsight GUI tools.
#
# Usage:
#   ./export_profile.sh
#   ./export_profile.sh my_experiment
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE_DIR="${PROJECT_ROOT}/profile"
ARCHIVE_NAME="${1:-profile_results_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${PROFILE_DIR}/../${ARCHIVE_NAME}.tar.gz"

echo "Packaging profiling results..."
echo "Source: ${PROFILE_DIR}"
echo "Output: ${OUTPUT}"

cd "${PROJECT_ROOT}"

tar -czf "${OUTPUT}" \
  --exclude='.gitkeep' \
  --exclude='*.tar.gz' \
  profile/results/ \
  profile/nsys/ \
  profile/ncu/ \
  profile/scripts/ \
  profile/README.md \
  2>/dev/null || true

if [ -f "${OUTPUT}" ]; then
  SIZE=$(du -h "${OUTPUT}" | cut -f1)
  echo ""
  echo "================================================"
  echo "Archive created: ${OUTPUT}"
  echo "Size:            ${SIZE}"
  echo ""
  echo "To extract on Windows:"
  echo "  tar -xzf ${ARCHIVE_NAME}.tar.gz"
  echo "  or use 7-Zip / WinRAR to open the .tar.gz"
  echo ""
  echo "Then open .nsys-rep with Nsight Systems GUI."
  echo "Open .ncu-rep with Nsight Compute GUI."
  echo "================================================"
else
  echo "ERROR: Failed to create archive."
  exit 1
fi

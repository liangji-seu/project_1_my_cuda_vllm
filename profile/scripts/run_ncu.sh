#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_ncu.sh — Profile a specific kernel with Nsight Compute
# ============================================================
#
# Usage:
#   KERNEL_REGEX="matmul" ./run_ncu.sh
#   KERNEL_REGEX="mha_kernel_cuda_fp32" LAUNCH_COUNT=3 ./run_ncu.sh
#   ./run_ncu.sh matmul
#
# This script analyzes a specific kernel, NOT the entire model.
# To get a kernel list first, run nsys profile which shows all
# kernel launches in the timeline.
#
# After running, copy the .ncu-rep file to Windows and open
# with Nsight Compute GUI:
#   File → Open → select profile/ncu/*.ncu-rep
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if ! command -v ncu &>/dev/null; then
  echo "ERROR: ncu (NVIDIA Nsight Compute) not found in PATH."
  echo "Please install CUDA toolkit or add ncu to PATH."
  exit 1
fi

# Parse args
KERNEL_REGEX="${KERNEL_REGEX:-}"
if [ $# -ge 1 ]; then
  KERNEL_REGEX="$1"
fi

if [ -z "${KERNEL_REGEX}" ]; then
  echo "USAGE: KERNEL_REGEX=\"matmul_kernel\" $0 [kernel_regex]"
  echo ""
  echo "Examples:"
  echo "  KERNEL_REGEX=\"matmul_kernel_cuda_fp32\" $0"
  echo "  KERNEL_REGEX=\"mha_kernel\" $0"
  echo "  KERNEL_REGEX=\"rmsnorm_kernel\" $0"
  echo ""
  echo "Tip: Run nsys profile first to see all kernel names,"
  echo "     then pick one to analyze with ncu."
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-"${PROJECT_ROOT}/demo/qwen2.5_0.5b_instruct.bin"}"
VOCAB_PATH="${VOCAB_PATH:-/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json}"
PROMPT="${PROMPT:-"Hello, how are you?"}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-1}"
NCU_OUTPUT="${NCU_OUTPUT:-"${PROJECT_ROOT}/profile/ncu/kernel_profile"}"
BIN_DIR="${BIN_DIR:-"${PROJECT_ROOT}/bin"}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Nsight Compute Kernel Profiling"
echo "================================================"
echo "ncu version:  $(ncu --version 2>&1 | head -1)"
echo "Kernel regex: ${KERNEL_REGEX}"
echo "Launch skip:  ${LAUNCH_SKIP}"
echo "Launch count: ${LAUNCH_COUNT}"
echo "Output:       ${NCU_OUTPUT}_${TIMESTAMP}"
echo "Model:        ${MODEL_PATH}"
echo "Max tokens:   ${MAX_NEW_TOKENS}"
echo "================================================"

mkdir -p "${PROJECT_ROOT}/profile/ncu"

DEMO_ARGS=(
  --model "${MODEL_PATH}"
  --tokenizer "${VOCAB_PATH}"
  --prompt "${PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --benchmark
  --greedy
  --no-stream-output
)

NCU_NAME="${NCU_OUTPUT}_${TIMESTAMP}"

echo ""
echo "Running ncu profile (this may take a while)..."
echo ""

# Run ncu
# --set full: Collect all metrics for comprehensive analysis
# --launch-skip: Skip first N launches (skip warmup)
# --launch-count: Only profile N launches
# --kernel-name regex: Only profile matching kernels
# Note: --target-processes all is needed to profile child processes
ncu \
  --target-processes all \
  --kernel-name regex:"${KERNEL_REGEX}" \
  --launch-skip "${LAUNCH_SKIP}" \
  --launch-count "${LAUNCH_COUNT}" \
  --set detailed \
  --export "${NCU_NAME}" \
  --force-overwrite \
  "${BIN_DIR}/demo" "${DEMO_ARGS[@]}" 2>&1 | tee "${NCU_NAME}_console.log" || {
  echo ""
  echo "WARNING: ncu exited with non-zero status."
  echo "This is common for kernels that don't match or launch-count issues."
  echo "Check ${NCU_NAME}_console.log for details."
}

echo ""
echo "================================================"
echo "Nsight Compute Results:"
echo "  Report:   ${NCU_NAME}.ncu-rep"
echo "  Console:  ${NCU_NAME}_console.log"

# Try CSV export if available
if ls "${NCU_NAME}.ncu-rep" &>/dev/null; then
  echo ""
  echo "Exporting CSV..."
  if ncu --import "${NCU_NAME}.ncu-rep" --csv --log-file "${NCU_NAME}.csv" 2>&1; then
    echo "  CSV:      ${NCU_NAME}.csv"
  else
    echo "  (CSV export not available for this ncu version)"
  fi
fi

# Create latest symlink
if ls "${NCU_NAME}.ncu-rep" &>/dev/null; then
  ln -sfn "$(basename "${NCU_NAME}.ncu-rep")" "${PROJECT_ROOT}/profile/ncu/latest.ncu-rep"
fi

echo ""
echo "================================================"
echo "To view in Nsight Compute GUI:"
echo "  1. Copy ${NCU_NAME}.ncu-rep to Windows"
echo "  2. Open Nsight Compute GUI"
echo "  3. File → Open → select the .ncu-rep file"
echo ""
echo "IMPORTANT:"
echo "  - ncu serializes kernel execution — timing is NOT"
echo "    representative of real performance."
echo "  - Only use ncu metrics for hardware-level analysis,"
echo "    not for throughput comparison."
echo "  - Default launch-count=1 profiles only the first"
echo "    matching kernel launch. To profile later launches,"
echo "    increase LAUNCH_SKIP."
echo "================================================"

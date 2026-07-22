#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_nsys.sh — Profile with Nsight Systems
# ============================================================
# Generates .nsys-rep for timeline analysis.
#
# Usage:
#   ./run_nsys.sh
#   MODEL_PATH=path/to/model.bin ./run_nsys.sh
#   MAX_NEW_TOKENS=16 ./run_nsys.sh
#
# After running, copy the .nsys-rep file to Windows and open
# with Nsight Systems GUI:
#   File → Open → select profile/nsys/*.nsys-rep
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if ! command -v nsys &>/dev/null; then
  echo "ERROR: nsys (NVIDIA Nsight Systems) not found in PATH."
  echo "Please install CUDA toolkit or add nsys to PATH."
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-"${PROJECT_ROOT}/demo/qwen2.5_0.5b_instruct.bin"}"
VOCAB_PATH="${VOCAB_PATH:-/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json}"
PROMPT="${PROMPT:-"Hello, how are you?"}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-2}"
NSYS_OUTPUT="${NSYS_OUTPUT:-"${PROJECT_ROOT}/profile/nsys/qwen_baseline"}"
BIN_DIR="${BIN_DIR:-"${PROJECT_ROOT}/bin"}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Nsight Systems Profiling"
echo "================================================"
echo "nsys version: $(nsys --version 2>&1 | head -1)"
echo "Output:       ${NSYS_OUTPUT}_${TIMESTAMP}"
echo "Model:        ${MODEL_PATH}"
echo "Max tokens:   ${MAX_NEW_TOKENS}"
echo "================================================"

mkdir -p "${PROJECT_ROOT}/profile/nsys"

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

# Run nsys profile
# Note: Adjust --trace flags based on nsys version.
# Common options: cuda,nvtx,osrt,cublas,cudnn,opengl,vulkan
# The --cuda-memory-usage flag tracks GPU memory allocations.
# If --force-overwrite is not available, remove existing .nsys-rep first.

echo ""
echo "Running nsys profile..."
echo ""

NSYS_NAME="${NSYS_OUTPUT}_${TIMESTAMP}"

# Try with --force-overwrite first (newer nsys), fall back to removing old file
if nsys profile --help 2>&1 | grep -q force-overwrite; then
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="${NSYS_NAME}" \
    "${BIN_DIR}/demo" "${DEMO_ARGS[@]}"
else
  # Older nsys: remove existing file manually
  rm -f "${NSYS_NAME}.nsys-rep" "${NSYS_NAME}.sqlite"
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --output="${NSYS_NAME}" \
    "${BIN_DIR}/demo" "${DEMO_ARGS[@]}"
fi

echo ""
echo "================================================"
echo "Nsight Systems Results:"
echo "  Report:  ${NSYS_NAME}.nsys-rep"
if ls "${NSYS_NAME}.sqlite" 2>/dev/null; then
  echo "  SQLite:  ${NSYS_NAME}.sqlite"
fi

# Try to generate stats
echo ""
echo "Generating stats..."
if nsys stats --help &>/dev/null 2>&1; then
  nsys stats "${NSYS_NAME}.nsys-rep" 2>&1 | tee "${NSYS_NAME}_stats.txt" || true
  echo "  Stats:   ${NSYS_NAME}_stats.txt"
fi

# Create latest symlink
ln -sfn "$(basename "${NSYS_NAME}.nsys-rep")" "${PROJECT_ROOT}/profile/nsys/latest.nsys-rep"

echo ""
echo "================================================"
echo "To view in Nsight Systems GUI:"
echo "  1. Copy ${NSYS_NAME}.nsys-rep to Windows"
echo "  2. Open Nsight Systems GUI"
echo "  3. File → Open → select the .nsys-rep file"
echo ""
echo "Note: Profiling overhead is significant."
echo "Do NOT compare nsys timing directly with baseline."
echo "================================================"

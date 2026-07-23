#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_baseline.sh — Run inference benchmark and save results
# ============================================================
# Usage:
#   ./run_baseline.sh
#   MODEL_PATH=path/to/model.bin ./run_baseline.sh
#   PROMPT="Hello" OUTPUT_TOKENS=64 ./run_baseline.sh
#
# Environment variables:
#   MODEL_PATH       - Path to .bin model file
#   VOCAB_PATH       - Path to tokenizer.json
#   PROMPT           - Input prompt text
#   PROMPT_FILE      - Read prompt from file
#   MAX_NEW_TOKENS   - Max tokens to generate (default: 128)
#   WARMUP           - Warmup iterations (default: 3)
#   REPEAT           - Benchmark repeat iterations (default: 10)
#   SEED             - Random seed (default: 42)
#   RESULTS_DIR      - Output directory (default: profile/results/baseline)
#   BIN_DIR          - Directory containing demo binary (default: bin)
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-"${PROJECT_ROOT}/demo/qwen2.5_0.5b_instruct.bin"}"
VOCAB_PATH="${VOCAB_PATH:-/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json}"
PROMPT="${PROMPT:-"请你给我介绍一下东南大学"}"
PROMPT_FILE="${PROMPT_FILE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-"${PROJECT_ROOT}/profile/results/baseline"}"
BIN_DIR="${BIN_DIR:-"${PROJECT_ROOT}/bin"}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="baseline_${TIMESTAMP}"

echo "================================================"
echo "Inference Baseline Benchmark"
echo "================================================"
echo "Model:       ${MODEL_PATH}"
echo "Vocab:       ${VOCAB_PATH}"
echo "Max tokens:  ${MAX_NEW_TOKENS}"
echo "Warmup:      ${WARMUP}"
echo "Repeat:      ${REPEAT}"
echo "Results:     ${RESULTS_DIR}/${RUN_NAME}"
echo "================================================"

# Check prerequisites
if [ ! -f "${BIN_DIR}/demo" ]; then
  echo "ERROR: demo binary not found at ${BIN_DIR}/demo"
  echo "Please build the project first."
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: Model file not found: ${MODEL_PATH}"
  exit 1
fi

if [ ! -f "${VOCAB_PATH}" ]; then
  echo "ERROR: Tokenizer not found: ${VOCAB_PATH}"
  exit 1
fi

# Create results directory
mkdir -p "${RESULTS_DIR}/${RUN_NAME}"

# Build command
CMD=("${BIN_DIR}/demo"
  --model "${MODEL_PATH}"
  --tokenizer "${VOCAB_PATH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --seed "${SEED}"
  --benchmark
  --greedy
  --no-stream-output
)

if [ -n "${PROMPT_FILE}" ]; then
  CMD+=(--prompt-file "${PROMPT_FILE}")
else
  CMD+=(--prompt "${PROMPT}")
fi

OUTPUT_JSON="${RESULTS_DIR}/${RUN_NAME}/results.json"
CMD+=(--output "${OUTPUT_JSON}")

# Run
echo ""
echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee "${RESULTS_DIR}/${RUN_NAME}/console.log"

echo ""
echo "================================================"
echo "Results saved to:"
echo "  JSON:       ${OUTPUT_JSON}"
echo "  Console:    ${RESULTS_DIR}/${RUN_NAME}/console.log"
echo "================================================"

# Create symlink to latest
ln -sfn "${RUN_NAME}" "${RESULTS_DIR}/latest"
echo "Latest symlink: ${RESULTS_DIR}/latest -> ${RUN_NAME}"

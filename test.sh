# 清除上一次测试结果
rm -rf profile/results/baseline/*/ profile/results/baseline/latest profile/nsys/* 2>/dev/null

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIR="profile/results/baseline/${TIMESTAMP}"
mkdir -p "$DIR"
ln -sfn "$TIMESTAMP" profile/results/baseline/latest

# benchmark
./bin/demo \
  --benchmark \
  --max-new-tokens 128 \
  --warmup 3 \
  --repeat 10 \
  --greedy \
  --no-early-stop \
  --no-stream-output \
  --output "${DIR}/results.json" 2>&1 | tee "${DIR}/console.log"

# nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --output=profile/nsys/qwen_baseline_512 \
  ./bin/demo \
    --benchmark \
    --max-new-tokens 128 \
    --warmup 1 \
    --repeat 1 \
    --greedy \
    --no-early-stop \
    --no-stream-output


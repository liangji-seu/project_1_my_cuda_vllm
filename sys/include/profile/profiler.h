#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/base.h"

namespace profile {

// ============================================================
// CUDA Event RAII wrapper — start/stop a named timer on a stream
// ============================================================
class CudaTimer {
 public:
  explicit CudaTimer(const std::string& name);
  ~CudaTimer();

  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;
  CudaTimer(CudaTimer&& other) noexcept;
  CudaTimer& operator=(CudaTimer&& other) noexcept;

  // Record start on the given stream (nullptr = default stream / synchronize).
  void record_start(cudaStream_t stream = nullptr);
  // Record stop on the given stream.
  void record_stop(cudaStream_t stream = nullptr);
  // Synchronize and return elapsed time in milliseconds.
  float elapsed_ms(bool sync = true);
  // Return name for reporting.
  const std::string& name() const { return name_; }
  // Check if events are created and valid.
  bool is_valid() const { return created_; }

 private:
  std::string name_;
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t stop_event_ = nullptr;
  bool started_ = false;
  bool stopped_ = false;
  bool created_ = false;
};

// ============================================================
// Stage-level profiling data
// ============================================================
//每个阶段的耗时
struct StageRecord {
  std::string name;
  int32_t calls = 0;
  float total_ms = 0.0f;
  float avg_ms = 0.0f;
};

// 每个层级的耗时
struct LayerModuleRecord {
  std::string module_name;
  std::string stage;      // "prefill" or "decode"
  int32_t layer_idx = -1;  // -1 = aggregated / non-layer module
  int32_t calls = 0;
  float total_ms = 0.0f;
  float avg_us = 0.0f;
  float percentage = 0.0f;  // of total GPU time for this stage
};

// 单次推理的结果
struct RunRecord {
  int32_t prompt_tokens = 0;
  int32_t output_tokens = 0;

  float tokenizer_encode_ms = 0.0f;
  float model_load_ms = 0.0f;
  float prefill_ms = 0.0f;
  float decode_total_ms = 0.0f;
  float sampling_ms = 0.0f;
  float e2e_ms = 0.0f;  // end-to-end wall clock

  // Inter-token latencies (ms) for each decode step
  std::vector<float> itl_ms;
};

// Full benchmark result
//平均的benchmark
struct BenchmarkResult {
  // Environment
  std::string gpu_name;
  std::string cuda_version;
  std::string model_path;
  std::string precision = "fp32";
  std::string git_commit;

  // Config
  int32_t prompt_tokens = 0;
  int32_t max_new_tokens = 128;
  int32_t warmup = 3;
  int32_t repeat = 10;
  bool greedy = true;
  int32_t seed = 42;

  // Summary (averages over repeat runs)
  float model_load_time_ms = 0.0f;
  float tokenizer_encode_time_ms = 0.0f;
  float prefill_time_ms = 0.0f;
  float decode_time_ms = 0.0f;
  int32_t output_tokens = 0;
  float ttft_model_only_ms = 0.0f;   // prefill → first token
  float ttft_end_to_end_ms = 0.0f;   // request start → first token
  float tpot_ms = 0.0f;             // decode_time / (output_tokens - 1)
  float prefill_throughput_tps = 0.0f;
  float decode_throughput_tps = 0.0f;
  float e2e_latency_ms = 0.0f;
  float e2e_throughput_tps = 0.0f;

  // ITL stats (mean/min/max/p50/p90/p95/p99)
  float mean_itl_ms = 0.0f;
  float min_itl_ms = 0.0f;
  float max_itl_ms = 0.0f;
  float p50_itl_ms = 0.0f;
  float p90_itl_ms = 0.0f;
  float p95_itl_ms = 0.0f;
  float p99_itl_ms = 0.0f;

  // GPU memory
  size_t gpu_memory_before_model_mb = 0;
  size_t gpu_memory_after_model_mb = 0;
  size_t gpu_memory_peak_mb = 0;

  // Per-run data
  std::vector<RunRecord> runs;

  // Stage profile (for normal benchmark mode)
  std::vector<StageRecord> stage_profile;

  // Layer profile (for --layer-profile mode)
  std::vector<LayerModuleRecord> layer_profile;

  // Output as JSON string
  std::string to_json() const;
};

// ============================================================
// Main Profiler — collects timing across the inference pipeline
// ============================================================
class Profiler {
 public:
  Profiler();

  // 显存统计
  void record_memory_before_model();//这3个是对cudaMemGetInfo的包装
  void record_memory_after_model();
  void record_peak_memory();
  size_t memory_before_mb() const { return mem_before_mb_; }
  size_t memory_after_mb() const { return mem_after_mb_; }
  size_t memory_peak_mb() const { return mem_peak_mb_; }

  //CPU时间计时打点
  // ---- CPU-wall-clock timing (for phases where GPU isn't involved) ----
  void set_cpu_start();
  float cpu_elapsed_ms() const;

  //GPU时间计时打点
  // ---- GPU cudaEvent timers ----
  // Create/get a named timer that lives for the profiler's lifetime.
  CudaTimer* get_timer(const std::string& name);

  // ---- Layer profiling ----
  // Record a layer module timing. Called by the model during forward() when
  // layer profiling is enabled.
  // 一个层跑完了，记录数据
  void add_layer_record(const std::string& module, const std::string& stage,
                        int32_t layer_idx, float elapsed_ms);

  // ---- Stage profiling (lightweight, no per-layer sync) ----
  void set_stage(const std::string& stage_name);
  void record_stage_time(const std::string& stage_name, float elapsed_ms);

  // ---- Benchmark result generation ----
  void add_run(const RunRecord& run);//一次完整的推理结束，记录数据


  //计算平均的benchmark
  BenchmarkResult compute_result(const std::string& model_path,
                                 int32_t prompt_tokens, int32_t max_new_tokens,
                                 int32_t warmup_iterations, int32_t repeat_iterations,
                                 bool greedy, int32_t seed);

  // 查询相关信息
  const std::vector<RunRecord>& runs() const { return runs_; }
  void clear_runs() { runs_.clear(); }
  const std::vector<LayerModuleRecord>& layer_records() const { return layer_records_; }
  const std::vector<StageRecord>& stage_records() const { return stage_records_; }

  // ---- Streaming control ----
  bool stream_output_enabled() const { return stream_output_; }
  void set_stream_output(bool v) { stream_output_ = v; }

  bool layer_profile_enabled() const { return layer_profile_enabled_; }
  void set_layer_profile_enabled(bool v) { layer_profile_enabled_ = v; }

 private:
 //三次 cudaMemGetInfo 的快照
  size_t mem_before_mb_ = 0;
  size_t mem_after_mb_ = 0;
  size_t mem_peak_mb_ = 0;

  //chrono 起点，给 tokenizer 等纯 CPU 工作计时
  std::chrono::steady_clock::time_point cpu_start_;

  //	map<string, CudaTimer>，按名字缓存 GPU 计时器
  std::map<std::string, std::unique_ptr<CudaTimer>> timers_;

  //每次单次推理的原始数据：
  //一次完整 prompt → 最后一个 token 输出，包含 prefill + 全部 decode step
  std::vector<RunRecord> runs_;
  std::vector<LayerModuleRecord> layer_records_;//每层耗时汇总
  std::vector<StageRecord> stage_records_;//每个阶段性耗时汇总

  bool stream_output_ = true;//是否是流式输出
  bool layer_profile_enabled_ = false;//是否开启每层同步计时

  // Helper for percentiles
  static float percentile(const std::vector<float>& sorted, float p);
};

}  // namespace profile

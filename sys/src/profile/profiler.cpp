#include "profile/profiler.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <nlohmann/json.hpp>

namespace profile {

// ============================================================
// CudaTimer
// ============================================================
CudaTimer::CudaTimer(const std::string& name) : name_(name) {
  auto err1 = cudaEventCreate(&start_event_);
  auto err2 = cudaEventCreate(&stop_event_);
  if (err1 == cudaSuccess && err2 == cudaSuccess) {
    created_ = true;
  }
}

CudaTimer::~CudaTimer() {
  if (start_event_) cudaEventDestroy(start_event_);
  if (stop_event_) cudaEventDestroy(stop_event_);
}

CudaTimer::CudaTimer(CudaTimer&& other) noexcept
    : name_(std::move(other.name_)),
      start_event_(other.start_event_),
      stop_event_(other.stop_event_),
      started_(other.started_),
      stopped_(other.stopped_),
      created_(other.created_) {
  other.start_event_ = nullptr;
  other.stop_event_ = nullptr;
  other.created_ = false;
}

CudaTimer& CudaTimer::operator=(CudaTimer&& other) noexcept {
  if (this != &other) {
    if (start_event_) cudaEventDestroy(start_event_);
    if (stop_event_) cudaEventDestroy(stop_event_);
    name_ = std::move(other.name_);
    start_event_ = other.start_event_;
    stop_event_ = other.stop_event_;
    started_ = other.started_;
    stopped_ = other.stopped_;
    created_ = other.created_;
    other.start_event_ = nullptr;
    other.stop_event_ = nullptr;
    other.created_ = false;
  }
  return *this;
}

void CudaTimer::record_start(cudaStream_t stream) {
  if (!created_) return;
  cudaEventRecord(start_event_, stream);
  started_ = true;
}

void CudaTimer::record_stop(cudaStream_t stream) {
  if (!created_ || !started_) return;
  cudaEventRecord(stop_event_, stream);
  stopped_ = true;
}

float CudaTimer::elapsed_ms(bool sync) {
  if (!created_ || !started_ || !stopped_) return 0.0f;
  if (sync) cudaEventSynchronize(stop_event_);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start_event_, stop_event_);
  return ms;
}

// ============================================================
// Profiler
// ============================================================
Profiler::Profiler() { cpu_start_ = std::chrono::steady_clock::now(); }

void Profiler::record_memory_before_model() {
  size_t free = 0, total = 0;
  cudaMemGetInfo(&free, &total);
  mem_before_mb_ = (total - free) / (1024 * 1024);
}

void Profiler::record_memory_after_model() {
  size_t free = 0, total = 0;
  cudaMemGetInfo(&free, &total);
  mem_after_mb_ = (total - free) / (1024 * 1024);
}

void Profiler::record_peak_memory() {
  // Use max of before/after as peak estimate since we don't track dynamically.
  mem_peak_mb_ = std::max(mem_before_mb_, mem_after_mb_);
}

void Profiler::set_cpu_start() {
  cpu_start_ = std::chrono::steady_clock::now();
}

float Profiler::cpu_elapsed_ms() const {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<float, std::milli>(now - cpu_start_).count();
}

CudaTimer* Profiler::get_timer(const std::string& name) {
  auto it = timers_.find(name);
  if (it != timers_.end()) return it->second.get();
  auto timer = std::make_unique<CudaTimer>(name);
  auto* ptr = timer.get();
  timers_[name] = std::move(timer);
  return ptr;
}

void Profiler::add_layer_record(const std::string& module, const std::string& stage,
                                int32_t layer_idx, float elapsed_ms) {
  // Find if there's already a record for this module+stage+layer
  for (auto& rec : layer_records_) {
    if (rec.module_name == module && rec.stage == stage &&
        rec.layer_idx == layer_idx) {
      rec.calls++;
      rec.total_ms += elapsed_ms;
      rec.avg_us = (rec.total_ms / rec.calls) * 1000.0f;
      return;
    }
  }
  LayerModuleRecord rec;
  rec.module_name = module;
  rec.stage = stage;
  rec.layer_idx = layer_idx;
  rec.calls = 1;
  rec.total_ms = elapsed_ms;
  rec.avg_us = elapsed_ms * 1000.0f;
  layer_records_.push_back(std::move(rec));
}

void Profiler::set_stage(const std::string& stage_name) {
  // no-op — stage is tracked per-record via record_stage_time
  (void)stage_name;
}

void Profiler::record_stage_time(const std::string& stage_name, float elapsed_ms) {
  for (auto& rec : stage_records_) {
    if (rec.name == stage_name) {
      rec.calls++;
      rec.total_ms += elapsed_ms;
      rec.avg_ms = rec.total_ms / rec.calls;
      return;
    }
  }
  StageRecord rec;
  rec.name = stage_name;
  rec.calls = 1;
  rec.total_ms = elapsed_ms;
  rec.avg_ms = elapsed_ms;
  stage_records_.push_back(std::move(rec));
}

void Profiler::add_run(const RunRecord& run) { runs_.push_back(run); }

// Compute percentiles from a sorted vector
float Profiler::percentile(const std::vector<float>& sorted, float p) {
  if (sorted.empty()) return 0.0f;
  if (p <= 0.0f) return sorted.front();
  if (p >= 100.0f) return sorted.back();
  float idx = (p / 100.0f) * static_cast<float>(sorted.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) return sorted[lo];
  float frac = idx - static_cast<float>(lo);
  return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

BenchmarkResult Profiler::compute_result(const std::string& model_path,
                                         int32_t prompt_tokens, int32_t max_new_tokens,
                                         int32_t warmup_iterations, int32_t repeat_iterations,
                                         bool greedy, int32_t seed) {
  (void)warmup_iterations;  // warmup runs are filtered before calling add_run()

  BenchmarkResult r;
  r.model_path = model_path;
  r.prompt_tokens = prompt_tokens;
  r.max_new_tokens = max_new_tokens;
  r.warmup = warmup_iterations;
  r.repeat = repeat_iterations;
  r.greedy = greedy;
  r.seed = seed;

  // GPU info
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    r.gpu_name = prop.name;
  }
  int cuda_ver = 0;
  if (cudaRuntimeGetVersion(&cuda_ver) == cudaSuccess) {
    r.cuda_version = std::to_string(cuda_ver / 1000) + "." +
                     std::to_string((cuda_ver % 1000) / 10);
  }

  // Memory
  r.gpu_memory_before_model_mb = mem_before_mb_;
  r.gpu_memory_after_model_mb = mem_after_mb_;
  r.gpu_memory_peak_mb = mem_peak_mb_;

  // Aggregate run metrics
  if (!runs_.empty()) {
    double prefill_sum = 0, decode_sum = 0, ttft_sum = 0, e2e_sum = 0, tokenizer_sum = 0;
    double model_load_sum = 0;
    int32_t total_prompt = 0, total_output = 0;
    std::vector<float> all_itl;

    for (const auto& run : runs_) {
      prefill_sum += run.prefill_ms;
      decode_sum += run.decode_total_ms;
      ttft_sum += run.prefill_ms;  // TTFT model-only = prefill time
      e2e_sum += run.e2e_ms;
      tokenizer_sum += run.tokenizer_encode_ms;
      model_load_sum += run.model_load_ms;
      total_prompt += run.prompt_tokens;
      total_output += run.output_tokens;
      for (auto v : run.itl_ms) all_itl.push_back(v);
    }

    int32_t n = static_cast<int32_t>(runs_.size());
    r.prompt_tokens = total_prompt / n;
    r.output_tokens = total_output / n;
    r.model_load_time_ms = static_cast<float>(model_load_sum / n);
    r.tokenizer_encode_time_ms = static_cast<float>(tokenizer_sum / n);
    r.prefill_time_ms = static_cast<float>(prefill_sum / n);
    r.decode_time_ms = static_cast<float>(decode_sum / n);
    r.e2e_latency_ms = static_cast<float>(e2e_sum / n);

    // TTFT: model-only = prefill time; end-to-end = encode + prefill
    r.ttft_model_only_ms = r.prefill_time_ms;
    r.ttft_end_to_end_ms = r.tokenizer_encode_time_ms + r.prefill_time_ms;

    // Throughput
    if (r.prefill_time_ms > 0) {
      r.prefill_throughput_tps =
          static_cast<float>(r.prompt_tokens) / r.prefill_time_ms * 1000.0f;
    }
    // TPOT and decode throughput (safe for output_tokens=1)
    int32_t decode_tokens = std::max(int32_t{1}, r.output_tokens - 1);
    r.tpot_ms = r.decode_time_ms / static_cast<float>(decode_tokens);
    if (r.decode_time_ms > 0) {
      r.decode_throughput_tps =
          static_cast<float>(decode_tokens) / r.decode_time_ms * 1000.0f;
    }
    if (r.e2e_latency_ms > 0) {
      r.e2e_throughput_tps =
          static_cast<float>(r.output_tokens) / r.e2e_latency_ms * 1000.0f;
    }

    // ITL stats
    if (!all_itl.empty()) {
      std::sort(all_itl.begin(), all_itl.end());
      r.mean_itl_ms = 0;
      for (auto v : all_itl) r.mean_itl_ms += v;
      r.mean_itl_ms /= static_cast<float>(all_itl.size());
      r.min_itl_ms = all_itl.front();
      r.max_itl_ms = all_itl.back();
      r.p50_itl_ms = percentile(all_itl, 50);
      r.p90_itl_ms = percentile(all_itl, 90);
      r.p95_itl_ms = percentile(all_itl, 95);
      r.p99_itl_ms = percentile(all_itl, 99);
    }
  } else {
    // Output tokens from max_new_tokens when no runs (shouldn't happen)
    r.output_tokens = max_new_tokens;
  }

  // Compute layer profile percentages
  if (!layer_records_.empty()) {
    // Calculate total GPU time per stage
    float prefill_total = 0.0f, decode_total = 0.0f;
    for (const auto& rec : layer_records_) {
      if (rec.stage == "prefill") prefill_total += rec.total_ms;
      else if (rec.stage == "decode") decode_total += rec.total_ms;
    }
    r.layer_profile = layer_records_;
    for (auto& rec : r.layer_profile) {
      float stage_total = (rec.stage == "prefill") ? prefill_total : decode_total;
      if (stage_total > 0) {
        rec.percentage = rec.total_ms / stage_total * 100.0f;
      }
    }
  }

  r.stage_profile = stage_records_;
  r.runs = runs_;
  return r;
}

// ============================================================
// BenchmarkResult JSON serialization
// ============================================================
std::string BenchmarkResult::to_json() const {
  nlohmann::json j;

  // Environment
  j["environment"]["gpu_name"] = gpu_name;
  j["environment"]["cuda_runtime_version"] = cuda_version;
  j["environment"]["model"] = model_path;
  j["environment"]["precision"] = precision;
  j["environment"]["git_commit"] = git_commit;

  // Config
  j["config"]["prompt_tokens"] = prompt_tokens;
  j["config"]["max_new_tokens"] = max_new_tokens;
  j["config"]["warmup"] = warmup;
  j["config"]["repeat"] = repeat;
  j["config"]["greedy"] = greedy;
  j["config"]["seed"] = seed;

  // Summary
  auto& s = j["summary"];
  s["model_load_time_ms"] = model_load_time_ms;
  s["tokenizer_encode_time_ms"] = tokenizer_encode_time_ms;
  s["prompt_tokens"] = prompt_tokens;
  s["output_tokens"] = output_tokens;
  s["prefill_time_ms"] = prefill_time_ms;
  s["ttft_model_only_ms"] = ttft_model_only_ms;
  s["ttft_end_to_end_ms"] = ttft_end_to_end_ms;
  s["decode_time_ms"] = decode_time_ms;
  s["tpot_ms"] = tpot_ms;
  s["prefill_throughput_tokens_per_s"] = prefill_throughput_tps;
  s["decode_throughput_tokens_per_s"] = decode_throughput_tps;
  s["end_to_end_latency_ms"] = e2e_latency_ms;
  s["end_to_end_throughput_tokens_per_s"] = e2e_throughput_tps;
  s["mean_itl_ms"] = mean_itl_ms;
  s["min_itl_ms"] = min_itl_ms;
  s["max_itl_ms"] = max_itl_ms;
  s["p50_itl_ms"] = p50_itl_ms;
  s["p90_itl_ms"] = p90_itl_ms;
  s["p95_itl_ms"] = p95_itl_ms;
  s["p99_itl_ms"] = p99_itl_ms;
  s["gpu_memory_before_model_mb"] = gpu_memory_before_model_mb;
  s["gpu_memory_after_model_mb"] = gpu_memory_after_model_mb;
  s["gpu_memory_peak_mb"] = gpu_memory_peak_mb;

  // Per-run records
  for (size_t i = 0; i < runs.size(); ++i) {
    nlohmann::json rj;
    const auto& run = runs[i];
    rj["run_index"] = i;
    rj["prompt_tokens"] = run.prompt_tokens;
    rj["output_tokens"] = run.output_tokens;
    rj["tokenizer_encode_ms"] = run.tokenizer_encode_ms;
    rj["prefill_ms"] = run.prefill_ms;
    rj["decode_total_ms"] = run.decode_total_ms;
    rj["sampling_ms"] = run.sampling_ms;
    rj["e2e_ms"] = run.e2e_ms;
    if (!run.itl_ms.empty()) {
      rj["itl_ms"] = run.itl_ms;
    }
    j["runs"].push_back(rj);
  }

  // Stage profile
  for (const auto& sp : stage_profile) {
    nlohmann::json sj;
    sj["stage"] = sp.name;
    sj["calls"] = sp.calls;
    sj["total_ms"] = sp.total_ms;
    sj["avg_ms"] = sp.avg_ms;
    j["stage_profile"].push_back(sj);
  }

  // Layer profile
  for (const auto& lp : layer_profile) {
    nlohmann::json lj;
    lj["module"] = lp.module_name;
    lj["stage"] = lp.stage;
    lj["layer_idx"] = lp.layer_idx;
    lj["calls"] = lp.calls;
    lj["total_ms"] = lp.total_ms;
    lj["avg_us"] = lp.avg_us;
    lj["percentage"] = lp.percentage;
    j["layer_profile"].push_back(lj);
  }

  return j.dump(2);
}

}  // namespace profile

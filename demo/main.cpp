#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "model/llama3.h"
#include "profile/profiler.h"

// ============================================================
// Default paths
// ============================================================
#define DEFAULT_MODEL_PATH "/home/liangji/AI_INFRA/projects/my_cuda_vllm/demo/qwen2.5_0.5b_instruct.bin"
#define DEFAULT_VOCAB_PATH "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json"
//#define DEFAULT_PROMPT     "请从大语言模型推理系统的角度，系统分析一个基于 C++ 和 CUDA 从零实现的自回归推理框架。该框架目前支持 Qwen2.5-0.5B-Instruct，已经完成模型权重加载、Tokenizer、Embedding、RMSNorm、RoPE、KV Cache、Multi-Head Attention、SwiGLU、线性投影、LM Head 和贪心采样，并能够在单张 RTX 4090 上完成连续文本生成。框架使用 RAII 管理资源，通过 Buffer、Tensor、Operator、Layer 和 Model 等模块组织代码，权重文件通过 mmap 映射到 CPU 虚拟地址空间，再复制到 GPU 显存。模型的中间张量和 KV Cache 在初始化阶段预先分配，推理过程分为 prefill 和 decode 两个阶段。当前框架使用 FP32 计算，batch size 为 1。性能测试发现，decode 阶段的每个输出 token 耗时约为数毫秒。Nsight Systems 显示，矩阵乘和注意力 kernel 占据了大部分 GPU kernel 执行时间，同时推理过程中还存在 cudaMemcpy、cudaMalloc、cudaFree 和 CPU-GPU 同步操作。LM Head 会生成完整词表大小的 logits，之后将 logits 从 GPU 拷贝到 CPU，再由 CPU 完成 argmax 采样。部分临时 Tensor 在设备转换时会重新创建 Buffer，并通过智能指针析构旧 Buffer，这可能引入额外的显存申请和释放开销。请分别从端到端指标、运行时调度、显存管理、数据传输、算子实现和模型结构六个方面分析该框架可能存在的性能瓶颈。首先解释 TTFT、TPOT、prefill throughput、decode throughput 和端到端延迟分别反映什么问题。然后说明如何使用 Nsight Systems 判断 CPU-GPU 同步、kernel launch、显存申请释放和数据搬运是否成为瓶颈，以及如何使用 Nsight Compute 分析矩阵乘和注意力 kernel 的计算吞吐、显存带宽、缓存命中率、occupancy 和 warp stall。最后，请给出一个分阶段的优化路线"
#define DEFAULT_PROMPT "你给我讲一个小红帽的故事"
// ============================================================
// CLI argument parsing (no third-party deps)
// ============================================================
struct CliArgs {
  std::string model_path = DEFAULT_MODEL_PATH;
  std::string vocab_path = DEFAULT_VOCAB_PATH;
  std::string prompt = DEFAULT_PROMPT;
  std::string prompt_file;
  std::string output;
  int32_t max_new_tokens = 128;
  int32_t warmup = 3;
  int32_t repeat = 10;
  int32_t seed = 42;
  bool benchmark = false;
  bool layer_profile = false;
  bool profile = false;
  bool greedy = true;
  bool no_stream_output = false;
  bool no_early_stop = false;
};

static void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("Options:\n");
  printf("  --model <path>          Model .bin file path\n");
  printf("  --tokenizer <path>      Tokenizer .json file path\n");
  printf("  --prompt <text>         Input prompt text\n");
  printf("  --prompt-file <path>    Read prompt from file\n");
  printf("  --max-new-tokens <N>    Max tokens to generate (default: 128)\n");
  printf("  --warmup <N>            Warmup iterations (default: 3)\n");
  printf("  --repeat <N>            Benchmark repeat iterations (default: 10)\n");
  printf("  --seed <N>              Random seed (default: 42)\n");
  printf("  --benchmark             Enable benchmark mode\n");
  printf("  --layer-profile         Enable per-layer module timing\n");
  printf("  --profile               Alias for --benchmark\n");
  printf("  --greedy                Use greedy decoding (default)\n");
  printf("  --no-stream-output      Suppress per-token output in benchmark\n");
  printf("  --no-early-stop         Ignore end token, generate all max-new-tokens\n");
  printf("  --output <path>         Write results JSON to file\n");
  printf("  --help                  Show this help\n");
}

static CliArgs parse_args(int argc, char* argv[]) {
  CliArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 < argc) return argv[++i];
      LOG(FATAL) << "Missing value for argument: " << arg;
      return "";
    };

    if (arg == "--model")           args.model_path = next();
    else if (arg == "--tokenizer")  args.vocab_path = next();
    else if (arg == "--prompt")     args.prompt = next();
    else if (arg == "--prompt-file") args.prompt_file = next();
    else if (arg == "--max-new-tokens") args.max_new_tokens = std::stoi(next());
    else if (arg == "--warmup")     args.warmup = std::stoi(next());
    else if (arg == "--repeat")     args.repeat = std::stoi(next());
    else if (arg == "--seed")       args.seed = std::stoi(next());
    else if (arg == "--output")     args.output = next();
    else if (arg == "--benchmark")  args.benchmark = true;
    else if (arg == "--layer-profile") args.layer_profile = true;
    else if (arg == "--profile")    args.benchmark = true;
    else if (arg == "--greedy")     args.greedy = true;
    else if (arg == "--no-stream-output") args.no_stream_output = true;
    else if (arg == "--no-early-stop") args.no_early_stop = true;
    else if (arg == "--help")       { print_usage(argv[0]); exit(0); }
    else {
      // Backward compatibility: positional model and vocab paths
      if (i == 1) args.model_path = arg;
      else if (i == 2) args.vocab_path = arg;
      else LOG(FATAL) << "Unknown argument: " << arg;
    }
  }
  return args;
}

// ============================================================
// Prompt reading
// ============================================================
static std::string read_prompt(const CliArgs& args) {
  if (!args.prompt_file.empty()) {
    std::ifstream f(args.prompt_file);
    if (!f.is_open()) {
      LOG(FATAL) << "Cannot open prompt file: " << args.prompt_file;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }
  return args.prompt;
}

// 简单 ChatML 包装
static std::string wrap_chatml(const std::string& user_input) {
  std::string prompt;
  prompt += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
  prompt += "<|im_start|>user\n";
  prompt += user_input;
  prompt += "<|im_end|>\n";
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

// ============================================================
// Inference function — returns output tokens and populates
// a RunRecord with timing data using the Profiler.
// ============================================================
static std::vector<int32_t> generate(
    model::LLama2Model& model,
    const std::string& sentence,
    int total_steps,
    profile::Profiler* profiler,
    bool stream_output,
    bool is_benchmark,
  bool force_all = false) {

  // ---- Tokenizer encode ----
  profiler->set_cpu_start();
  auto tokens = model.encode(sentence);
  int32_t prompt_len = static_cast<int32_t>(tokens.size());
  float tokenizer_ms = profiler->cpu_elapsed_ms();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  auto& pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  cudaStream_t stream = nullptr;
  // If on GPU, grab the CUDA stream for cudaEvent timing
  {
    auto& test_buf = model.get_buffer(model::ModelBufferType::kQuery);
    if (test_buf.get_device_type() == base::DeviceType_t::GPU) {
      // Stream not directly accessible, model manages it internally.
    }
  }

  std::vector<int32_t> words;
  std::vector<float> itl_ms;

  // For benchmark timing
  profile::CudaTimer* timer_prefill = nullptr;
  profile::CudaTimer* timer_decode_step = nullptr;
  profile::CudaTimer* timer_e2e = nullptr;
  if (is_benchmark) {
    timer_prefill = profiler->get_timer("prefill");
    timer_decode_step = profiler->get_timer("decode_step");
    timer_e2e = profiler->get_timer("e2e");
  }

  bool first_decode = true;

  // ---- E2E start ----
  profiler->set_cpu_start();
  if (timer_e2e) timer_e2e->record_start();

  while (pos < total_steps) {
    pos_tensor.peek_index<int32_t>(0) = pos;

    if (pos < prompt_len - 1) {
      // ---- Prefill step ----
      if (is_benchmark && pos == 0 && timer_prefill) {
        timer_prefill->record_start();
      }

      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);

      if (is_benchmark && pos == prompt_len - 2 && timer_prefill) {
        timer_prefill->record_stop();
      }

    } else {
      // ---- Decode step ----
      is_prompt = false;
      if (first_decode) {
        first_decode = false;
        // Prefill just finished — record prefill end
      }

      // Per-step decode timing
      profiler->set_cpu_start();
      if (timer_decode_step) timer_decode_step->record_start();

      // 首个 decode step: 从 CPU prefill token 过渡, 走 embedding(tokens)
      // 后续 decode steps: token 已在 GPU (post_processing 闭环写入), 走 embed_next_token 零拷贝
      op::EmbeddingOutput token_embedding;
      if (first_decode) {
        tokens = std::vector<int32_t>{next};
        token_embedding = model.embedding(tokens);
      } else {
        token_embedding = model.embed_next_token(next);
      }
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);

      if (timer_decode_step) timer_decode_step->record_stop();

      // Measure per-token decode latency
      // cudaEventSynchronize needed for accurate GPU timing
      if (timer_decode_step) {
        float step_ms = timer_decode_step->elapsed_ms(true);
        itl_ms.push_back(step_ms);
      } else {
        float step_ms = profiler->cpu_elapsed_ms();
        itl_ms.push_back(step_ms);
      }
    }

    if (!force_all && model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
      // Decode and stream output
      if (stream_output) {
        std::string piece = model.decode(next);
        std::cout << piece << std::flush;
      }
    }
    pos += 1;
  }

  if (timer_e2e) timer_e2e->record_stop();

  // ---- Final sync for E2E ----
  float e2e_ms = 0;
  if (timer_e2e) {
    e2e_ms = timer_e2e->elapsed_ms(true);
  } else {
    e2e_ms = profiler->cpu_elapsed_ms();
  }

  if (stream_output) {
    std::cout << std::endl;
    // Optionally decode all
    std::string decoded = model.decode(words);
    // Already streamed, don't re-print
  }

  // ---- Populate run record ----
  profile::RunRecord run;
  run.prompt_tokens = prompt_len;
  run.output_tokens = static_cast<int32_t>(words.size());
  run.tokenizer_encode_ms = tokenizer_ms;
  run.model_load_ms = 0;  // measured separately
  run.e2e_ms = e2e_ms;

  if (timer_prefill) {
    run.prefill_ms = timer_prefill->elapsed_ms(false);
  }
  // Decode total = sum of all ITL measurements
  float decode_sum = 0;
  for (auto v : itl_ms) decode_sum += v;
  run.decode_total_ms = decode_sum;

  // Sampling time is included in decode step — approximate as 0 for now
  run.sampling_ms = 0.0f;
  run.itl_ms = itl_ms;

  profiler->add_run(run);
  return words;
}














// ============================================================
// Print human-readable benchmark summary
// ============================================================
static void print_summary(const profile::BenchmarkResult& r) {
  printf("\n");
  printf("==================================================\n");
  printf("Inference Benchmark\n");
  printf("==================================================\n");
  printf("Model:        %s\n", r.model_path.c_str());
  printf("GPU:          %s\n", r.gpu_name.c_str());
  printf("Precision:    %s\n", r.precision.c_str());
  printf("Prompt tokens:%d\n", r.prompt_tokens);
  printf("Output tokens:%d\n", r.output_tokens);
  printf("Warmup:       %d\n", r.warmup);
  printf("Repeat:       %d\n", r.repeat);
  printf("\n");
  printf("Model load:          %8.2f ms\n", r.model_load_time_ms);
  printf("TTFT model-only:     %8.2f ms\n", r.ttft_model_only_ms);
  printf("TTFT end-to-end:     %8.2f ms\n", r.ttft_end_to_end_ms);
  printf("Prefill:             %8.2f ms\n", r.prefill_time_ms);
  printf("Prefill throughput:  %8.2f tok/s\n", r.prefill_throughput_tps);
  printf("Decode:              %8.2f ms\n", r.decode_time_ms);
  printf("TPOT:                %8.2f ms\n", r.tpot_ms);
  printf("Decode throughput:   %8.2f tok/s\n", r.decode_throughput_tps);
  printf("E2E latency:         %8.2f ms\n", r.e2e_latency_ms);
  printf("E2E throughput:      %8.2f tok/s\n", r.e2e_throughput_tps);
  printf("Peak GPU memory:     %8zu MB\n", r.gpu_memory_peak_mb);
  printf("\n");
  printf("ITL P50/P90/P95/P99: %.2f / %.2f / %.2f / %.2f ms\n",
         r.p50_itl_ms, r.p90_itl_ms, r.p95_itl_ms, r.p99_itl_ms);
  printf("==================================================\n");

  // Stage profile
  if (!r.stage_profile.empty()) {
    printf("\nStage Profile:\n");
    printf("%-24s %8s %12s %12s\n", "Stage", "Calls", "Total(ms)", "Avg(ms)");
    for (const auto& sp : r.stage_profile) {
      printf("%-24s %8d %12.3f %12.3f\n",
             sp.name.c_str(), sp.calls, sp.total_ms, sp.avg_ms);
    }
  }

  // Layer profile
  if (!r.layer_profile.empty()) {
    printf("\nLayer Profile:\n");
    printf("%-30s %-8s %6s %8s %8s %8s %8s\n",
           "Module", "Stage", "Layer", "Calls", "TotalMs", "AvgUs", "Pct");
    for (const auto& lp : r.layer_profile) {
      printf("%-30s %-8s %6d %8d %8.3f %8.1f %7.2f%%\n",
             lp.module_name.c_str(), lp.stage.c_str(), lp.layer_idx,
             lp.calls, lp.total_ms, lp.avg_us, lp.percentage);
    }
  }
  fflush(stdout);
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  auto args = parse_args(argc, argv);
  auto raw_prompt = read_prompt(args);
  // ChatML 包装，让模型知道这是对话
  auto prompt_text = wrap_chatml(raw_prompt);

  // ---- Create profiler ----
  auto profiler = std::make_unique<profile::Profiler>();
  if (args.benchmark) {
    profiler->set_stream_output(!args.no_stream_output);
  }
  if (args.layer_profile) {
    profiler->set_layer_profile_enabled(true);
    args.benchmark = true;  // layer profiling implies benchmark mode
  }

  // ---- Model load timing ----
  profiler->record_memory_before_model();
  profiler->set_cpu_start();

  model::LLama2Model model(base::TokenizerType::kEncodeBpe,
                           args.vocab_path, args.model_path, false);

  auto init_status = model.init(base::DeviceType_t::GPU);
  if (!init_status) {
    LOG(FATAL) << "Model init failed, error code: "
               << static_cast<int>(init_status.get_err_code());
  }

  float model_load_ms = profiler->cpu_elapsed_ms();
  profiler->record_memory_after_model();
  profiler->record_peak_memory();

  if (args.benchmark) {
    model.set_profiler(profiler.get());
  }

  // ---- Normal interactive mode (no benchmark) ----
  if (!args.benchmark) {
    printf("Generating...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    auto words = generate(model, prompt_text, args.max_new_tokens,
                          profiler.get(), true, false, false);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    printf("\nsteps/s:%lf\n",
           static_cast<double>(words.size()) / duration);
    fflush(stdout);
    return 0;
  }

  // ---- Benchmark mode ----
  // 打印 prompt token 数量
  {
    auto prompt_tokens = model.encode(prompt_text);
    printf("Prompt token count: %d\n", static_cast<int>(prompt_tokens.size()));
  }

  // 1. Warmup runs (not recorded)
  if (args.warmup > 0) {
    printf("Warmup (%d iterations)...\n", args.warmup);
    fflush(stdout);
    for (int i = 0; i < args.warmup; ++i) {
      model.set_nvtx_context("W" + std::to_string(i + 1));
      generate(model, prompt_text, args.max_new_tokens, profiler.get(),
               false, false, args.no_early_stop);
    }
    // Clear warmup records (they were added to profiler)
  }

  // Reset profiler runs (discard warmup)
  profiler->clear_runs();

  // 2. Measured runs
  printf("Benchmark (%d iterations)...\n", args.repeat);
  fflush(stdout);
  for (int i = 0; i < args.repeat; ++i) {
    printf("  Run %d/%d", i + 1, args.repeat);
    fflush(stdout);
    model.set_nvtx_context("R" + std::to_string(i + 1));
    generate(model, prompt_text, args.max_new_tokens, profiler.get(),
             false, true, args.no_early_stop);
    const auto& last_run = profiler->runs().back();
    printf(" — %d tokens, %.1f ms\n", last_run.output_tokens, last_run.e2e_ms);
    fflush(stdout);
  }

  // 3. Compute and output results
  auto result = profiler->compute_result(
      args.model_path,
      static_cast<int32_t>(model.encode(prompt_text).size()),
      args.max_new_tokens, args.warmup, args.repeat,
      args.greedy, args.seed);

  result.model_load_time_ms = model_load_ms;
  result.gpu_memory_before_model_mb = profiler->memory_before_mb();
  result.gpu_memory_after_model_mb = profiler->memory_after_mb();
  result.gpu_memory_peak_mb = profiler->memory_peak_mb();

  // Git commit
  {
    FILE* f = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (f) {
      char buf[64] = {};
      if (fgets(buf, sizeof(buf), f)) {
        std::string commit(buf);
        if (!commit.empty() && commit.back() == '\n')
          commit.pop_back();
        result.git_commit = commit;
      }
      pclose(f);
    }
  }

  // Print summary
  print_summary(result);

  // Write JSON output
  if (!args.output.empty()) {
    std::string json_str = result.to_json();
    std::ofstream of(args.output);
    if (of.is_open()) {
      of << json_str;
      of.close();
      printf("\nResults written to: %s\n", args.output.c_str());
    } else {
      LOG(ERROR) << "Cannot open output file: " << args.output;
    }
  }

  return 0;
}

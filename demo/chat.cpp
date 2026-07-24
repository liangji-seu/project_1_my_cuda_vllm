#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "model/llama3.h"
#include "profile/profiler.h"

#define DEFAULT_MODEL_PATH   "/home/liangji/AI_INFRA/projects/my_cuda_vllm/demo/qwen2.5_0.5b_instruct.bin"
#define DEFAULT_VOCAB_PATH   "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json"

// ============================================================
// Simple CLI parser for chat
// ============================================================
struct ChatArgs {
  std::string model_path = DEFAULT_MODEL_PATH;
  std::string vocab_path = DEFAULT_VOCAB_PATH;
  bool benchmark = false;
  bool quant = false;
  int32_t max_new_tokens = 128;
  int32_t warmup = 3;
  int32_t repeat = 10;
  std::string prompt;
  std::string output;
};

static ChatArgs parse_chat_args(int argc, char* argv[]) {
  ChatArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model")           args.model_path = argv[++i];
    else if (arg == "--tokenizer")  args.vocab_path = argv[++i];
    else if (arg == "--benchmark")  args.benchmark = true;
    else if (arg == "--quant")      args.quant = true;
    else if (arg == "--max-new-tokens") args.max_new_tokens = std::stoi(argv[++i]);
    else if (arg == "--warmup")     args.warmup = std::stoi(argv[++i]);
    else if (arg == "--repeat")     args.repeat = std::stoi(argv[++i]);
    else if (arg == "--prompt")     args.prompt = argv[++i];
    else if (arg == "--output")     args.output = argv[++i];
    else {
      // Positional: model_path, vocab_path
      if (i == 1) args.model_path = arg;
      else if (i == 2) args.vocab_path = arg;
    }
  }
  return args;
}

// ============================================================
// ChatML prompt building
// ============================================================
static std::string build_chatml_prompt(const std::vector<std::string>& history) {
  std::string prompt;
  prompt += "system\n";
  prompt += "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  prompt += "<|im_end|>\n";
  for (size_t i = 0; i < history.size(); i += 2) {
    prompt += "<|im_start|>user\n";
    prompt += history[i];
    prompt += "<|im_end|>\n";
    if (i + 1 < history.size()) {
      prompt += "<|im_start|>assistant\n";
      prompt += history[i + 1];
      prompt += "<|im_end|>\n";
    }
  }
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

// ============================================================
// Generation function (same as main.cpp, without profiling)
// ============================================================
static std::vector<int32_t> generate_tokens(
    model::LLama2Model& model, const std::string& prompt,
    int max_steps, std::vector<int32_t>& words_out,
    profile::Profiler* profiler) {
  auto tokens = model.encode(prompt);
  int32_t prompt_len = static_cast<int32_t>(tokens.size());
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  auto& pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  std::vector<float> itl_ms;

  profile::CudaTimer* timer_e2e = nullptr;
  profile::CudaTimer* timer_prefill = nullptr;
  profile::CudaTimer* timer_decode_step = nullptr;
  if (profiler) {
    timer_e2e = profiler->get_timer("e2e");
    timer_prefill = profiler->get_timer("prefill");
    timer_decode_step = profiler->get_timer("decode_step");
    profiler->set_cpu_start();
  }
  if (timer_e2e) timer_e2e->record_start();

  while (pos < max_steps) {
    pos_tensor.peek_index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      // Prefill step
      if (profiler && pos == 0 && timer_prefill) {
        timer_prefill->record_start();
      }
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (profiler && pos == prompt_len - 2 && timer_prefill) {
        timer_prefill->record_stop();
      }
    } else {
      // Decode step
      is_prompt = false;
      if (timer_decode_step) timer_decode_step->record_start();

      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);

      if (timer_decode_step) {
        timer_decode_step->record_stop();
        float ms = timer_decode_step->elapsed_ms(true);
        itl_ms.push_back(ms);
      }
    }
    if (model.is_sentence_ending(next)) break;

    if (is_prompt) {
      next = tokens.at(pos + 1);
    } else {
      words.push_back(next);
    }
    pos += 1;
  }

  if (timer_e2e) timer_e2e->record_stop();
  float e2e_ms = timer_e2e ? timer_e2e->elapsed_ms(true) : 0.0f;

  if (profiler) {
    profile::RunRecord run;
    run.prompt_tokens = prompt_len;
    run.output_tokens = static_cast<int32_t>(words.size());
    run.e2e_ms = e2e_ms;
    run.itl_ms = itl_ms;
    run.prefill_ms = timer_prefill ? timer_prefill->elapsed_ms(false) : 0.0f;

    float decode_sum = 0;
    for (auto v : itl_ms) decode_sum += v;
    run.decode_total_ms = decode_sum;
    profiler->add_run(run);
  }

  words_out = words;
  return words;
}

static std::string generate(model::LLama2Model& model, const std::string& prompt,
                            int max_steps, profile::Profiler* profiler = nullptr) {
  std::vector<int32_t> words;
  generate_tokens(model, prompt, max_steps, words, profiler);

  // Stream per-token decode (for interactive mode, profiler is nullptr)
  if (!profiler) {
    for (size_t i = 0; i < words.size(); ++i) {
      std::string piece = model.decode(words[i]);
      std::cout << piece << std::flush;
    }
  }
  std::cout << std::endl;

  std::string response = model.decode(words);
  size_t end_pos = response.find("<|im_end|>");
  if (end_pos != std::string::npos) {
    response = response.substr(0, end_pos);
  }
  return response;
}

// ============================================================
// Benchmark mode for chat
// ============================================================
static void run_chat_benchmark(const ChatArgs& args) {
  auto profiler = std::make_unique<profile::Profiler>();
  profiler->set_stream_output(false);

  profiler->record_memory_before_model();
  profiler->set_cpu_start();

  model::LLama2Model model(base::TokenizerType::kEncodeBpe,
                           args.vocab_path, args.model_path, args.quant);
  auto init_status = model.init(base::DeviceType_t::GPU);
  if (!init_status) {
    LOG(FATAL) << "Model init failed: " << static_cast<int>(init_status.get_err_code());
  }

  float model_load_ms = profiler->cpu_elapsed_ms();
  profiler->record_memory_after_model();
  profiler->record_peak_memory();

  model.set_profiler(profiler.get());

  std::string prompt_text = args.prompt.empty()
      ? build_chatml_prompt({"Hello, who are you?"}) : args.prompt;

  // Warmup
  printf("Warmup (%d iterations)...\n", args.warmup);
  for (int i = 0; i < args.warmup; ++i) {
    std::vector<int32_t> words;
    generate_tokens(model, prompt_text, args.max_new_tokens, words, nullptr);
  }

  // Reset & benchmark
  profiler->clear_runs();
  printf("Benchmark (%d iterations)...\n", args.repeat);
  for (int i = 0; i < args.repeat; ++i) {
    std::vector<int32_t> words;
    generate_tokens(model, prompt_text, args.max_new_tokens, words, profiler.get());
    const auto& run = profiler->runs().back();
    printf("  Run %d/%d — %d tokens, %.1f ms\n",
           i + 1, args.repeat, run.output_tokens, run.e2e_ms);
  }

  auto result = profiler->compute_result(
      args.model_path,
      static_cast<int32_t>(model.encode(prompt_text).size()),
      args.max_new_tokens, args.warmup, args.repeat, true, 42);
  result.model_load_time_ms = model_load_ms;
  result.gpu_memory_before_model_mb = profiler->memory_before_mb();
  result.gpu_memory_after_model_mb = profiler->memory_after_mb();
  result.gpu_memory_peak_mb = profiler->memory_peak_mb();

  printf("\n==================================================\n");
  printf("Chat Benchmark Summary\n");
  printf("==================================================\n");
  printf("Model:        %s\n", result.model_path.c_str());
  printf("GPU:          %s\n", result.gpu_name.c_str());
  printf("Prompt tokens:%d\n", result.prompt_tokens);
  printf("Output tokens:%d\n", result.output_tokens);
  printf("Warmup/Repeat: %d/%d\n", args.warmup, args.repeat);
  printf("Model load:   %.2f ms\n", result.model_load_time_ms);
  printf("Prefill:      %.2f ms\n", result.prefill_time_ms);
  printf("Decode:       %.2f ms\n", result.decode_time_ms);
  printf("TPOT:         %.2f ms\n", result.tpot_ms);
  printf("E2E latency:  %.2f ms\n", result.e2e_latency_ms);
  printf("Peak GPU mem: %zu MB\n", result.gpu_memory_peak_mb);
  printf("ITL P50/P95/P99: %.2f / %.2f / %.2f ms\n",
         result.p50_itl_ms, result.p95_itl_ms, result.p99_itl_ms);
  printf("==================================================\n");

  if (!args.output.empty()) {
    std::ofstream of(args.output);
    if (of.is_open()) {
      of << result.to_json();
      printf("Results written to: %s\n", args.output.c_str());
    }
  }
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  auto args = parse_chat_args(argc, argv);

  // Model init
  model::LLama2Model model(base::TokenizerType::kEncodeBpe,
                           args.vocab_path, args.model_path, args.quant);
  auto init_status = model.init(base::DeviceType_t::GPU);
  if (!init_status) {
    LOG(FATAL) << "Model init failed: " << static_cast<int>(init_status.get_err_code());
  }

  // Benchmark mode
  if (args.benchmark) {
    run_chat_benchmark(args);
    return 0;
  }

  // Interactive chat mode
  std::cout << "Model loaded. Type 'quit' to exit.\n" << std::endl;

  std::vector<std::string> history;
  std::string input;

  while (true) {
    std::cout << "> " << std::flush;
    std::getline(std::cin, input);
    if (std::cin.eof() || input == "quit") break;
    if (input.empty()) continue;

    history.push_back(input);

    std::string prompt = build_chatml_prompt(history);
    auto start = std::chrono::steady_clock::now();

    std::string response = generate(model, prompt, 512);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    history.push_back(response);
    std::cout << "[ " << duration << "s ]" << std::endl;
  }

  std::cout << "Bye." << std::endl;
  return 0;
}

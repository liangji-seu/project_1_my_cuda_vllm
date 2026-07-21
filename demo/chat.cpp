#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "model/llama3.h"

#define DEFAULT_MODEL_PATH   "/home/liangji/AI_INFRA/projects/my_cuda_vllm/demo/qwen2.5_0.5b_instruct.bin"
#define DEFAULT_VOCAB_PATH   "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json"

static std::string build_chatml_prompt(const std::vector<std::string>& history) {
  // Note: encode() prepends BOS (<|im_start|>), so the prompt itself should
  // NOT start with <|im_start|> to avoid a double BOS token.
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

static std::string generate(model::LLama2Model& model, const std::string& prompt,
                            int max_steps) {
  auto tokens = model.encode(prompt);
  int32_t prompt_len = static_cast<int32_t>(tokens.size());
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  // DEBUG: print prompt info
  LOG(INFO) << "Prompt token count: " << prompt_len;
  LOG(INFO) << "First 5 prompt tokens:";
  for (int i = 0; i < std::min(5, prompt_len); ++i)
    LOG(INFO) << "  [" << i << "] id=" << tokens[i] << " -> '" << model.decode(tokens[i]) << "'";
  LOG(INFO) << "Last 5 prompt tokens:";
  for (int i = std::max(0, prompt_len - 5); i < prompt_len; ++i)
    LOG(INFO) << "  [" << i << "] id=" << tokens[i] << " -> '" << model.decode(tokens[i]) << "'";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  auto& pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < max_steps) {
    pos_tensor.peek_index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
    } else {
      words.push_back(next);
      // DEBUG: print generated token info
      if (words.size() <= 10) {
        LOG(INFO) << "Gen token[" << words.size() - 1 << "] id=" << next
                   << " -> '" << model.decode(next) << "'";
      }
      std::string piece = model.decode(next);
      std::cout << piece << std::flush;
    }
    pos += 1;
  }
  std::cout << std::endl;

  // Decode response and strip end tokens
  std::string response = model.decode(words);
  size_t end_pos = response.find("<|im_end|>");
  if (end_pos != std::string::npos) {
    response = response.substr(0, end_pos);
  }
  return response;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  const char* model_path = (argc >= 2) ? argv[1] : DEFAULT_MODEL_PATH;
  const char* vocab_path  = (argc >= 3) ? argv[2] : DEFAULT_VOCAB_PATH;

  model::LLama2Model model(base::TokenizerType::kEncodeBpe, vocab_path,
                           model_path, false);
  auto init_status = model.init(base::DeviceType_t::CPU);
  if (!init_status) {
    LOG(FATAL) << "Model init failed: " << static_cast<int>(init_status.get_err_code());
  }

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

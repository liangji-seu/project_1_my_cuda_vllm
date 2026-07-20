#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "model/llama3.h"

static std::string build_chatml_prompt(const std::vector<std::string>& history) {
  std::string prompt;
  prompt += "<|im_start|>system\n";
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
      // Decode the latest token and stream to stdout
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

  if (argc != 3) {
    LOG(INFO) << "Usage: ./chat <checkpoint_path> <tokenizer_path>";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];

  model::LLama2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                           checkpoint_path, false);
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

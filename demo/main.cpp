#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "model/llama3.h"

#define DEFAULT_MODEL_PATH "/home/liangji/AI_INFRA/projects/my_cuda_vllm/demo/qwen2.5_0.5b_instruct.bin"
#define DEFAULT_VOCAB_PATH "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json"
#define DEFAULT_PROMPT     "请你给我介绍一下东南大学"

int32_t generate(model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = static_cast<int32_t>(tokens.size());
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  auto& pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
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
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
  if (argc != 1 && argc != 3) {
    LOG(INFO) << "Usage: ./demo [model_path vocab_path]";
    return -1;
  }
  const char* model_path = (argc >= 3) ? argv[1] : DEFAULT_MODEL_PATH;
  const char* vocab_path  = (argc >= 3) ? argv[2] : DEFAULT_VOCAB_PATH;

  model::LLama2Model model(base::TokenizerType::kEncodeBpe, vocab_path,
                           model_path, false);
  auto init_status = model.init(base::DeviceType_t::CPU);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: "
               << static_cast<int>(init_status.get_err_code());
  }

  const std::string sentence = DEFAULT_PROMPT;

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}

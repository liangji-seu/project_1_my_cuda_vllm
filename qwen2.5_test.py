from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


model_path = "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="cuda",
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)


prompt = "请你给我介绍一下东南大学"

messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    {
        "role": "user",
        "content": prompt
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


# tokenize
model_inputs = tokenizer(
    [text],
    return_tensors="pt"
).to("cuda")


input_tokens = model_inputs.input_ids.shape[1]


print("=" * 60)
print(f"Input tokens: {input_tokens}")
print("=" * 60)


# warmup
with torch.no_grad():
    _ = model.generate(
        **model_inputs,
        max_new_tokens=10
    )

torch.cuda.synchronize()


# 清空显存统计
torch.cuda.reset_peak_memory_stats()


# ==========================
# Benchmark start
# ==========================

start_time = time.perf_counter()

first_token_time = None


generated_tokens = []

with torch.no_grad():

    # 使用 streamer 无法精确统计每个token
    # 所以这里用 generate + 记录整体指标

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False
    )


torch.cuda.synchronize()

end_time = time.perf_counter()


# ==========================
# Decode
# ==========================

output_ids = outputs[0][input_tokens:]

output_tokens = len(output_ids)


response = tokenizer.decode(
    output_ids,
    skip_special_tokens=True
)


# ==========================
# Metrics
# ==========================

total_time = end_time - start_time


# generate阶段包含prefill+decode
decode_time = total_time


tps = output_tokens / decode_time

tpot = decode_time / output_tokens * 1000


peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)


print("\n")
print("=" * 60)
print("Benchmark Result")
print("=" * 60)

print(f"Input tokens        : {input_tokens}")
print(f"Output tokens       : {output_tokens}")
print(f"Total tokens        : {input_tokens + output_tokens}")

print("-" * 60)

print(f"Total latency       : {total_time*1000:.2f} ms")

print("-" * 60)

print(f"Decode throughput   : {tps:.2f} tokens/s")
print(f"TPOT                : {tpot:.2f} ms/token")

print("-" * 60)

print(f"Peak GPU memory     : {peak_memory:.2f} GB")

print("=" * 60)


print("\nResponse:")
print(response)
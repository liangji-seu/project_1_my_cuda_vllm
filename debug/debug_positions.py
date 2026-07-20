"""
Minimal script to debug the .bin weight loading position tracking.
Compares actual byte positions against expected positions.
"""
import struct
import torch

BIN_PATH = "/home/liangji/AI_INFRA/projects/my_cuda_vllm/demo/qwen2.5_0.5b_instruct.bin"

with open(BIN_PATH, 'rb') as f:
    data = f.read()

file_size = len(data)
header = struct.unpack('iiiiiiii', data[:32])
print(f"File size: {file_size} bytes")
print(f"Header: dim={header[0]}, hidden_dim={header[1]}, n_layers={header[2]}, n_heads={header[3]}, n_kv_heads={header[4]}, vocab={header[5]}, seq_len={header[6]}, head_dim={header[7]}")

dim = header[0]
hidden_dim = header[1]
layer_num = header[2]
n_heads = header[3]
n_kv_heads = header[4]
vocab_size = abs(header[5])
shared = header[5] > 0
seq_len = header[6]
head_dim = header[7]
head_size = head_dim if head_dim > 0 else dim // n_heads
q_dim = head_size * n_heads
kv_dim = head_size * n_kv_heads

weight_data_bytes = data[32:]
max_floats = len(weight_data_bytes) // 4

print(f"\ndim={dim}, hidden_dim={hidden_dim}, layer_num={layer_num}")
print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}")
print(f"seq_len={seq_len}, head_dim={head_dim}, head_size={head_size}")
print(f"q_dim={q_dim}, kv_dim={kv_dim}")
print(f"shared_weights={shared}")
print(f"max_floats={max_floats}")

# Expected total size
expected_size = (
    dim * vocab_size                    # embedding
    + layer_num * dim                   # attn rmsnorms (NOT seq_len*dim — just dim each!)
    + layer_num * q_dim * dim           # Wq weights
    + layer_num * q_dim                 # Wq biases (since Qwen2.5 has them)
    + layer_num * kv_dim * dim          # Wk weights
    + layer_num * kv_dim                # Wk biases
    + layer_num * kv_dim * dim          # Wv weights
    + layer_num * kv_dim                # Wv biases
    + layer_num * dim * q_dim           # Wo weights
    + layer_num * dim                   # ffn rmsnorms
    + layer_num * dim * hidden_dim      # W1 (gate)
    + layer_num * dim * hidden_dim      # W2 (down)
    + layer_num * dim * hidden_dim      # W3 (up)
    + dim                               # final rmsnorm
    + 2 * seq_len * head_size           # freqs (cos + sin)
)
# No CLS (shared), no Q/K norms (Qwen2.5)
print(f"\nExpected total floats: {expected_size}")
print(f"Actual total floats:   {max_floats}")
print(f"Difference: {expected_size - max_floats}")

# Now trace through the file byte-by-byte
pos = 0
print("\n=== Position trace ===")

pieces = [
    ("embedding", dim * vocab_size),
    ("attn_norm", layer_num * dim),
    ("Wq", layer_num * q_dim * dim),
    ("Wq_bias", layer_num * q_dim),
    ("Wk", layer_num * kv_dim * dim),
    ("Wk_bias", layer_num * kv_dim),
    ("Wv", layer_num * kv_dim * dim),
    ("Wv_bias", layer_num * kv_dim),
    ("Wo", layer_num * dim * q_dim),
    ("ffn_norm", layer_num * dim),
    ("W1(gate)", layer_num * dim * hidden_dim),
    ("W2(down)", layer_num * dim * hidden_dim),
    ("W3(up)", layer_num * dim * hidden_dim),
    ("final_norm", dim),
    ("freqs_cos", seq_len * head_size),
    ("freqs_sin", seq_len * head_size),
]

for name, size in pieces:
    end = pos + size
    print(f"  {name:15s}: pos={pos:>10,} end={end:>10,} size={size:>10,} valid={end <= max_floats}")
    pos = end

print(f"\n  Final pos: {pos}, max_floats: {max_floats}, leftover: {max_floats - pos}")

# Now check: are biases actually present at the expected position?
# Read Wq bias for layer 0 from the expected position and compare with a known-bad comparison
pos_check = dim * vocab_size + layer_num * dim + layer_num * q_dim * dim
print(f"\n=== Checking Wq_bias at pos={pos_check} ===")
# Last few floats of Wq (layer 23, last few elements) should be:
wq_end = dim * vocab_size + layer_num * dim + layer_num * q_dim * dim - 5
wq_last_vals = struct.unpack('5f', weight_data_bytes[wq_end*4:(wq_end+5)*4])
print(f"  Last 5 floats of WQ: {[f'{v:.6f}' for v in wq_last_vals]}")

# First few floats at bias pos:
bias_first_vals = struct.unpack('10f', weight_data_bytes[pos_check*4:(pos_check+10)*4])
print(f"  First 10 floats at bias pos: {[f'{v:.6f}' for v in bias_first_vals]}")

# Also check: what would be the bias values if we looked at the HF model?
# Load HF model and compare
from transformers import AutoModelForCausalLM
MODEL_PATH = "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map='cpu', local_files_only=True)
hf_sd = model.state_dict()

hf_q_bias_0 = hf_sd['model.layers.0.self_attn.q_proj.bias'].float()
print(f"\n  HF q_bias[0] first 10: {[f'{v:.6f}' for v in hf_q_bias_0[:10].tolist()]}")
print(f"  HF q_bias[0] last 10:  {[f'{v:.6f}' for v in hf_q_bias_0[-10:].tolist()]}")

# Verify WQ layer 0 matches
hf_q_weight_0 = hf_sd['model.layers.0.self_attn.q_proj.weight'].float()
bin_q_flat = struct.unpack(f'{q_dim * dim}f', weight_data_bytes[pos_check - layer_num * q_dim * dim * 4:pos_check*4])
bin_q_tensor = torch.tensor(bin_q_flat, dtype=torch.float32).reshape(q_dim, dim)
max_diff_q = (bin_q_tensor - hf_q_weight_0).abs().max().item()
print(f"\n  WQ[0] .bin vs HF max_diff: {max_diff_q:.10f}")

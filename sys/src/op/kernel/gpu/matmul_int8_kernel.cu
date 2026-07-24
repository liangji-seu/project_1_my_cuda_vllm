#include "matmul_int8_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace kernel {

/**
 * Per-group INT8 权重矩阵乘 kernel — 1 block per output element
 *
 * 计算: output[m, n] = Σ_g scale[n,g] * Σ_{k in group g} input[m,k] * weight[n,k] + bias[n]
 *
 * 数据布局:
 *   input       [M, K] fp32 行优先
 *   weight      [N, K] int8 行优先
 *   scales      [N * num_groups] fp32 per-group, row-major: scales[n*num_groups + g]
 *   output      [M, N] fp32 行优先
 */
template <int THREAD_PER_BLOCK>
__global__ void matmul_int8_group_kernel(
    const float* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K, int num_groups, int group_size) {

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int m = bid / N;
  int n = bid % N;
  if (m >= M || n >= N) return;

  const float* input_row = input + m * K;
  const int8_t* weight_row = weight + n * K;

  // ── 预加载 per-group scales 到 shared memory ──
  // 最多 num_groups 个 scale (max 38 for K=4864, group=128)
  extern __shared__ float s_scales[];
  const float* scales_row = scales + n * num_groups;
  for (int g = tid; g < num_groups; g += blockDim.x) {
    s_scales[g] = scales_row[g];
  }
  __syncthreads();

  // ── 向量化点积 + per-group scale ──
  constexpr int pack_size = 4;
  const int pack_num = K / pack_size;
  const int pack_off = pack_size * pack_num;

  __shared__ float sdata[THREAD_PER_BLOCK];
  sdata[tid] = 0.0f;

  const char4* weight_char4_ptr = reinterpret_cast<const char4*>(weight_row);
  const float4* input_float4_ptr = reinterpret_cast<const float4*>(input_row);

#pragma unroll
  for (int i = tid; i < pack_num; i += blockDim.x) {
    int k = i * pack_size;
    char4 w_pack = weight_char4_ptr[i];
    float4 in_pack = input_float4_ptr[i];

    // 每个元素属于不同 group: scale_index = k / group_size
    float s0 = s_scales[k / group_size];
    float s1 = s_scales[(k + 1) / group_size];
    float s2 = s_scales[(k + 2) / group_size];
    float s3 = s_scales[(k + 3) / group_size];

    sdata[tid] += static_cast<float>(w_pack.x) * in_pack.x * s0
                + static_cast<float>(w_pack.y) * in_pack.y * s1
                + static_cast<float>(w_pack.z) * in_pack.z * s2
                + static_cast<float>(w_pack.w) * in_pack.w * s3;
  }

  // 余数 (不完整 pack)
  for (int k = pack_off + tid; k < K; k += blockDim.x) {
    float s = s_scales[k / group_size];
    sdata[tid] += static_cast<float>(weight_row[k]) * input_row[k] * s;
  }

  __syncthreads();

  // ── BlockReduce + bias ──
  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float sum = BlockReduce(temp).Sum(sdata[tid]);

  if (tid == 0) {
    float val = sum + (bias ? bias[n] : 0.0f);
    output[m * N + n] = val;
  }
}

// ═══ GPU 对外接口 ═══
void matmul_int8_kernel_cuda(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& scales,
                              const float* bias,
                              const tensor::Tensor& output,
                              int32_t group_size,
                              void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!scales.is_empty());
  CHECK(!output.is_empty());
  CHECK(weight.get_data_type() == tensor::DataType_t::int8)
      << "Weight must be INT8 for quantized matmul";

  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));
  int32_t M = 1;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0));

  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), K)
      << "Input last dim must equal weight inner dim";

  const int32_t num_groups = (K + group_size - 1) / group_size;
  CHECK_EQ(static_cast<int32_t>(scales.get_size()), N * num_groups)
      << "Scales must have N * num_groups elements (per-group)";

  constexpr int THREADS = 256;
  int grid = M * N;
  size_t shared_mem = num_groups * sizeof(float);  // scale cache

  if (stream) {
    matmul_int8_group_kernel<THREADS>
        <<<grid, THREADS, shared_mem, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const int8_t*>(weight.get_ptr()),
            static_cast<const float*>(scales.get_ptr()),
            bias,
            const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            M, N, K, num_groups, group_size);
  } else {
    matmul_int8_group_kernel<THREADS>
        <<<grid, THREADS, shared_mem>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const int8_t*>(weight.get_ptr()),
            static_cast<const float*>(scales.get_ptr()),
            bias,
            const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            M, N, K, num_groups, group_size);
  }
}

// ═══ CPU 回退 ═══
void matmul_int8_kernel_cpu(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& scales,
                             const float* bias,
                             const tensor::Tensor& output,
                             int32_t group_size,
                             void* stream) {
  (void)stream;

  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));
  int32_t M = 1;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0));

  const int32_t num_groups = (K + group_size - 1) / group_size;
  const float* in = static_cast<const float*>(input.get_ptr());
  const int8_t* w = static_cast<const int8_t*>(weight.get_ptr());
  const float* s = static_cast<const float*>(scales.get_ptr());
  float* out = const_cast<float*>(static_cast<const float*>(output.get_ptr()));

  for (int32_t m = 0; m < M; ++m) {
    for (int32_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int32_t k = 0; k < K; ++k) {
        int32_t g = k / group_size;
        sum += static_cast<float>(w[n * K + k]) * in[m * K + k] * s[n * num_groups + g];
      }
      out[m * N + n] = sum + (bias ? bias[n] : 0.0f);
    }
  }
}

}  // namespace kernel

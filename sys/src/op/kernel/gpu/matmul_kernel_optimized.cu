#include "matmul_kernel_optimized.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// ── 超参数 ──────────────────────────────────────────────
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int NUM_THREADS = (BN / TN) * (BM / TM);  // 256
constexpr int WARP_SIZE = 32;

#define OFFSET(r, c, ld) ((r) * (ld) + (c))

// ── float4 辅助函数 (兼容 const) ──────────────────────────
__device__ __forceinline__ float4 ldg4(const float* p) {
  return make_float4(p[0], p[1], p[2], p[3]);
}
__device__ __forceinline__ void stg4(float* p, float4 v) {
  reinterpret_cast<float4*>(p)[0] = v;
}

/**
 * 优化的矩阵乘 kernel — 双缓冲共享内存 + 寄存器预取 + float4 向量化
 *
 * 计算: output[M,N] = scale * input[M,K] × weight[N,K]^T + bias[N]
 * weight 为 [N,K] 行优先，按转置 B[k][n]=weight[n][k] 访问
 * weight[n][k..k+3] 连续 → float4 读取; 标量写入 Bs
 */
template <int M_BLOCK, int N_BLOCK, int K_BLOCK, int T_M, int T_N>
__global__ void matmul_kernel_tiled(int M, int N, int K,
                                     const float* __restrict__ A,
                                     const float* __restrict__ W,
                                     const float* __restrict__ bias,
                                     float* __restrict__ C, float scale) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = (threadIdx.x % (N_BLOCK / T_N)) * T_N;
  int ty = (threadIdx.x / (N_BLOCK / T_N)) * T_M;

  __shared__ float As[2][K_BLOCK * M_BLOCK];
  __shared__ float Bs[2][K_BLOCK * N_BLOCK];

  // A 加载索引
  const int a_row = threadIdx.x / (K_BLOCK / 4);
  const int a_col = threadIdx.x % (K_BLOCK / 4) * 4;
  const int a_ldg_num = K_BLOCK * M_BLOCK / NUM_THREADS / 4;
  const int a_stride = (a_ldg_num > 0) ? (M_BLOCK / a_ldg_num) : M_BLOCK;

  // B 加载索引 (weight[N,K] 按转置读)
  // BK=8, BN=128: 每个thread加载1个float4(4 floats沿K连续)
  // 4 floats → Bs的4行, 需要 BK/4=2 行threads × BN/4=32 列threads = 64 threads
  // 其余threads通过取模复用
  const int b_col = (threadIdx.x % (N_BLOCK / 4)) * 4;
  const int b_row_base = ((threadIdx.x / (N_BLOCK / 4)) % (K_BLOCK / 4)) * 4;

  float accum[T_M][T_N] = {{0.0f}};
  float a_frag[2][T_M] = {{0.0f}};
  float b_frag[2][T_N] = {{0.0f}};
  float a_reg[4 * (a_ldg_num > 0 ? a_ldg_num : 1)] = {0.0f};
  // B: 每个 thread 固定加载 1 个 float4 (4 floats along K), 无需循环
  float b_reg[4] = {0.0f};

  A = &A[OFFSET(by * M_BLOCK, 0, K)];
  W = &W[bx * N_BLOCK];
  C = &C[OFFSET(by * M_BLOCK, bx * N_BLOCK, N)];

  // ═══ 第一个 tile: 全局内存 → 共享内存 ═══

  #pragma unroll
  for (int i = 0; i < M_BLOCK; i += a_stride) {
    int idx = i / a_stride * 4;
    stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, a_col, K)));
    As[0][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
    As[0][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
    As[0][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
    As[0][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
  }

  // B: 加载 4 个连续 K方向值 → Bs 的 4 行同一列
  {
    float4 val = ldg4(W + OFFSET(b_col, b_row_base, K));
    Bs[0][OFFSET(b_row_base + 0, b_col, N_BLOCK)] = val.x;
    Bs[0][OFFSET(b_row_base + 1, b_col, N_BLOCK)] = val.y;
    Bs[0][OFFSET(b_row_base + 2, b_col, N_BLOCK)] = val.z;
    Bs[0][OFFSET(b_row_base + 3, b_col, N_BLOCK)] = val.w;
  }

  __syncthreads();

  // ── 预取 l=0 ──
  #pragma unroll
  for (int m = 0; m < T_M; m += 4)
    stg4(a_frag[0] + m, ldg4(As[0] + OFFSET(0, ty + m, M_BLOCK)));
  #pragma unroll
  for (int n = 0; n < T_N; n += 4)
    stg4(b_frag[0] + n, ldg4(Bs[0] + OFFSET(0, tx + n, N_BLOCK)));

  // ═══ 主循环: K/BK 步 ═══
  int smem_write = 1;

  for (int k = K_BLOCK; k < K + K_BLOCK; k += K_BLOCK) {
    int smem_read = smem_write ^ 1;

    // 预取下一个 tile
    if (k < K) {
      #pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, k + a_col, K)));
      }
      // B: 预取下一个 tile
      stg4(b_reg, ldg4(W + OFFSET(b_col, k + b_row_base, K)));
    }

    // 计算 l = 0..BK-2
    #pragma unroll
    for (int l = 0; l < K_BLOCK - 1; l++) {
      #pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[(l + 1) & 1] + m, ldg4(As[smem_read] + OFFSET(l + 1, ty + m, M_BLOCK)));
      #pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[(l + 1) & 1] + n, ldg4(Bs[smem_read] + OFFSET(l + 1, tx + n, N_BLOCK)));

      #pragma unroll
      for (int m = 0; m < T_M; m++)
        #pragma unroll
        for (int n = 0; n < T_N; n++)
          accum[m][n] += a_frag[l & 1][m] * b_frag[l & 1][n];
    }

    // l = BK-1 的计算
    #pragma unroll
    for (int m = 0; m < T_M; m++)
      #pragma unroll
      for (int n = 0; n < T_N; n++)
        accum[m][n] += a_frag[(K_BLOCK - 1) & 1][m] * b_frag[(K_BLOCK - 1) & 1][n];

    // 写回共享内存
    if (k < K) {
      #pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        As[smem_write][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
        As[smem_write][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
        As[smem_write][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
        As[smem_write][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
      }

      // B: 预取寄存器 → 共享内存
      {
        float4 val = ldg4(b_reg);
        Bs[smem_write][OFFSET(b_row_base + 0, b_col, N_BLOCK)] = val.x;
        Bs[smem_write][OFFSET(b_row_base + 1, b_col, N_BLOCK)] = val.y;
        Bs[smem_write][OFFSET(b_row_base + 2, b_col, N_BLOCK)] = val.z;
        Bs[smem_write][OFFSET(b_row_base + 3, b_col, N_BLOCK)] = val.w;
      }

      __syncthreads();

      // 预取新 tile l=0
      #pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[0] + m, ldg4(As[smem_write] + OFFSET(0, ty + m, M_BLOCK)));
      #pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[0] + n, ldg4(Bs[smem_write] + OFFSET(0, tx + n, N_BLOCK)));

      smem_write ^= 1;
    }
  }

  // ═══ 写回全局内存 ═══
  #pragma unroll
  for (int m = 0; m < T_M; m++) {
    int row = ty + m;
    bool row_valid = (by * M_BLOCK + row < M);
    if (!row_valid) continue;
    #pragma unroll
    for (int n = 0; n < T_N; n += 4) {
      int col = tx + n;
      float4 ct = ldg4(C + OFFSET(row, col, N));
      ct.x = scale * accum[m][n + 0] + (bias ? bias[bx * N_BLOCK + col + 0] : 0.0f);
      ct.y = scale * accum[m][n + 1] + (bias ? bias[bx * N_BLOCK + col + 1] : 0.0f);
      ct.z = scale * accum[m][n + 2] + (bias ? bias[bx * N_BLOCK + col + 2] : 0.0f);
      ct.w = scale * accum[m][n + 3] + (bias ? bias[bx * N_BLOCK + col + 3] : 0.0f);
      stg4(C + OFFSET(row, col, N), ct);
    }
  }
}

// ═══ 对外接口 ═══
void matmul_kernel_cuda_optimized(const tensor::Tensor& input,
                                   const tensor::Tensor& weight,
                                   const float* bias, float scale,
                                   const tensor::Tensor& output,
                                   void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));

  int32_t M = 1;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0));

  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), K)
      << "Input last dim must equal weight inner dim";

  dim3 block(NUM_THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  matmul_kernel_tiled<BM, BN, BK, TM, TN>
      <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          M, N, K,
          static_cast<const float*>(input.get_ptr()),
          static_cast<const float*>(weight.get_ptr()),
          bias,
          const_cast<float*>(static_cast<const float*>(output.get_ptr())),
          scale);
}

}  // namespace kernel

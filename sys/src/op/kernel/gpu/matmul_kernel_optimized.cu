#include "matmul_kernel_optimized.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// ── 超参数 ──────────────────────────────────────────────
// BM, BN: 输出 C_sub 的 M/N 维度分块
// BK:     K 维度每次加载的切片大小
// TM, TN: 每个线程负责的输出元素数
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int NUM_THREADS = (BN / TN) * (BM / TM);  // 256

#define OFFSET(r, c, ld) ((r) * (ld) + (c))

// ── float4 辅助函数 (兼容 const) ──────────────────────────
__device__ __forceinline__ float4 ldg4(const float* p) {
  return make_float4(p[0], p[1], p[2], p[3]);
}
__device__ __forceinline__ void stg4(float* p, float4 v) {
  reinterpret_cast<float4*>(p)[0] = v;
}

/**
 * 优化的矩阵乘 kernel — 双缓冲共享内存 + 寄存器预取 + float4 向量化 + do-while 流水线
 *
 * 数据布局:
 *   input  [M, K] 行优先 → A[m][k] = A[m * K + k]
 *   weight [N, K] 行优先 → W[n][k] = W[n * K + k]
 *
 * 计算: output[M, N] = scale * input[M, K] @ weight[N, K]^T + bias[N]
 *       C[m, n] = scale * Σ_k A[m, k] * W[n, k] + bias[n]
 *
 * 共享内存布局:
 *   As[BK][BM] — A 的转置: As[k][row] 连续访问 row 维度
 *   Bs[BK][BN] — W 按 K 维度展开: Bs[k][col] 连续访问 col 维度
 */
template <int M_BLOCK, int N_BLOCK, int K_BLOCK, int T_M, int T_N>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_kernel_tiled(int M, int N, int K,
                    const float* __restrict__ A,
                    const float* __restrict__ W,
                    const float* __restrict__ bias,
                    float* __restrict__ C, float scale) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // ── 线程到输出元素的映射 ──
  // tx, ty: 该线程负责的 C_tile 在 C_sub 内的左上角坐标
  int tx = (threadIdx.x % (N_BLOCK / T_N)) * T_N;  // 0, 8, 16, ..., 120
  int ty = (threadIdx.x / (N_BLOCK / T_N)) * T_M;  // 0, 8, 16, ..., 120

  // ── 双缓冲共享内存 ──
  __shared__ float As[2][K_BLOCK * M_BLOCK];  // [write_idx][BK * BM]
  __shared__ float Bs[2][K_BLOCK * N_BLOCK];  // [write_idx][BK * BN]

  // ── A 加载参数 (转置: 行优先 A → As[k][row]) ──
  // 256 threads, BK=8: 每个 thread 读 1 个 float4 (覆盖4个K), 占2个col-group
  // a_row: 0..127 (覆盖 BM=128), a_col: 0 或 4 (覆盖 BK=8)
  const int a_row = threadIdx.x / (K_BLOCK / 4);
  const int a_col = threadIdx.x % (K_BLOCK / 4) * 4;
  const int a_ldg_num = K_BLOCK * M_BLOCK / NUM_THREADS / 4;  // 1
  const int a_stride = (a_ldg_num > 0) ? (M_BLOCK / a_ldg_num) : M_BLOCK;  // 128

  // ── W 加载参数 (沿 K 读: W[n][k..k+3] → Bs[k..k+3][n]) ──
  // 256 threads, BK=8, BN=128: 每 thread 处理 1 个 N-col + 4 个 K-row
  // b_n_col: 0..127, b_k_base: 0 或 4
  const int b_n_col = threadIdx.x % N_BLOCK;
  const int b_k_base = (threadIdx.x / N_BLOCK) * 4;
  const int b_k_stride = (NUM_THREADS / N_BLOCK) * 4;  // stride for K rows
  static_assert(NUM_THREADS / N_BLOCK > 0, "must have enough threads per N col");

  // ── 累加器 + 寄存器缓冲区 ──
  float accum[T_M][T_N] = {{0.0f}};
  float a_frag[2][T_M] = {{0.0f}};
  float b_frag[2][T_N] = {{0.0f}};
  float a_reg[4 * (a_ldg_num > 0 ? a_ldg_num : 1)] = {0.0f};

  // ── 全局指针重定向 ──
  // A: 指向第 by 个 M-block 的第 0 列
  A = &A[OFFSET(by * M_BLOCK, 0, K)];
  // W: 指向第 bx 个 N-block 的第 0 列 (行优先 [N,K], 跳 bx*BN 整行)
  W = &W[bx * N_BLOCK * K];
  // C: 指向第 (by, bx) 个输出子块
  C = &C[OFFSET(by * M_BLOCK, bx * N_BLOCK, N)];

  // ── 边界守卫 ──
  const bool valid_a_row = (by * M_BLOCK + a_row < M);
  const bool valid_b_col = (bx * N_BLOCK + b_n_col < N);

  // ════════════════════════════════════════════════════════════
  // 第一道: 全局内存 → 寄存器 → 共享内存 (tile 0, smem buffer 0)
  // ════════════════════════════════════════════════════════════

  // A: float4 读取 (行方向K连续) → 转置写入 As[col][row]
#pragma unroll
  for (int i = 0; i < M_BLOCK; i += a_stride) {
    int idx = i / a_stride * 4;
    if (valid_a_row && (a_row + i < M)) {
      // 向量化读全局内存 A[(by*BM + a_row + i)][a_col..a_col+3]
      stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, a_col, K)));
    } else {
      a_reg[idx + 0] = a_reg[idx + 1] = a_reg[idx + 2] = a_reg[idx + 3] = 0.0f;
    }
    // 转置写入 As: 4个连续K → As[a_col..a_col+3][i + a_row]
    As[0][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
    As[0][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
    As[0][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
    As[0][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
  }

  // W: 沿 K 维度 float4 读取 → 散布写入 Bs[k..k+3][n]
  // stride loop: 每个 thread 覆盖其 b_n_col 对应的所有 K-row 块
#pragma unroll 1
  for (int k_off = b_k_base; k_off < K_BLOCK; k_off += b_k_stride) {
    float4 val;
    if (valid_b_col) {
      val = ldg4(W + OFFSET(b_n_col, k_off, K));
    } else {
      val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    Bs[0][OFFSET(k_off + 0, b_n_col, N_BLOCK)] = val.x;
    Bs[0][OFFSET(k_off + 1, b_n_col, N_BLOCK)] = val.y;
    Bs[0][OFFSET(k_off + 2, b_n_col, N_BLOCK)] = val.z;
    Bs[0][OFFSET(k_off + 3, b_n_col, N_BLOCK)] = val.w;
  }

  __syncthreads();

  // ── 共享内存 → 寄存器: 预取 tile 0 的 l=0 行/列 ──
#pragma unroll
  for (int m = 0; m < T_M; m += 4)
    stg4(a_frag[0] + m, ldg4(As[0] + OFFSET(0, ty + m, M_BLOCK)));
#pragma unroll
  for (int n = 0; n < T_N; n += 4)
    stg4(b_frag[0] + n, ldg4(Bs[0] + OFFSET(0, tx + n, N_BLOCK)));

  // ════════════════════════════════════════════════════════════
  // 主循环: 双缓冲共享内存 + 寄存器预取 + 指令重叠 (do-while)
  // ════════════════════════════════════════════════════════════
  int write_idx = 1;
  int k = 0;

  do {
    k += K_BLOCK;
    int read_idx = write_idx ^ 1;

    // ── 全局内存 → 寄存器: 预取下一个 tile (k < K 时) ──
    if (k < K) {
      // A: 预取 A[(by*BM + a_row + i)][k + a_col..k + a_col + 3]
#pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        if (valid_a_row && (a_row + i < M)) {
          stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, k + a_col, K)));
        } else {
          a_reg[idx + 0] = a_reg[idx + 1] = a_reg[idx + 2] = a_reg[idx + 3] = 0.0f;
        }
      }
    }

    // ★★★ 指令重叠发生在此处 ★★★
    // 全局内存 → 寄存器 (A/W 预取) 与 共享内存 → 寄存器 (计算) 可并行

    // ── 计算: l = 0 .. BK-2 (预取 l+1, 计算 l) ──
#pragma unroll
    for (int l = 0; l < K_BLOCK - 1; l++) {
      // 预取 l+1: 共享内存 → 寄存器
#pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[(l + 1) & 1] + m,
             ldg4(As[read_idx] + OFFSET(l + 1, ty + m, M_BLOCK)));
#pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[(l + 1) & 1] + n,
             ldg4(Bs[read_idx] + OFFSET(l + 1, tx + n, N_BLOCK)));

      // 计算 l: 寄存器 FMA
#pragma unroll
      for (int m = 0; m < T_M; m++)
#pragma unroll
        for (int n = 0; n < T_N; n++)
          accum[m][n] += a_frag[l & 1][m] * b_frag[l & 1][n];
    }

    // ── 计算: l = BK-1 (最后一行, 无需预取) ──
#pragma unroll
    for (int m = 0; m < T_M; m++)
#pragma unroll
      for (int n = 0; n < T_N; n++)
        accum[m][n] += a_frag[(K_BLOCK - 1) & 1][m] * b_frag[(K_BLOCK - 1) & 1][n];

    // ── 寄存器 → 共享内存 + 预取新 tile l=0 ──
    if (k < K) {
      // A: 寄存器 → As[write_idx]
#pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        As[write_idx][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
        As[write_idx][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
        As[write_idx][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
        As[write_idx][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
      }

      // W: 预取 (已在寄存器中) → Bs[write_idx]
#pragma unroll 1
      for (int k_off = b_k_base; k_off < K_BLOCK; k_off += b_k_stride) {
        if (valid_b_col) {
          // 重新从全局内存读取 (当前 k-offset)
          float4 val = ldg4(W + OFFSET(b_n_col, k + k_off, K));
          Bs[write_idx][OFFSET(k_off + 0, b_n_col, N_BLOCK)] = val.x;
          Bs[write_idx][OFFSET(k_off + 1, b_n_col, N_BLOCK)] = val.y;
          Bs[write_idx][OFFSET(k_off + 2, b_n_col, N_BLOCK)] = val.z;
          Bs[write_idx][OFFSET(k_off + 3, b_n_col, N_BLOCK)] = val.w;
        } else {
          Bs[write_idx][OFFSET(k_off + 0, b_n_col, N_BLOCK)] = 0.0f;
          Bs[write_idx][OFFSET(k_off + 1, b_n_col, N_BLOCK)] = 0.0f;
          Bs[write_idx][OFFSET(k_off + 2, b_n_col, N_BLOCK)] = 0.0f;
          Bs[write_idx][OFFSET(k_off + 3, b_n_col, N_BLOCK)] = 0.0f;
        }
      }

      __syncthreads();

      // 预取新 tile l=0: 共享内存 → 寄存器缓冲区
#pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[0] + m, ldg4(As[write_idx] + OFFSET(0, ty + m, M_BLOCK)));
#pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[0] + n, ldg4(Bs[write_idx] + OFFSET(0, tx + n, N_BLOCK)));

      write_idx ^= 1;
    }
  } while (k < K);

  // ════════════════════════════════════════════════════════════
  // 写回全局内存: 寄存器 accum → C (带 scale + bias, float4)
  // ════════════════════════════════════════════════════════════
#pragma unroll
  for (int m = 0; m < T_M; m++) {
    int row = ty + m;
    if (by * M_BLOCK + row >= M) continue;
#pragma unroll
    for (int n = 0; n < T_N; n += 4) {
      int col = tx + n;
      // N 维度边界保护 (行优先, col 可能部分越界)
      if (bx * N_BLOCK + col >= N) {
        // 逐标量处理越界列
        for (int nn = 0; nn < 4 && (bx * N_BLOCK + col + nn < N); nn++) {
          float val = scale * accum[m][n + nn];
          if (bias) val += bias[bx * N_BLOCK + col + nn];
          C[OFFSET(row, col + nn, N)] = val;
        }
        continue;
      }
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

  // weight 形状: [N, K]
  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));

  // input 形状: [M=1, K] 或 [M, K]
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

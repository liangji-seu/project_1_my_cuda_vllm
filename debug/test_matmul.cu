// Simple numerical test for optimized matmul kernel
// Tests: C[M,N] = A[M,K] @ W[N,K]^T + bias[N]
// Both A and W are row-major
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define OFFSET(r, c, ld) ((r) * (ld) + (c))
#define CHECK(err) do { if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int NUM_THREADS = (BN / TN) * (BM / TM);

__device__ __forceinline__ float4 ldg4(const float* p) {
  return make_float4(p[0], p[1], p[2], p[3]);
}
__device__ __forceinline__ void stg4(float* p, float4 v) {
  reinterpret_cast<float4*>(p)[0] = v;
}

template <int M_BLOCK, int N_BLOCK, int K_BLOCK, int T_M, int T_N>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_kernel_tiled(int M, int N, int K,
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

  // A loading params
  const int a_row = threadIdx.x / (K_BLOCK / 4);
  const int a_col = threadIdx.x % (K_BLOCK / 4) * 4;
  const int a_ldg_num = K_BLOCK * M_BLOCK / NUM_THREADS / 4;
  const int a_stride = (a_ldg_num > 0) ? (M_BLOCK / a_ldg_num) : M_BLOCK;

  // W loading params (along K dimension: W[n][k..k+3] → Bs[k..k+3][n])
  const int b_n_col = threadIdx.x % N_BLOCK;
  const int b_k_base = (threadIdx.x / N_BLOCK) * 4;
  const int b_k_stride = (NUM_THREADS / N_BLOCK) * 4;

  float accum[T_M][T_N] = {{0.0f}};
  float a_frag[2][T_M] = {{0.0f}};
  float b_frag[2][T_N] = {{0.0f}};
  float a_reg[4 * (a_ldg_num > 0 ? a_ldg_num : 1)] = {0.0f};

  const bool valid_a_row = (by * M_BLOCK + a_row < M);
  const bool valid_b_col = (bx * N_BLOCK + b_n_col < N);

  A = &A[OFFSET(by * M_BLOCK, 0, K)];
  W = &W[bx * N_BLOCK * K];
  C = &C[OFFSET(by * M_BLOCK, bx * N_BLOCK, N)];

  // === Tile 0: global → shared memory ===

#pragma unroll
  for (int i = 0; i < M_BLOCK; i += a_stride) {
    int idx = i / a_stride * 4;
    if (valid_a_row && (a_row + i < M)) {
      stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, a_col, K)));
    } else {
      a_reg[idx + 0] = a_reg[idx + 1] = a_reg[idx + 2] = a_reg[idx + 3] = 0.0f;
    }
    As[0][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
    As[0][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
    As[0][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
    As[0][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
  }

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

#pragma unroll
  for (int m = 0; m < T_M; m += 4)
    stg4(a_frag[0] + m, ldg4(As[0] + OFFSET(0, ty + m, M_BLOCK)));
#pragma unroll
  for (int n = 0; n < T_N; n += 4)
    stg4(b_frag[0] + n, ldg4(Bs[0] + OFFSET(0, tx + n, N_BLOCK)));

  // === Main loop: double-buffered ===
  int write_idx = 1;
  int k = 0;

  do {
    k += K_BLOCK;
    int read_idx = write_idx ^ 1;

    if (k < K) {
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

#pragma unroll
    for (int l = 0; l < K_BLOCK - 1; l++) {
#pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[(l + 1) & 1] + m,
             ldg4(As[read_idx] + OFFSET(l + 1, ty + m, M_BLOCK)));
#pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[(l + 1) & 1] + n,
             ldg4(Bs[read_idx] + OFFSET(l + 1, tx + n, N_BLOCK)));

#pragma unroll
      for (int m = 0; m < T_M; m++)
#pragma unroll
        for (int n = 0; n < T_N; n++)
          accum[m][n] += a_frag[l & 1][m] * b_frag[l & 1][n];
    }

#pragma unroll
    for (int m = 0; m < T_M; m++)
#pragma unroll
      for (int n = 0; n < T_N; n++)
        accum[m][n] += a_frag[(K_BLOCK - 1) & 1][m] * b_frag[(K_BLOCK - 1) & 1][n];

    if (k < K) {
#pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        As[write_idx][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
        As[write_idx][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
        As[write_idx][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
        As[write_idx][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
      }

#pragma unroll 1
      for (int k_off = b_k_base; k_off < K_BLOCK; k_off += b_k_stride) {
        if (valid_b_col) {
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

#pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[0] + m, ldg4(As[write_idx] + OFFSET(0, ty + m, M_BLOCK)));
#pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[0] + n, ldg4(Bs[write_idx] + OFFSET(0, tx + n, N_BLOCK)));

      write_idx ^= 1;
    }
  } while (k < K);

  // === Writeback: accum → C ===
#pragma unroll
  for (int m = 0; m < T_M; m++) {
    int row = ty + m;
    if (by * M_BLOCK + row >= M) continue;
#pragma unroll
    for (int n = 0; n < T_N; n += 4) {
      int col = tx + n;
      if (bx * N_BLOCK + col >= N) {
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

// CPU reference
void cpu_gemm(int M, int N, int K, const float* A, const float* W,
              const float* bias, float scale, float* C) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;
      for (int k = 0; k < K; k++)
        sum += A[m * K + k] * W[n * K + k];
      C[m * N + n] = scale * sum + (bias ? bias[n] : 0);
    }
  }
}

int main() {
  int total_tests = 0, passed = 0;

  // Test various sizes
  struct TestCase { int M; int N; int K; };
  TestCase cases[] = {
    {1,   128,  8},     // single tile, tiny K
    {1,   128,  128},   // K == BN (old bug would pass)
    {1,   128,  896},   // K != BN (old bug exposed!)
    {1,   256,  128},
    {1,   896,  896},   // typical attention matmul
    {4,   128,  896},   // multi-row input
    {1,   4864, 896},   // typical FFN matmul (large N)
    {1,   151936, 896}, // classifier (very large N)
  };

  for (auto tc : cases) {
    int M = tc.M, N = tc.N, K = tc.K;
    printf("Test M=%d N=%d K=%d ... ", M, N, K);

    size_t a_size = M * K * sizeof(float);
    size_t w_size = N * K * sizeof(float);
    size_t c_size = M * N * sizeof(float);
    size_t bias_size = N * sizeof(float);

    float *A = (float*)malloc(a_size);
    float *W = (float*)malloc(w_size);
    float *bias_h = (float*)malloc(bias_size);
    float *C_cpu = (float*)malloc(c_size);
    float *C_gpu = (float*)malloc(c_size);

    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < N * K; i++) W[i] = (float)((i % 5) + 1);
    for (int i = 0; i < N; i++) bias_h[i] = (float)(i % 3);

    float *dA, *dW, *dB, *dC;
    CHECK(cudaMalloc(&dA, a_size));
    CHECK(cudaMalloc(&dW, w_size));
    CHECK(cudaMalloc(&dB, bias_size));
    CHECK(cudaMalloc(&dC, c_size));
    CHECK(cudaMemcpy(dA, A, a_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dW, W, w_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, bias_h, bias_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, c_size));

    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    matmul_kernel_tiled<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, dA, dW, dB, dC, 1.0f);

    CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("KERNEL ERROR: %s\n", cudaGetErrorString(err));
      total_tests++;
      goto cleanup;
    }
    CHECK(cudaMemcpy(C_gpu, dC, c_size, cudaMemcpyDeviceToHost));

    cpu_gemm(M, N, K, A, W, bias_h, 1.0f, C_cpu);

    {
      int errors = 0;
      float max_err = 0.0f;
      for (int i = 0; i < M * N; i++) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
          if (errors < 5)
            printf("  [%d] cpu=%.3f gpu=%.3f err=%.3f\n", i, C_cpu[i], C_gpu[i], err);
          errors++;
        }
      }
      total_tests++;
      if (errors == 0) { printf("PASS (max_err=%.6f)\n", max_err); passed++; }
      else printf("FAIL (%d errors, max_err=%.6f)\n", errors, max_err);
    }

cleanup:
    free(A); free(W); free(bias_h); free(C_cpu); free(C_gpu);
    cudaFree(dA); cudaFree(dW); cudaFree(dB); cudaFree(dC);
  }

  printf("\n=== %d/%d tests passed ===\n", passed, total_tests);
  return (passed == total_tests) ? 0 : 1;
}

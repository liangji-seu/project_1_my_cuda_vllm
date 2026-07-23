// Simple numerical test for optimized matmul kernel
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

  const int a_row = threadIdx.x / (K_BLOCK / 4);
  const int a_col = threadIdx.x % (K_BLOCK / 4) * 4;
  const int a_ldg_num = K_BLOCK * M_BLOCK / NUM_THREADS / 4;
  const int a_stride = (a_ldg_num > 0) ? (M_BLOCK / a_ldg_num) : M_BLOCK;

  const int b_col = (threadIdx.x % (N_BLOCK / 4)) * 4;
  const int b_row_base = ((threadIdx.x / (N_BLOCK / 4)) % (K_BLOCK / 4)) * 4;

  float accum[T_M][T_N] = {{0.0f}};
  float a_frag[2][T_M] = {{0.0f}};
  float b_frag[2][T_N] = {{0.0f}};
  float a_reg[4 * (a_ldg_num > 0 ? a_ldg_num : 1)] = {0.0f};
  float b_reg[4] = {0.0f};

  // Guard out-of-bounds for M < BM
  bool valid_a_row = (by * M_BLOCK + a_row < M);

  A = &A[OFFSET(by * M_BLOCK, 0, K)];
  W = &W[bx * N_BLOCK];
  C = &C[OFFSET(by * M_BLOCK, bx * N_BLOCK, N)];

  #pragma unroll
  for (int i = 0; i < M_BLOCK; i += a_stride) {
    int idx = i / a_stride * 4;
    if (valid_a_row && a_row + i < M) {
      stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, a_col, K)));
    } else {
      a_reg[idx+0] = a_reg[idx+1] = a_reg[idx+2] = a_reg[idx+3] = 0.0f;
    }
    As[0][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
    As[0][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
    As[0][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
    As[0][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
  }

  {
    float4 val = ldg4(W + OFFSET(b_col, b_row_base, K));
    Bs[0][OFFSET(b_row_base + 0, b_col, N_BLOCK)] = val.x;
    Bs[0][OFFSET(b_row_base + 1, b_col, N_BLOCK)] = val.y;
    Bs[0][OFFSET(b_row_base + 2, b_col, N_BLOCK)] = val.z;
    Bs[0][OFFSET(b_row_base + 3, b_col, N_BLOCK)] = val.w;
  }

  __syncthreads();

  #pragma unroll
  for (int m = 0; m < T_M; m += 4)
    stg4(a_frag[0] + m, ldg4(As[0] + OFFSET(0, ty + m, M_BLOCK)));
  #pragma unroll
  for (int n = 0; n < T_N; n += 4)
    stg4(b_frag[0] + n, ldg4(Bs[0] + OFFSET(0, tx + n, N_BLOCK)));

  int smem_write = 1;

  for (int k = K_BLOCK; k < K + K_BLOCK; k += K_BLOCK) {
    int smem_read = smem_write ^ 1;

    if (k < K) {
      #pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        if (valid_a_row && a_row + i < M) {
          stg4(a_reg + idx, ldg4(A + OFFSET(a_row + i, k + a_col, K)));
        } else {
          a_reg[idx+0] = a_reg[idx+1] = a_reg[idx+2] = a_reg[idx+3] = 0.0f;
        }
      }
      stg4(b_reg, ldg4(W + OFFSET(b_col, k + b_row_base, K)));
    }

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

    #pragma unroll
    for (int m = 0; m < T_M; m++)
      #pragma unroll
      for (int n = 0; n < T_N; n++)
        accum[m][n] += a_frag[(K_BLOCK - 1) & 1][m] * b_frag[(K_BLOCK - 1) & 1][n];

    if (k < K) {
      #pragma unroll
      for (int i = 0; i < M_BLOCK; i += a_stride) {
        int idx = i / a_stride * 4;
        As[smem_write][OFFSET(a_col + 0, i + a_row, M_BLOCK)] = a_reg[idx + 0];
        As[smem_write][OFFSET(a_col + 1, i + a_row, M_BLOCK)] = a_reg[idx + 1];
        As[smem_write][OFFSET(a_col + 2, i + a_row, M_BLOCK)] = a_reg[idx + 2];
        As[smem_write][OFFSET(a_col + 3, i + a_row, M_BLOCK)] = a_reg[idx + 3];
      }

      {
        float4 val = ldg4(b_reg);
        Bs[smem_write][OFFSET(b_row_base + 0, b_col, N_BLOCK)] = val.x;
        Bs[smem_write][OFFSET(b_row_base + 1, b_col, N_BLOCK)] = val.y;
        Bs[smem_write][OFFSET(b_row_base + 2, b_col, N_BLOCK)] = val.z;
        Bs[smem_write][OFFSET(b_row_base + 3, b_col, N_BLOCK)] = val.w;
      }

      __syncthreads();

      #pragma unroll
      for (int m = 0; m < T_M; m += 4)
        stg4(a_frag[0] + m, ldg4(As[smem_write] + OFFSET(0, ty + m, M_BLOCK)));
      #pragma unroll
      for (int n = 0; n < T_N; n += 4)
        stg4(b_frag[0] + n, ldg4(Bs[smem_write] + OFFSET(0, tx + n, N_BLOCK)));

      smem_write ^= 1;
    }
  }

  #pragma unroll
  for (int m = 0; m < T_M; m++) {
    int row = ty + m;
    if (by * M_BLOCK + row >= M) continue;
    #pragma unroll
    for (int nn = 0; nn < T_N; nn += 4) {
      int col = tx + nn;
      float4 ct = ldg4(C + OFFSET(row, col, N));
      ct.x = scale * accum[m][nn + 0] + (bias ? bias[bx * N_BLOCK + col + 0] : 0.0f);
      ct.y = scale * accum[m][nn + 1] + (bias ? bias[bx * N_BLOCK + col + 1] : 0.0f);
      ct.z = scale * accum[m][nn + 2] + (bias ? bias[bx * N_BLOCK + col + 2] : 0.0f);
      ct.w = scale * accum[m][nn + 3] + (bias ? bias[bx * N_BLOCK + col + 3] : 0.0f);
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
  // Test various sizes
  for (int M = 1; M <= 4; M++) {
    for (int N : {128, 256, 896, 1024}) {
      int K = 128;
      printf("Test M=%d N=%d K=%d ... ", M, N, K);

      size_t a_size = M * K * sizeof(float);
      size_t w_size = N * K * sizeof(float);
      size_t c_size = M * N * sizeof(float);
      size_t bias_size = N * sizeof(float);

      float *A = (float*)malloc(a_size);
      float *W = (float*)malloc(w_size);
      float *bias_h = (float*)malloc(bias_size);
      float *C_cpu = (float*)calloc(c_size, 1);
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
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpy(C_gpu, dC, c_size, cudaMemcpyDeviceToHost));

      cpu_gemm(M, N, K, A, W, bias_h, 1.0f, C_cpu);

      int errors = 0;
      for (int i = 0; i < M * N; i++) {
        if (fabsf(C_gpu[i] - C_cpu[i]) > 1e-3f) {
          if (errors < 5)
            printf("  [%d] cpu=%.1f gpu=%.1f\n", i, C_cpu[i], C_gpu[i]);
          errors++;
        }
      }

      if (errors == 0) printf("PASS\n");
      else printf("FAIL (%d errors)\n", errors);

      free(A); free(W); free(bias_h); free(C_cpu); free(C_gpu);
      cudaFree(dA); cudaFree(dW); cudaFree(dB); cudaFree(dC);
    }
  }
  return 0;
}

// Also test with K=8
{
  int M=1, N=128, K=8;
  printf("Test M=%d N=%d K=%d (single tile) ... ", M, N, K);
  float *A=(float*)malloc(M*K*sizeof(float));
  float *W=(float*)malloc(N*K*sizeof(float));
  float *C_cpu=(float*)calloc(M*N,sizeof(float));
  float *C_gpu=(float*)malloc(M*N*sizeof(float));
  for(int i=0;i<M*K;i++)A[i]=(float)(i%7+1);
  for(int i=0;i<N*K;i++)W[i]=(float)((i%5)+1);
  float *dA,*dW,*dC;
  cudaMalloc(&dA,M*K*4);cudaMalloc(&dW,N*K*4);cudaMalloc(&dC,M*N*4);
  cudaMemcpy(dA,A,M*K*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dW,W,N*K*4,cudaMemcpyHostToDevice);
  cudaMemset(dC,0,M*N*4);
  dim3 b(NUM_THREADS), g((N+127)/128,(M+127)/128);
  matmul_kernel_tiled<BM,BN,BK,TM,TN><<<g,b>>>(M,N,K,dA,dW,0,dC,1.0f);
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu,dC,M*N*4,cudaMemcpyDeviceToHost);
  cpu_gemm(M,N,K,A,W,0,1.0f,C_cpu);
  int err=0;
  for(int i=0;i<M*N;i++)if(fabsf(C_gpu[i]-C_cpu[i])>1e-3f){if(err<10)printf("[%d]cpu=%.1f gpu=%.1f ",i,C_cpu[i],C_gpu[i]);err++;}
  if(err==0)printf("PASS\n");else printf("FAIL(%d)\n",err);
  free(A);free(W);free(C_cpu);free(C_gpu);cudaFree(dA);cudaFree(dW);cudaFree(dC);
}

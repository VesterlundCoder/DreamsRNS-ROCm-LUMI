#include <hip/hip_runtime.h>
#include <cstdint>
#include "rns/rns_ops.h"
#include "rns/rns_kernels.h"

namespace rns {

static constexpr int M = 8;
static constexpr int E = M * M;  // 64

/**
 * Specialized 8x8 matrix multiplication kernel.
 */
__global__ void k_gemm_mod_m8(
    uint32_t* __restrict__ C,
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    const Modulus32* __restrict__ mods,
    int K, int Batch)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = K * Batch * E;
  if (tid >= total) return;
  
  int tmp = tid;
  int k = tmp / (Batch * E);
  tmp -= k * (Batch * E);
  int b = tmp / E;
  int e = tmp - b * E;
  int i = e / M;
  int j = e % M;
  
  Modulus32 mod = mods[k];
  int base = k * (Batch * E) + b * E;
  
  // Unrolled dot product for M=8
  uint32_t acc = 0;
  
  #pragma unroll
  for (int t = 0; t < M; ++t) {
    uint32_t a_it = A[base + i * M + t];
    uint32_t b_tj = B[base + t * M + j];
    acc = fma_mod(a_it, b_tj, acc, mod);
  }
  
  C[base + i * M + j] = acc;
}

void rns_gemm_mod_m8(uint32_t* C, const uint32_t* A, const uint32_t* B,
                      const Modulus32* mods, int K, int Batch)
{
  int total = K * Batch * E;
  dim3 block(256);
  dim3 grid((total + 255) / 256);
  
  hipLaunchKernelGGL(k_gemm_mod_m8, grid, block, 0, 0,
                     C, A, B, mods, K, Batch);
}

}  // namespace rns

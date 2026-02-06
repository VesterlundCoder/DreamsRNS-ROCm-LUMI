#include <hip/hip_runtime.h>
#include <cstdint>
#include "rns/rns_ops.h"
#include "rns/rns_kernels.h"

namespace rns {

static constexpr int M = 4;
static constexpr int E = M * M;  // 16

/**
 * Specialized 4x4 matrix multiplication kernel.
 * Fully unrolled inner loop for maximum performance.
 */
__global__ void k_gemm_mod_m4(
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
  
  // Load row i of A
  uint32_t a0 = A[base + i * M + 0];
  uint32_t a1 = A[base + i * M + 1];
  uint32_t a2 = A[base + i * M + 2];
  uint32_t a3 = A[base + i * M + 3];
  
  // Load column j of B
  uint32_t b0 = B[base + 0 * M + j];
  uint32_t b1 = B[base + 1 * M + j];
  uint32_t b2 = B[base + 2 * M + j];
  uint32_t b3 = B[base + 3 * M + j];
  
  // Compute dot product with fma_mod
  uint32_t acc = 0;
  acc = fma_mod(a0, b0, acc, mod);
  acc = fma_mod(a1, b1, acc, mod);
  acc = fma_mod(a2, b2, acc, mod);
  acc = fma_mod(a3, b3, acc, mod);
  
  C[base + i * M + j] = acc;
}

void rns_gemm_mod_m4(uint32_t* C, const uint32_t* A, const uint32_t* B,
                      const Modulus32* mods, int K, int Batch)
{
  int total = K * Batch * E;
  dim3 block(256);
  dim3 grid((total + 255) / 256);
  
  hipLaunchKernelGGL(k_gemm_mod_m4, grid, block, 0, 0,
                     C, A, B, mods, K, Batch);
}

}  // namespace rns

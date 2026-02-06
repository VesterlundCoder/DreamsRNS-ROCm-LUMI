#include <hip/hip_runtime.h>
#include <cstdint>
#include "rns/rns_ops.h"
#include "rns/rns_kernels.h"

namespace rns {

static inline dim3 grid1d(int n, int block = 256) {
  return dim3((n + block - 1) / block);
}

// ============================================================================
// Elementwise Kernels
// ============================================================================

__global__ void k_add_u32(
    uint32_t* out, const uint32_t* a, const uint32_t* b,
    const Modulus32* mods, int K, int N)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = K * N;
  if (tid >= total) return;
  
  int k = tid / N;
  const Modulus32 mod = mods[k];
  out[tid] = add_mod(a[tid], b[tid], mod);
}

__global__ void k_sub_u32(
    uint32_t* out, const uint32_t* a, const uint32_t* b,
    const Modulus32* mods, int K, int N)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = K * N;
  if (tid >= total) return;
  
  int k = tid / N;
  const Modulus32 mod = mods[k];
  out[tid] = sub_mod(a[tid], b[tid], mod);
}

__global__ void k_mul_u32(
    uint32_t* out, const uint32_t* a, const uint32_t* b,
    const Modulus32* mods, int K, int N)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = K * N;
  if (tid >= total) return;
  
  int k = tid / N;
  const Modulus32 mod = mods[k];
  out[tid] = mul_mod(a[tid], b[tid], mod);
}

void rns_add_u32(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const Modulus32* mods, int K, int N)
{
  int total = K * N;
  hipLaunchKernelGGL(k_add_u32, grid1d(total), dim3(256), 0, 0,
                     out, a, b, mods, K, N);
}

void rns_sub_u32(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const Modulus32* mods, int K, int N)
{
  int total = K * N;
  hipLaunchKernelGGL(k_sub_u32, grid1d(total), dim3(256), 0, 0,
                     out, a, b, mods, K, N);
}

void rns_mul_u32(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const Modulus32* mods, int K, int N)
{
  int total = K * N;
  hipLaunchKernelGGL(k_mul_u32, grid1d(total), dim3(256), 0, 0,
                     out, a, b, mods, K, N);
}

// ============================================================================
// Generic GEMM Kernel (fallback for non-specialized sizes)
// ============================================================================

__global__ void k_gemm_mod_generic(
    uint32_t* C, const uint32_t* A, const uint32_t* Bmat,
    const Modulus32* mods, int K, int B, int m)
{
  int E = m * m;
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = K * B * E;
  if (tid >= total) return;
  
  int tmp = tid;
  int k = tmp / (B * E);
  tmp -= k * (B * E);
  int b = tmp / E;
  int e = tmp - b * E;
  int i = e / m;
  int j = e - i * m;
  
  Modulus32 mod = mods[k];
  int base = k * (B * E) + b * E;
  
  uint32_t acc = 0;
  for (int t = 0; t < m; ++t) {
    uint32_t a_it = A[base + i * m + t];
    uint32_t b_tj = Bmat[base + t * m + j];
    acc = fma_mod(a_it, b_tj, acc, mod);
  }
  
  C[base + i * m + j] = acc;
}

// ============================================================================
// GEMM Dispatcher
// ============================================================================

void rns_gemm_mod_u32(
    uint32_t* C, const uint32_t* A, const uint32_t* Bmat,
    const Modulus32* mods, int K, int B, int m)
{
  int E = m * m;
  int total = K * B * E;
  dim3 block(256);
  dim3 grid = grid1d(total);
  
  // Dispatch to specialized kernels
  if (m == 4) {
    rns_gemm_mod_m4(C, A, Bmat, mods, K, B);
  } else if (m == 6) {
    rns_gemm_mod_m6(C, A, Bmat, mods, K, B);
  } else if (m == 8) {
    rns_gemm_mod_m8(C, A, Bmat, mods, K, B);
  } else if (m == 10) {
    rns_gemm_mod_m10(C, A, Bmat, mods, K, B);
  } else {
    // Generic fallback
    hipLaunchKernelGGL(k_gemm_mod_generic, grid, block, 0, 0,
                       C, A, Bmat, mods, K, B, m);
  }
}

}  // namespace rns

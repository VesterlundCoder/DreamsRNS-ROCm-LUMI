// Auto-generated GEMM kernel for m=6
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#include "rns/config.h"
#include "rns/modops.h"

#ifdef RNS_HAS_GPU
#include <hip/hip_runtime.h>
#endif

namespace rns {

#ifdef RNS_HAS_GPU
__global__ void k_gemm_mod_m6(
    u32* __restrict__ C,
    const u32* __restrict__ A,
    const u32* __restrict__ B,
    const u32* __restrict__ primes,
    const u64* __restrict__ mus,
    int K, int batch)
{
    int k = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= K || b >= batch) return;
    
    u32 p = primes[k];
    u64 mu = mus[k];
    constexpr int E = 36;
    int base = k * batch * E + b * E;
    
    // Load A and B into registers
    u32 c0_0 = 0;
    u32 c0_1 = 0;
    u32 c0_2 = 0;
    u32 c0_3 = 0;
    u32 c0_4 = 0;
    u32 c0_5 = 0;
    u32 c1_0 = 0;
    u32 c1_1 = 0;
    u32 c1_2 = 0;
    u32 c1_3 = 0;
    u32 c1_4 = 0;
    u32 c1_5 = 0;
    u32 c2_0 = 0;
    u32 c2_1 = 0;
    u32 c2_2 = 0;
    u32 c2_3 = 0;
    u32 c2_4 = 0;
    u32 c2_5 = 0;
    u32 c3_0 = 0;
    u32 c3_1 = 0;
    u32 c3_2 = 0;
    u32 c3_3 = 0;
    u32 c3_4 = 0;
    u32 c3_5 = 0;
    u32 c4_0 = 0;
    u32 c4_1 = 0;
    u32 c4_2 = 0;
    u32 c4_3 = 0;
    u32 c4_4 = 0;
    u32 c4_5 = 0;
    u32 c5_0 = 0;
    u32 c5_1 = 0;
    u32 c5_2 = 0;
    u32 c5_3 = 0;
    u32 c5_4 = 0;
    u32 c5_5 = 0;
    
    // Unrolled matrix multiplication
    c0_0 = fma_mod(A[base + 0], B[base + 0], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 1], B[base + 6], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 2], B[base + 12], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 3], B[base + 18], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 4], B[base + 24], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 5], B[base + 30], c0_0, p, mu);
    c0_1 = fma_mod(A[base + 0], B[base + 1], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 1], B[base + 7], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 2], B[base + 13], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 3], B[base + 19], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 4], B[base + 25], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 5], B[base + 31], c0_1, p, mu);
    c0_2 = fma_mod(A[base + 0], B[base + 2], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 1], B[base + 8], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 2], B[base + 14], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 3], B[base + 20], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 4], B[base + 26], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 5], B[base + 32], c0_2, p, mu);
    c0_3 = fma_mod(A[base + 0], B[base + 3], c0_3, p, mu);
    c0_3 = fma_mod(A[base + 1], B[base + 9], c0_3, p, mu);
    c0_3 = fma_mod(A[base + 2], B[base + 15], c0_3, p, mu);
    c0_3 = fma_mod(A[base + 3], B[base + 21], c0_3, p, mu);
    c0_3 = fma_mod(A[base + 4], B[base + 27], c0_3, p, mu);
    c0_3 = fma_mod(A[base + 5], B[base + 33], c0_3, p, mu);
    c0_4 = fma_mod(A[base + 0], B[base + 4], c0_4, p, mu);
    c0_4 = fma_mod(A[base + 1], B[base + 10], c0_4, p, mu);
    c0_4 = fma_mod(A[base + 2], B[base + 16], c0_4, p, mu);
    c0_4 = fma_mod(A[base + 3], B[base + 22], c0_4, p, mu);
    c0_4 = fma_mod(A[base + 4], B[base + 28], c0_4, p, mu);
    c0_4 = fma_mod(A[base + 5], B[base + 34], c0_4, p, mu);
    c0_5 = fma_mod(A[base + 0], B[base + 5], c0_5, p, mu);
    c0_5 = fma_mod(A[base + 1], B[base + 11], c0_5, p, mu);
    c0_5 = fma_mod(A[base + 2], B[base + 17], c0_5, p, mu);
    c0_5 = fma_mod(A[base + 3], B[base + 23], c0_5, p, mu);
    c0_5 = fma_mod(A[base + 4], B[base + 29], c0_5, p, mu);
    c0_5 = fma_mod(A[base + 5], B[base + 35], c0_5, p, mu);
    c1_0 = fma_mod(A[base + 6], B[base + 0], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 7], B[base + 6], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 8], B[base + 12], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 9], B[base + 18], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 10], B[base + 24], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 11], B[base + 30], c1_0, p, mu);
    c1_1 = fma_mod(A[base + 6], B[base + 1], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 7], B[base + 7], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 8], B[base + 13], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 9], B[base + 19], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 10], B[base + 25], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 11], B[base + 31], c1_1, p, mu);
    c1_2 = fma_mod(A[base + 6], B[base + 2], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 7], B[base + 8], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 8], B[base + 14], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 9], B[base + 20], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 10], B[base + 26], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 11], B[base + 32], c1_2, p, mu);
    c1_3 = fma_mod(A[base + 6], B[base + 3], c1_3, p, mu);
    c1_3 = fma_mod(A[base + 7], B[base + 9], c1_3, p, mu);
    c1_3 = fma_mod(A[base + 8], B[base + 15], c1_3, p, mu);
    c1_3 = fma_mod(A[base + 9], B[base + 21], c1_3, p, mu);
    c1_3 = fma_mod(A[base + 10], B[base + 27], c1_3, p, mu);
    c1_3 = fma_mod(A[base + 11], B[base + 33], c1_3, p, mu);
    c1_4 = fma_mod(A[base + 6], B[base + 4], c1_4, p, mu);
    c1_4 = fma_mod(A[base + 7], B[base + 10], c1_4, p, mu);
    c1_4 = fma_mod(A[base + 8], B[base + 16], c1_4, p, mu);
    c1_4 = fma_mod(A[base + 9], B[base + 22], c1_4, p, mu);
    c1_4 = fma_mod(A[base + 10], B[base + 28], c1_4, p, mu);
    c1_4 = fma_mod(A[base + 11], B[base + 34], c1_4, p, mu);
    c1_5 = fma_mod(A[base + 6], B[base + 5], c1_5, p, mu);
    c1_5 = fma_mod(A[base + 7], B[base + 11], c1_5, p, mu);
    c1_5 = fma_mod(A[base + 8], B[base + 17], c1_5, p, mu);
    c1_5 = fma_mod(A[base + 9], B[base + 23], c1_5, p, mu);
    c1_5 = fma_mod(A[base + 10], B[base + 29], c1_5, p, mu);
    c1_5 = fma_mod(A[base + 11], B[base + 35], c1_5, p, mu);
    c2_0 = fma_mod(A[base + 12], B[base + 0], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 13], B[base + 6], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 14], B[base + 12], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 15], B[base + 18], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 16], B[base + 24], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 17], B[base + 30], c2_0, p, mu);
    c2_1 = fma_mod(A[base + 12], B[base + 1], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 13], B[base + 7], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 14], B[base + 13], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 15], B[base + 19], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 16], B[base + 25], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 17], B[base + 31], c2_1, p, mu);
    c2_2 = fma_mod(A[base + 12], B[base + 2], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 13], B[base + 8], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 14], B[base + 14], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 15], B[base + 20], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 16], B[base + 26], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 17], B[base + 32], c2_2, p, mu);
    c2_3 = fma_mod(A[base + 12], B[base + 3], c2_3, p, mu);
    c2_3 = fma_mod(A[base + 13], B[base + 9], c2_3, p, mu);
    c2_3 = fma_mod(A[base + 14], B[base + 15], c2_3, p, mu);
    c2_3 = fma_mod(A[base + 15], B[base + 21], c2_3, p, mu);
    c2_3 = fma_mod(A[base + 16], B[base + 27], c2_3, p, mu);
    c2_3 = fma_mod(A[base + 17], B[base + 33], c2_3, p, mu);
    c2_4 = fma_mod(A[base + 12], B[base + 4], c2_4, p, mu);
    c2_4 = fma_mod(A[base + 13], B[base + 10], c2_4, p, mu);
    c2_4 = fma_mod(A[base + 14], B[base + 16], c2_4, p, mu);
    c2_4 = fma_mod(A[base + 15], B[base + 22], c2_4, p, mu);
    c2_4 = fma_mod(A[base + 16], B[base + 28], c2_4, p, mu);
    c2_4 = fma_mod(A[base + 17], B[base + 34], c2_4, p, mu);
    c2_5 = fma_mod(A[base + 12], B[base + 5], c2_5, p, mu);
    c2_5 = fma_mod(A[base + 13], B[base + 11], c2_5, p, mu);
    c2_5 = fma_mod(A[base + 14], B[base + 17], c2_5, p, mu);
    c2_5 = fma_mod(A[base + 15], B[base + 23], c2_5, p, mu);
    c2_5 = fma_mod(A[base + 16], B[base + 29], c2_5, p, mu);
    c2_5 = fma_mod(A[base + 17], B[base + 35], c2_5, p, mu);
    c3_0 = fma_mod(A[base + 18], B[base + 0], c3_0, p, mu);
    c3_0 = fma_mod(A[base + 19], B[base + 6], c3_0, p, mu);
    c3_0 = fma_mod(A[base + 20], B[base + 12], c3_0, p, mu);
    c3_0 = fma_mod(A[base + 21], B[base + 18], c3_0, p, mu);
    c3_0 = fma_mod(A[base + 22], B[base + 24], c3_0, p, mu);
    c3_0 = fma_mod(A[base + 23], B[base + 30], c3_0, p, mu);
    c3_1 = fma_mod(A[base + 18], B[base + 1], c3_1, p, mu);
    c3_1 = fma_mod(A[base + 19], B[base + 7], c3_1, p, mu);
    c3_1 = fma_mod(A[base + 20], B[base + 13], c3_1, p, mu);
    c3_1 = fma_mod(A[base + 21], B[base + 19], c3_1, p, mu);
    c3_1 = fma_mod(A[base + 22], B[base + 25], c3_1, p, mu);
    c3_1 = fma_mod(A[base + 23], B[base + 31], c3_1, p, mu);
    c3_2 = fma_mod(A[base + 18], B[base + 2], c3_2, p, mu);
    c3_2 = fma_mod(A[base + 19], B[base + 8], c3_2, p, mu);
    c3_2 = fma_mod(A[base + 20], B[base + 14], c3_2, p, mu);
    c3_2 = fma_mod(A[base + 21], B[base + 20], c3_2, p, mu);
    c3_2 = fma_mod(A[base + 22], B[base + 26], c3_2, p, mu);
    c3_2 = fma_mod(A[base + 23], B[base + 32], c3_2, p, mu);
    c3_3 = fma_mod(A[base + 18], B[base + 3], c3_3, p, mu);
    c3_3 = fma_mod(A[base + 19], B[base + 9], c3_3, p, mu);
    c3_3 = fma_mod(A[base + 20], B[base + 15], c3_3, p, mu);
    c3_3 = fma_mod(A[base + 21], B[base + 21], c3_3, p, mu);
    c3_3 = fma_mod(A[base + 22], B[base + 27], c3_3, p, mu);
    c3_3 = fma_mod(A[base + 23], B[base + 33], c3_3, p, mu);
    c3_4 = fma_mod(A[base + 18], B[base + 4], c3_4, p, mu);
    c3_4 = fma_mod(A[base + 19], B[base + 10], c3_4, p, mu);
    c3_4 = fma_mod(A[base + 20], B[base + 16], c3_4, p, mu);
    c3_4 = fma_mod(A[base + 21], B[base + 22], c3_4, p, mu);
    c3_4 = fma_mod(A[base + 22], B[base + 28], c3_4, p, mu);
    c3_4 = fma_mod(A[base + 23], B[base + 34], c3_4, p, mu);
    c3_5 = fma_mod(A[base + 18], B[base + 5], c3_5, p, mu);
    c3_5 = fma_mod(A[base + 19], B[base + 11], c3_5, p, mu);
    c3_5 = fma_mod(A[base + 20], B[base + 17], c3_5, p, mu);
    c3_5 = fma_mod(A[base + 21], B[base + 23], c3_5, p, mu);
    c3_5 = fma_mod(A[base + 22], B[base + 29], c3_5, p, mu);
    c3_5 = fma_mod(A[base + 23], B[base + 35], c3_5, p, mu);
    c4_0 = fma_mod(A[base + 24], B[base + 0], c4_0, p, mu);
    c4_0 = fma_mod(A[base + 25], B[base + 6], c4_0, p, mu);
    c4_0 = fma_mod(A[base + 26], B[base + 12], c4_0, p, mu);
    c4_0 = fma_mod(A[base + 27], B[base + 18], c4_0, p, mu);
    c4_0 = fma_mod(A[base + 28], B[base + 24], c4_0, p, mu);
    c4_0 = fma_mod(A[base + 29], B[base + 30], c4_0, p, mu);
    c4_1 = fma_mod(A[base + 24], B[base + 1], c4_1, p, mu);
    c4_1 = fma_mod(A[base + 25], B[base + 7], c4_1, p, mu);
    c4_1 = fma_mod(A[base + 26], B[base + 13], c4_1, p, mu);
    c4_1 = fma_mod(A[base + 27], B[base + 19], c4_1, p, mu);
    c4_1 = fma_mod(A[base + 28], B[base + 25], c4_1, p, mu);
    c4_1 = fma_mod(A[base + 29], B[base + 31], c4_1, p, mu);
    c4_2 = fma_mod(A[base + 24], B[base + 2], c4_2, p, mu);
    c4_2 = fma_mod(A[base + 25], B[base + 8], c4_2, p, mu);
    c4_2 = fma_mod(A[base + 26], B[base + 14], c4_2, p, mu);
    c4_2 = fma_mod(A[base + 27], B[base + 20], c4_2, p, mu);
    c4_2 = fma_mod(A[base + 28], B[base + 26], c4_2, p, mu);
    c4_2 = fma_mod(A[base + 29], B[base + 32], c4_2, p, mu);
    c4_3 = fma_mod(A[base + 24], B[base + 3], c4_3, p, mu);
    c4_3 = fma_mod(A[base + 25], B[base + 9], c4_3, p, mu);
    c4_3 = fma_mod(A[base + 26], B[base + 15], c4_3, p, mu);
    c4_3 = fma_mod(A[base + 27], B[base + 21], c4_3, p, mu);
    c4_3 = fma_mod(A[base + 28], B[base + 27], c4_3, p, mu);
    c4_3 = fma_mod(A[base + 29], B[base + 33], c4_3, p, mu);
    c4_4 = fma_mod(A[base + 24], B[base + 4], c4_4, p, mu);
    c4_4 = fma_mod(A[base + 25], B[base + 10], c4_4, p, mu);
    c4_4 = fma_mod(A[base + 26], B[base + 16], c4_4, p, mu);
    c4_4 = fma_mod(A[base + 27], B[base + 22], c4_4, p, mu);
    c4_4 = fma_mod(A[base + 28], B[base + 28], c4_4, p, mu);
    c4_4 = fma_mod(A[base + 29], B[base + 34], c4_4, p, mu);
    c4_5 = fma_mod(A[base + 24], B[base + 5], c4_5, p, mu);
    c4_5 = fma_mod(A[base + 25], B[base + 11], c4_5, p, mu);
    c4_5 = fma_mod(A[base + 26], B[base + 17], c4_5, p, mu);
    c4_5 = fma_mod(A[base + 27], B[base + 23], c4_5, p, mu);
    c4_5 = fma_mod(A[base + 28], B[base + 29], c4_5, p, mu);
    c4_5 = fma_mod(A[base + 29], B[base + 35], c4_5, p, mu);
    c5_0 = fma_mod(A[base + 30], B[base + 0], c5_0, p, mu);
    c5_0 = fma_mod(A[base + 31], B[base + 6], c5_0, p, mu);
    c5_0 = fma_mod(A[base + 32], B[base + 12], c5_0, p, mu);
    c5_0 = fma_mod(A[base + 33], B[base + 18], c5_0, p, mu);
    c5_0 = fma_mod(A[base + 34], B[base + 24], c5_0, p, mu);
    c5_0 = fma_mod(A[base + 35], B[base + 30], c5_0, p, mu);
    c5_1 = fma_mod(A[base + 30], B[base + 1], c5_1, p, mu);
    c5_1 = fma_mod(A[base + 31], B[base + 7], c5_1, p, mu);
    c5_1 = fma_mod(A[base + 32], B[base + 13], c5_1, p, mu);
    c5_1 = fma_mod(A[base + 33], B[base + 19], c5_1, p, mu);
    c5_1 = fma_mod(A[base + 34], B[base + 25], c5_1, p, mu);
    c5_1 = fma_mod(A[base + 35], B[base + 31], c5_1, p, mu);
    c5_2 = fma_mod(A[base + 30], B[base + 2], c5_2, p, mu);
    c5_2 = fma_mod(A[base + 31], B[base + 8], c5_2, p, mu);
    c5_2 = fma_mod(A[base + 32], B[base + 14], c5_2, p, mu);
    c5_2 = fma_mod(A[base + 33], B[base + 20], c5_2, p, mu);
    c5_2 = fma_mod(A[base + 34], B[base + 26], c5_2, p, mu);
    c5_2 = fma_mod(A[base + 35], B[base + 32], c5_2, p, mu);
    c5_3 = fma_mod(A[base + 30], B[base + 3], c5_3, p, mu);
    c5_3 = fma_mod(A[base + 31], B[base + 9], c5_3, p, mu);
    c5_3 = fma_mod(A[base + 32], B[base + 15], c5_3, p, mu);
    c5_3 = fma_mod(A[base + 33], B[base + 21], c5_3, p, mu);
    c5_3 = fma_mod(A[base + 34], B[base + 27], c5_3, p, mu);
    c5_3 = fma_mod(A[base + 35], B[base + 33], c5_3, p, mu);
    c5_4 = fma_mod(A[base + 30], B[base + 4], c5_4, p, mu);
    c5_4 = fma_mod(A[base + 31], B[base + 10], c5_4, p, mu);
    c5_4 = fma_mod(A[base + 32], B[base + 16], c5_4, p, mu);
    c5_4 = fma_mod(A[base + 33], B[base + 22], c5_4, p, mu);
    c5_4 = fma_mod(A[base + 34], B[base + 28], c5_4, p, mu);
    c5_4 = fma_mod(A[base + 35], B[base + 34], c5_4, p, mu);
    c5_5 = fma_mod(A[base + 30], B[base + 5], c5_5, p, mu);
    c5_5 = fma_mod(A[base + 31], B[base + 11], c5_5, p, mu);
    c5_5 = fma_mod(A[base + 32], B[base + 17], c5_5, p, mu);
    c5_5 = fma_mod(A[base + 33], B[base + 23], c5_5, p, mu);
    c5_5 = fma_mod(A[base + 34], B[base + 29], c5_5, p, mu);
    c5_5 = fma_mod(A[base + 35], B[base + 35], c5_5, p, mu);
    
    // Store results
    C[base + 0] = c0_0;
    C[base + 1] = c0_1;
    C[base + 2] = c0_2;
    C[base + 3] = c0_3;
    C[base + 4] = c0_4;
    C[base + 5] = c0_5;
    C[base + 6] = c1_0;
    C[base + 7] = c1_1;
    C[base + 8] = c1_2;
    C[base + 9] = c1_3;
    C[base + 10] = c1_4;
    C[base + 11] = c1_5;
    C[base + 12] = c2_0;
    C[base + 13] = c2_1;
    C[base + 14] = c2_2;
    C[base + 15] = c2_3;
    C[base + 16] = c2_4;
    C[base + 17] = c2_5;
    C[base + 18] = c3_0;
    C[base + 19] = c3_1;
    C[base + 20] = c3_2;
    C[base + 21] = c3_3;
    C[base + 22] = c3_4;
    C[base + 23] = c3_5;
    C[base + 24] = c4_0;
    C[base + 25] = c4_1;
    C[base + 26] = c4_2;
    C[base + 27] = c4_3;
    C[base + 28] = c4_4;
    C[base + 29] = c4_5;
    C[base + 30] = c5_0;
    C[base + 31] = c5_1;
    C[base + 32] = c5_2;
    C[base + 33] = c5_3;
    C[base + 34] = c5_4;
    C[base + 35] = c5_5;
}
#endif

void rns_gemm_mod_m6(
    u32* C, const u32* A, const u32* B,
    const u32* d_primes, const u64* d_mus,
    int K, int batch)
{
#ifdef RNS_HAS_GPU
    dim3 grid(K, (batch + 255) / 256);
    dim3 block(256);
    hipLaunchKernelGGL(k_gemm_mod_m6, grid, block, 0, 0,
                       C, A, B, d_primes, d_mus, K, batch);
#else
    // CPU fallback
    constexpr int m = 6;
    constexpr int E = m * m;
    for (int k = 0; k < K; ++k) {
        u32 p = d_primes[k];
        u64 mu = d_mus[k];
        for (int b = 0; b < batch; ++b) {
            int base = k * batch * E + b * E;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    u32 acc = 0;
                    for (int t = 0; t < m; ++t) {
                        acc = fma_mod(A[base + i*m + t], 
                                     B[base + t*m + j], 
                                     acc, p, mu);
                    }
                    C[base + i*m + j] = acc;
                }
            }
        }
    }
#endif
}

} // namespace rns

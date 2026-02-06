// Auto-generated GEMM kernel for m=3
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#include "rns/config.h"
#include "rns/modops.h"

#ifdef RNS_HAS_GPU
#include <hip/hip_runtime.h>
#endif

namespace rns {

#ifdef RNS_HAS_GPU
__global__ void k_gemm_mod_m3(
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
    constexpr int E = 9;
    int base = k * batch * E + b * E;
    
    // Load A and B into registers
    u32 c0_0 = 0;
    u32 c0_1 = 0;
    u32 c0_2 = 0;
    u32 c1_0 = 0;
    u32 c1_1 = 0;
    u32 c1_2 = 0;
    u32 c2_0 = 0;
    u32 c2_1 = 0;
    u32 c2_2 = 0;
    
    // Unrolled matrix multiplication
    c0_0 = fma_mod(A[base + 0], B[base + 0], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 1], B[base + 3], c0_0, p, mu);
    c0_0 = fma_mod(A[base + 2], B[base + 6], c0_0, p, mu);
    c0_1 = fma_mod(A[base + 0], B[base + 1], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 1], B[base + 4], c0_1, p, mu);
    c0_1 = fma_mod(A[base + 2], B[base + 7], c0_1, p, mu);
    c0_2 = fma_mod(A[base + 0], B[base + 2], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 1], B[base + 5], c0_2, p, mu);
    c0_2 = fma_mod(A[base + 2], B[base + 8], c0_2, p, mu);
    c1_0 = fma_mod(A[base + 3], B[base + 0], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 4], B[base + 3], c1_0, p, mu);
    c1_0 = fma_mod(A[base + 5], B[base + 6], c1_0, p, mu);
    c1_1 = fma_mod(A[base + 3], B[base + 1], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 4], B[base + 4], c1_1, p, mu);
    c1_1 = fma_mod(A[base + 5], B[base + 7], c1_1, p, mu);
    c1_2 = fma_mod(A[base + 3], B[base + 2], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 4], B[base + 5], c1_2, p, mu);
    c1_2 = fma_mod(A[base + 5], B[base + 8], c1_2, p, mu);
    c2_0 = fma_mod(A[base + 6], B[base + 0], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 7], B[base + 3], c2_0, p, mu);
    c2_0 = fma_mod(A[base + 8], B[base + 6], c2_0, p, mu);
    c2_1 = fma_mod(A[base + 6], B[base + 1], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 7], B[base + 4], c2_1, p, mu);
    c2_1 = fma_mod(A[base + 8], B[base + 7], c2_1, p, mu);
    c2_2 = fma_mod(A[base + 6], B[base + 2], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 7], B[base + 5], c2_2, p, mu);
    c2_2 = fma_mod(A[base + 8], B[base + 8], c2_2, p, mu);
    
    // Store results
    C[base + 0] = c0_0;
    C[base + 1] = c0_1;
    C[base + 2] = c0_2;
    C[base + 3] = c1_0;
    C[base + 4] = c1_1;
    C[base + 5] = c1_2;
    C[base + 6] = c2_0;
    C[base + 7] = c2_1;
    C[base + 8] = c2_2;
}
#endif

void rns_gemm_mod_m3(
    u32* C, const u32* A, const u32* B,
    const u32* d_primes, const u64* d_mus,
    int K, int batch)
{
#ifdef RNS_HAS_GPU
    dim3 grid(K, (batch + 255) / 256);
    dim3 block(256);
    hipLaunchKernelGGL(k_gemm_mod_m3, grid, block, 0, 0,
                       C, A, B, d_primes, d_mus, K, batch);
#else
    // CPU fallback
    constexpr int m = 3;
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

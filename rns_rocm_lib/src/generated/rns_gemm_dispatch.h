// Auto-generated GEMM dispatcher
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#ifndef RNS_GEMM_DISPATCH_H
#define RNS_GEMM_DISPATCH_H

#include "rns/config.h"

namespace rns {

// Forward declarations for all specialized kernels
void rns_gemm_mod_m2(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m3(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m4(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m5(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m6(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m7(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m8(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m9(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m10(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m11(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);
void rns_gemm_mod_m12(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);

// Dispatcher function
inline void rns_gemm_dispatch(
    u32* C, const u32* A, const u32* B,
    const u32* d_primes, const u64* d_mus,
    int K, int batch, int m)
{
    switch (m) {
        case 2: rns_gemm_mod_m2(C, A, B, d_primes, d_mus, K, batch); break;
        case 3: rns_gemm_mod_m3(C, A, B, d_primes, d_mus, K, batch); break;
        case 4: rns_gemm_mod_m4(C, A, B, d_primes, d_mus, K, batch); break;
        case 5: rns_gemm_mod_m5(C, A, B, d_primes, d_mus, K, batch); break;
        case 6: rns_gemm_mod_m6(C, A, B, d_primes, d_mus, K, batch); break;
        case 7: rns_gemm_mod_m7(C, A, B, d_primes, d_mus, K, batch); break;
        case 8: rns_gemm_mod_m8(C, A, B, d_primes, d_mus, K, batch); break;
        case 9: rns_gemm_mod_m9(C, A, B, d_primes, d_mus, K, batch); break;
        case 10: rns_gemm_mod_m10(C, A, B, d_primes, d_mus, K, batch); break;
        case 11: rns_gemm_mod_m11(C, A, B, d_primes, d_mus, K, batch); break;
        case 12: rns_gemm_mod_m12(C, A, B, d_primes, d_mus, K, batch); break;
        default:
            // Generic fallback for unsupported sizes
            break;
    }
}

} // namespace rns

#endif // RNS_GEMM_DISPATCH_H

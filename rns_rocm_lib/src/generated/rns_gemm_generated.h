// Auto-generated GEMM kernel declarations
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#ifndef RNS_GEMM_GENERATED_H
#define RNS_GEMM_GENERATED_H

#include "rns/config.h"

namespace rns {

constexpr int GEMM_MAX_M = 12;
constexpr int GEMM_MIN_M = 2;

// Specialized kernel declarations
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

// Check if m has a specialized kernel
inline bool has_specialized_gemm(int m) {
    return m >= GEMM_MIN_M && m <= GEMM_MAX_M;
}

} // namespace rns

#endif // RNS_GEMM_GENERATED_H

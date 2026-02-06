#pragma once
#include <cstdint>
#include "rns_types.h"

namespace rns {

/**
 * Elementwise RNS addition.
 * Layout: data[K][N] flattened as k*N + idx
 * 
 * @param out Output array (device)
 * @param a   First operand (device)
 * @param b   Second operand (device)
 * @param mods Moduli array (device), length K
 * @param K   Number of primes
 * @param N   Elements per prime
 */
void rns_add_u32(
    uint32_t* out, const uint32_t* a, const uint32_t* b,
    const Modulus32* mods, int K, int N);

/**
 * Elementwise RNS multiplication.
 */
void rns_mul_u32(
    uint32_t* out, const uint32_t* a, const uint32_t* b,
    const Modulus32* mods, int K, int N);

/**
 * Elementwise RNS subtraction.
 */
void rns_sub_u32(
    uint32_t* out, const uint32_t* a, const uint32_t* b,
    const Modulus32* mods, int K, int N);

/**
 * Batched small matrix multiplication mod p.
 * 
 * Computes C[k][b] = A[k][b] @ B[k][b] mod p_k for all k, b.
 * 
 * Layout: A, B, C are [K][B*E] where E = m*m
 *   element (k, b, i, j) at index: k*(B*E) + b*E + i*m + j
 * 
 * Specialized kernels for m = 4, 6, 8, 10.
 * Falls back to generic kernel for other sizes.
 * 
 * @param C    Output matrices (device)
 * @param A    Left matrices (device)
 * @param B    Right matrices (device)
 * @param mods Moduli array (device)
 * @param K    Number of primes
 * @param B    Batch size
 * @param m    Matrix dimension
 */
void rns_gemm_mod_u32(
    uint32_t* C, const uint32_t* A, const uint32_t* Bmat,
    const Modulus32* mods, int K, int B, int m);

// Specialized GEMM kernels (called by rns_gemm_mod_u32)
void rns_gemm_mod_m4(uint32_t* C, const uint32_t* A, const uint32_t* B,
                      const Modulus32* mods, int K, int Batch);
void rns_gemm_mod_m6(uint32_t* C, const uint32_t* A, const uint32_t* B,
                      const Modulus32* mods, int K, int Batch);
void rns_gemm_mod_m8(uint32_t* C, const uint32_t* A, const uint32_t* B,
                      const Modulus32* mods, int K, int Batch);
void rns_gemm_mod_m10(uint32_t* C, const uint32_t* A, const uint32_t* B,
                       const Modulus32* mods, int K, int Batch);

}  // namespace rns

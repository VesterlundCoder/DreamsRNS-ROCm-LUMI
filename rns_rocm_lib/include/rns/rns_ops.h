#pragma once
#include <cstdint>
#include "rns_types.h"
#include <hip/hip_runtime.h>

// Device/host function specifiers
#if defined(__HIPCC__) || defined(__CUDACC__)
  #define RNS_DEVICE __device__ __forceinline__
  #define RNS_HOST_DEVICE __host__ __device__ __forceinline__
#else
  #define RNS_DEVICE inline
  #define RNS_HOST_DEVICE inline
#endif

namespace rns {

/**
 * Barrett reduction for x in [0, 2^64).
 * Returns x mod p using precomputed mu = floor(2^64/p).
 */
RNS_HOST_DEVICE uint32_t barrett_reduce_u64(uint64_t x, const Modulus32& mod) {
#if defined(__HIP_DEVICE_COMPILE__)
  // GPU: use __umul64hi for high 64 bits
  uint64_t q = __umul64hi(x, mod.mu);
#else
  // CPU: use __uint128_t
  __uint128_t prod = ((__uint128_t)x * (__uint128_t)mod.mu);
  uint64_t q = (uint64_t)(prod >> 64);
#endif
  uint64_t r = x - q * (uint64_t)mod.p;
  // r may be in [0, 2p). Fix up:
  if (r >= mod.p) r -= mod.p;
  if (r >= mod.p) r -= mod.p;
  return (uint32_t)r;
}

/**
 * Modular addition: (a + b) mod p
 * Assumes a, b < p
 */
RNS_HOST_DEVICE uint32_t add_mod(uint32_t a, uint32_t b, const Modulus32& mod) {
  uint64_t s = (uint64_t)a + (uint64_t)b;
  if (s >= mod.p) s -= mod.p;
  return (uint32_t)s;
}

/**
 * Modular subtraction: (a - b) mod p
 * Assumes a, b < p
 */
RNS_HOST_DEVICE uint32_t sub_mod(uint32_t a, uint32_t b, const Modulus32& mod) {
  uint64_t s = (uint64_t)a + (uint64_t)mod.p - (uint64_t)b;
  if (s >= mod.p) s -= mod.p;
  return (uint32_t)s;
}

/**
 * Modular multiplication: (a * b) mod p
 * Uses Barrett reduction.
 */
RNS_HOST_DEVICE uint32_t mul_mod(uint32_t a, uint32_t b, const Modulus32& mod) {
  uint64_t x = (uint64_t)a * (uint64_t)b;
  return barrett_reduce_u64(x, mod);
}

/**
 * Fused multiply-add: (a * b + c) mod p
 */
RNS_HOST_DEVICE uint32_t fma_mod(uint32_t a, uint32_t b, uint32_t c, const Modulus32& mod) {
  uint64_t x = (uint64_t)a * (uint64_t)b + (uint64_t)c;
  return barrett_reduce_u64(x, mod);
}

/**
 * Modular negation: (-a) mod p = p - a
 * Assumes a < p
 */
RNS_HOST_DEVICE uint32_t neg_mod(uint32_t a, const Modulus32& mod) {
  return (a == 0) ? 0 : (mod.p - a);
}

/**
 * Modular exponentiation by squaring: a^exp mod p
 * For small exponents (2, 3, etc.)
 */
RNS_HOST_DEVICE uint32_t pow_mod(uint32_t a, uint32_t exp, const Modulus32& mod) {
  uint32_t result = 1;
  uint32_t base = a;
  while (exp > 0) {
    if (exp & 1) {
      result = mul_mod(result, base, mod);
    }
    base = mul_mod(base, base, mod);
    exp >>= 1;
  }
  return result;
}

/**
 * Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
 * Requires p to be prime. Returns 0 if a == 0.
 */
RNS_HOST_DEVICE uint32_t inv_mod(uint32_t a, const Modulus32& mod) {
  if (a == 0) return 0;  // Not invertible
  return pow_mod(a, mod.p - 2, mod);
}

// Convenience functions for powers
RNS_HOST_DEVICE uint32_t pow2_mod(uint32_t a, const Modulus32& mod) {
  return mul_mod(a, a, mod);
}

RNS_HOST_DEVICE uint32_t pow3_mod(uint32_t a, const Modulus32& mod) {
  uint32_t a2 = mul_mod(a, a, mod);
  return mul_mod(a2, a, mod);
}

}  // namespace rns

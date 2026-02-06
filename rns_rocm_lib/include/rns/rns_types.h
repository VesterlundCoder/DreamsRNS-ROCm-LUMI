#pragma once
#include <cstdint>
#include <vector>

namespace rns {

/**
 * 32-bit prime modulus with precomputed Barrett constant.
 * mu = floor(2^64 / p) for fast modular reduction.
 */
struct Modulus32 {
  uint32_t p;      // The prime modulus
  uint64_t mu;     // Barrett constant: floor(2^64 / p)
  uint32_t r2;     // Montgomery: R^2 mod p (optional, for future use)
  uint32_t p_inv;  // Montgomery: -p^(-1) mod 2^32 (optional)
};

/**
 * Collection of primes with precomputed constants.
 */
struct PrimeSet {
  std::vector<Modulus32> mods;
  int K;  // Number of primes
  
  // Total bit capacity: sum of log2(p_k)
  double bit_capacity() const;
  
  // Product of all primes (for CRT capacity check)
  // Returns empty if overflow would occur
  std::vector<uint32_t> product_limbs() const;
};

/**
 * Device context holding GPU resources.
 */
struct DeviceContext {
  Modulus32* d_mods = nullptr;  // Device array of moduli, length K
  int K = 0;                     // Number of primes
  bool valid = false;            // Context is initialized
};

/**
 * RNS-encoded matrix batch.
 * Layout: data[k * (B * E) + b * E + (i * m + j)]
 * where E = m * m, k is prime index, b is batch index.
 */
struct RnsMatrixBatch {
  uint32_t* data = nullptr;  // Device or host pointer
  int K;      // Number of primes
  int B;      // Batch size
  int m;      // Matrix dimension
  bool on_device;
  
  size_t total_elements() const { return (size_t)K * B * m * m; }
  size_t bytes() const { return total_elements() * sizeof(uint32_t); }
};

}  // namespace rns

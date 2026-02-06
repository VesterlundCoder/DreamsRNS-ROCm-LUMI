#pragma once
#include <cstdint>
#include <vector>
#include "rns_types.h"

namespace rns {

/**
 * Big integer represented as little-endian limbs (base 2^32).
 * limbs[0] is the least significant word.
 */
struct BigInt {
  std::vector<uint32_t> limbs;
  bool negative = false;
  
  BigInt() = default;
  BigInt(uint64_t val);
  BigInt(const std::vector<uint32_t>& limbs_, bool neg = false);
  
  // Comparison
  bool is_zero() const;
  int compare_abs(const BigInt& other) const;
  bool operator==(const BigInt& other) const;
  bool operator!=(const BigInt& other) const;
  
  // Arithmetic (for CRT)
  BigInt operator+(const BigInt& other) const;
  BigInt operator-(const BigInt& other) const;
  BigInt operator*(const BigInt& other) const;
  BigInt operator*(uint32_t scalar) const;
  
  // Division and modulo
  std::pair<BigInt, BigInt> divmod(const BigInt& divisor) const;
  BigInt operator%(const BigInt& mod) const;
  uint32_t mod_u32(uint32_t p) const;
  
  // Utilities
  void normalize();  // Remove leading zeros
  std::string to_string() const;
  static BigInt from_string(const std::string& s);
};

/**
 * Precomputed constants for Garner's CRT algorithm.
 * 
 * For primes p_0, p_1, ..., p_{K-1}:
 *   M_i = p_0 * p_1 * ... * p_{i-1}  (M_0 = 1)
 *   c_i = M_i^{-1} mod p_i
 */
struct CrtPlan {
  std::vector<Modulus32> mods;  // Primes with Barrett constants
  int K;                         // Number of primes
  
  // Garner coefficients: c[i] = (p_0 * ... * p_{i-1})^{-1} mod p_i
  std::vector<uint32_t> garner_c;
  
  // Partial products for final combination
  // M_partial[i] = p_0 * p_1 * ... * p_{i-1} as BigInt
  std::vector<BigInt> M_partial;
  
  // Total modulus M = p_0 * p_1 * ... * p_{K-1}
  BigInt M_total;
  
  // Bit capacity (sum of log2(p_i))
  double bit_capacity;
  
  // Create plan from primes
  static CrtPlan create(const std::vector<uint32_t>& primes);
  
  // Verify all primes are distinct and coprime
  bool verify() const;
};

}  // namespace rns

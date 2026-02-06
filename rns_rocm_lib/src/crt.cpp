#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "rns/crt.h"
#include "rns/crt_types.h"
#include "rns/rns_ops.h"
#include "rns/utils.h"

namespace rns {

// ============================================================================
// BigInt Implementation
// ============================================================================

BigInt::BigInt(uint64_t val) {
  if (val == 0) {
    limbs = {0};
  } else {
    limbs.push_back((uint32_t)(val & 0xFFFFFFFF));
    if (val >> 32) {
      limbs.push_back((uint32_t)(val >> 32));
    }
  }
  negative = false;
}

BigInt::BigInt(const std::vector<uint32_t>& limbs_, bool neg)
    : limbs(limbs_), negative(neg) {
  normalize();
}

bool BigInt::is_zero() const {
  return limbs.empty() || (limbs.size() == 1 && limbs[0] == 0);
}

void BigInt::normalize() {
  while (limbs.size() > 1 && limbs.back() == 0) {
    limbs.pop_back();
  }
  if (is_zero()) {
    negative = false;
  }
}

int BigInt::compare_abs(const BigInt& other) const {
  if (limbs.size() != other.limbs.size()) {
    return limbs.size() < other.limbs.size() ? -1 : 1;
  }
  for (int i = (int)limbs.size() - 1; i >= 0; --i) {
    if (limbs[i] != other.limbs[i]) {
      return limbs[i] < other.limbs[i] ? -1 : 1;
    }
  }
  return 0;
}

bool BigInt::operator==(const BigInt& other) const {
  return negative == other.negative && compare_abs(other) == 0;
}

bool BigInt::operator!=(const BigInt& other) const {
  return !(*this == other);
}

BigInt BigInt::operator+(const BigInt& other) const {
  if (negative == other.negative) {
    // Same sign: add magnitudes
    BigInt result;
    result.negative = negative;
    result.limbs.resize(std::max(limbs.size(), other.limbs.size()) + 1, 0);
    
    uint64_t carry = 0;
    for (size_t i = 0; i < result.limbs.size(); ++i) {
      uint64_t a = (i < limbs.size()) ? limbs[i] : 0;
      uint64_t b = (i < other.limbs.size()) ? other.limbs[i] : 0;
      uint64_t sum = a + b + carry;
      result.limbs[i] = (uint32_t)(sum & 0xFFFFFFFF);
      carry = sum >> 32;
    }
    result.normalize();
    return result;
  } else {
    // Different signs: subtract
    if (negative) {
      // (-a) + b = b - a
      BigInt pos_this = *this;
      pos_this.negative = false;
      return other - pos_this;
    } else {
      // a + (-b) = a - b
      BigInt pos_other = other;
      pos_other.negative = false;
      return *this - pos_other;
    }
  }
}

BigInt BigInt::operator-(const BigInt& other) const {
  if (negative != other.negative) {
    // Different signs: a - (-b) = a + b
    BigInt pos_other = other;
    pos_other.negative = !other.negative;
    return *this + pos_other;
  }
  
  // Same sign: subtract magnitudes
  int cmp = compare_abs(other);
  if (cmp == 0) {
    return BigInt(0);
  }
  
  const BigInt* larger = (cmp > 0) ? this : &other;
  const BigInt* smaller = (cmp > 0) ? &other : this;
  
  BigInt result;
  result.limbs.resize(larger->limbs.size(), 0);
  result.negative = (cmp > 0) ? negative : !other.negative;
  
  int64_t borrow = 0;
  for (size_t i = 0; i < larger->limbs.size(); ++i) {
    int64_t a = larger->limbs[i];
    int64_t b = (i < smaller->limbs.size()) ? smaller->limbs[i] : 0;
    int64_t diff = a - b - borrow;
    if (diff < 0) {
      diff += ((int64_t)1 << 32);
      borrow = 1;
    } else {
      borrow = 0;
    }
    result.limbs[i] = (uint32_t)diff;
  }
  result.normalize();
  return result;
}

BigInt BigInt::operator*(const BigInt& other) const {
  if (is_zero() || other.is_zero()) {
    return BigInt(0);
  }
  
  BigInt result;
  result.limbs.resize(limbs.size() + other.limbs.size(), 0);
  result.negative = (negative != other.negative);
  
  for (size_t i = 0; i < limbs.size(); ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < other.limbs.size(); ++j) {
      uint64_t prod = (uint64_t)limbs[i] * (uint64_t)other.limbs[j];
      uint64_t sum = (uint64_t)result.limbs[i + j] + prod + carry;
      result.limbs[i + j] = (uint32_t)(sum & 0xFFFFFFFF);
      carry = sum >> 32;
    }
    if (carry) {
      result.limbs[i + other.limbs.size()] += (uint32_t)carry;
    }
  }
  result.normalize();
  return result;
}

BigInt BigInt::operator*(uint32_t scalar) const {
  if (scalar == 0 || is_zero()) {
    return BigInt(0);
  }
  
  BigInt result;
  result.limbs.resize(limbs.size() + 1, 0);
  result.negative = negative;
  
  uint64_t carry = 0;
  for (size_t i = 0; i < limbs.size(); ++i) {
    uint64_t prod = (uint64_t)limbs[i] * (uint64_t)scalar + carry;
    result.limbs[i] = (uint32_t)(prod & 0xFFFFFFFF);
    carry = prod >> 32;
  }
  if (carry) {
    result.limbs[limbs.size()] = (uint32_t)carry;
  }
  result.normalize();
  return result;
}

std::pair<BigInt, BigInt> BigInt::divmod(const BigInt& divisor) const {
  if (divisor.is_zero()) {
    throw std::runtime_error("Division by zero");
  }
  
  if (is_zero()) {
    return {BigInt(0), BigInt(0)};
  }
  
  if (compare_abs(divisor) < 0) {
    return {BigInt(0), *this};
  }
  
  // Simple long division (not optimized, but correct)
  BigInt quotient, remainder;
  quotient.limbs.resize(limbs.size(), 0);
  remainder.limbs = {};
  
  for (int i = (int)limbs.size() - 1; i >= 0; --i) {
    // Shift remainder left by 32 bits and add next limb
    remainder.limbs.insert(remainder.limbs.begin(), limbs[i]);
    remainder.normalize();
    
    // Binary search for quotient digit
    uint32_t lo = 0, hi = 0xFFFFFFFF;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo + 1) / 2;
      BigInt test = divisor * mid;
      if (test.compare_abs(remainder) <= 0) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    
    quotient.limbs[i] = lo;
    if (lo > 0) {
      BigInt sub = divisor * lo;
      remainder = remainder - sub;
    }
  }
  
  quotient.negative = (negative != divisor.negative);
  remainder.negative = negative;
  quotient.normalize();
  remainder.normalize();
  
  return {quotient, remainder};
}

BigInt BigInt::operator%(const BigInt& mod) const {
  auto [q, r] = divmod(mod);
  if (r.negative && !r.is_zero()) {
    r = r + mod;
  }
  return r;
}

uint32_t BigInt::mod_u32(uint32_t p) const {
  if (p == 0) {
    throw std::runtime_error("Modulo by zero");
  }
  
  uint64_t result = 0;
  uint64_t base = 1;
  
  for (size_t i = 0; i < limbs.size(); ++i) {
    result = (result + ((uint64_t)limbs[i] % p) * (base % p)) % p;
    base = (base * (((uint64_t)1 << 32) % p)) % p;
  }
  
  if (negative && result != 0) {
    result = p - result;
  }
  
  return (uint32_t)result;
}

std::string BigInt::to_string() const {
  if (is_zero()) return "0";
  
  std::string result;
  BigInt tmp = *this;
  tmp.negative = false;
  
  while (!tmp.is_zero()) {
    auto [q, r] = tmp.divmod(BigInt(10));
    result = char('0' + r.limbs[0]) + result;
    tmp = q;
  }
  
  if (negative) {
    result = "-" + result;
  }
  return result;
}

BigInt BigInt::from_string(const std::string& s) {
  if (s.empty()) return BigInt(0);
  
  bool neg = (s[0] == '-');
  size_t start = neg ? 1 : 0;
  
  BigInt result(0);
  for (size_t i = start; i < s.length(); ++i) {
    if (s[i] < '0' || s[i] > '9') {
      throw std::runtime_error("Invalid character in number string");
    }
    result = result * 10 + BigInt(s[i] - '0');
  }
  result.negative = neg;
  return result;
}

// ============================================================================
// CRT Plan
// ============================================================================

CrtPlan CrtPlan::create(const std::vector<uint32_t>& primes) {
  CrtPlan plan;
  plan.K = (int)primes.size();
  
  if (plan.K == 0) {
    throw std::runtime_error("Empty prime list");
  }
  
  // Build moduli with Barrett constants
  plan.mods.resize(plan.K);
  for (int i = 0; i < plan.K; ++i) {
    plan.mods[i].p = primes[i];
    __uint128_t one = ((__uint128_t)1 << 64);
    plan.mods[i].mu = (uint64_t)(one / primes[i]);
  }
  
  // Compute partial products M_i = p_0 * p_1 * ... * p_{i-1}
  plan.M_partial.resize(plan.K);
  plan.M_partial[0] = BigInt(1);
  for (int i = 1; i < plan.K; ++i) {
    plan.M_partial[i] = plan.M_partial[i - 1] * primes[i - 1];
  }
  
  // Total modulus
  plan.M_total = plan.M_partial[plan.K - 1] * primes[plan.K - 1];
  
  // Garner coefficients: c[i] = M_i^{-1} mod p_i
  plan.garner_c.resize(plan.K);
  for (int i = 0; i < plan.K; ++i) {
    uint32_t M_i_mod_pi = plan.M_partial[i].mod_u32(primes[i]);
    plan.garner_c[i] = mod_inverse(M_i_mod_pi, primes[i]);
    
    if (plan.garner_c[i] == 0 && M_i_mod_pi != 0) {
      throw std::runtime_error("Failed to compute modular inverse - primes may not be coprime");
    }
  }
  
  // Bit capacity
  plan.bit_capacity = 0;
  for (int i = 0; i < plan.K; ++i) {
    plan.bit_capacity += std::log2(primes[i]);
  }
  
  return plan;
}

bool CrtPlan::verify() const {
  // Check all primes are distinct
  for (int i = 0; i < K; ++i) {
    for (int j = i + 1; j < K; ++j) {
      if (mods[i].p == mods[j].p) {
        return false;
      }
    }
  }
  
  // Check all are actually prime (basic check)
  for (int i = 0; i < K; ++i) {
    if (!is_prime(mods[i].p)) {
      return false;
    }
  }
  
  return true;
}

// ============================================================================
// CRT Reconstruction (Garner's Algorithm)
// ============================================================================

BigInt crt_reconstruct(const uint32_t* residues, const CrtPlan& plan) {
  // Garner's algorithm:
  // x = r_0 + M_0 * v_0 + M_1 * v_1 + ... + M_{K-1} * v_{K-1}
  // where v_i = (r_i - (x mod p_i)) * c_i mod p_i
  
  std::vector<uint32_t> v(plan.K);
  
  // v[0] = r[0]
  v[0] = residues[0];
  
  // For each subsequent prime
  for (int i = 1; i < plan.K; ++i) {
    uint32_t p_i = plan.mods[i].p;
    Modulus32 mod = plan.mods[i];
    
    // Compute x mod p_i from previous v values
    uint32_t x_mod_pi = 0;
    uint32_t M_mod_pi = 1;
    
    for (int j = 0; j < i; ++j) {
      x_mod_pi = add_mod(x_mod_pi, mul_mod(v[j], M_mod_pi, mod), mod);
      M_mod_pi = mul_mod(M_mod_pi, plan.mods[j].p % p_i, mod);
    }
    
    // v[i] = (r[i] - x_mod_pi) * c[i] mod p_i
    uint32_t diff = sub_mod(residues[i], x_mod_pi, mod);
    v[i] = mul_mod(diff, plan.garner_c[i], mod);
  }
  
  // Reconstruct: x = sum(v[i] * M_partial[i])
  BigInt result(0);
  for (int i = 0; i < plan.K; ++i) {
    result = result + (plan.M_partial[i] * v[i]);
  }
  
  return result;
}

std::vector<BigInt> crt_reconstruct_batch(
    const uint32_t* residues, const CrtPlan& plan, int N) {
  std::vector<BigInt> results(N);
  
  for (int n = 0; n < N; ++n) {
    // Extract residues for element n: residues[k * N + n]
    std::vector<uint32_t> elem_residues(plan.K);
    for (int k = 0; k < plan.K; ++k) {
      elem_residues[k] = residues[k * N + n];
    }
    results[n] = crt_reconstruct(elem_residues.data(), plan);
  }
  
  return results;
}

std::vector<BigInt> crt_reconstruct_matrix(
    const uint32_t* residues, const CrtPlan& plan, int m) {
  int E = m * m;
  return crt_reconstruct_batch(residues, plan, E);
}

// ============================================================================
// RNS Encoding
// ============================================================================

void rns_encode(const BigInt& x, const CrtPlan& plan, uint32_t* out) {
  for (int k = 0; k < plan.K; ++k) {
    out[k] = x.mod_u32(plan.mods[k].p);
  }
}

void rns_encode_u64(uint64_t x, const CrtPlan& plan, uint32_t* out) {
  for (int k = 0; k < plan.K; ++k) {
    out[k] = (uint32_t)(x % plan.mods[k].p);
  }
}

void rns_encode_matrix(
    const std::vector<BigInt>& matrix, const CrtPlan& plan, int m, uint32_t* out) {
  int E = m * m;
  
  for (int k = 0; k < plan.K; ++k) {
    for (int e = 0; e < E; ++e) {
      out[k * E + e] = matrix[e].mod_u32(plan.mods[k].p);
    }
  }
}

// ============================================================================
// Signed Integer Support
// ============================================================================

BigInt crt_reconstruct_signed(const uint32_t* residues, const CrtPlan& plan) {
  BigInt x = crt_reconstruct(residues, plan);
  
  // If x > M/2, interpret as negative: x - M
  BigInt half_M = plan.M_total;
  // Divide by 2: shift right
  for (size_t i = 0; i < half_M.limbs.size(); ++i) {
    uint32_t carry = (i + 1 < half_M.limbs.size()) ? (half_M.limbs[i + 1] & 1) : 0;
    half_M.limbs[i] = (half_M.limbs[i] >> 1) | (carry << 31);
  }
  half_M.normalize();
  
  if (x.compare_abs(half_M) > 0) {
    x = x - plan.M_total;
  }
  
  return x;
}

void rns_encode_signed(const BigInt& x, const CrtPlan& plan, uint32_t* out) {
  if (x.negative) {
    BigInt pos = plan.M_total + x;
    rns_encode(pos, plan, out);
  } else {
    rns_encode(x, plan, out);
  }
}

}  // namespace rns

#include <stdexcept>
#include <set>
#include <random>

#include "rns/primes.h"
#include "rns/utils.h"

namespace rns {

std::vector<uint32_t> generate_primes(int K, uint32_t seed) {
  if (K <= 0) {
    throw std::runtime_error("K must be positive");
  }
  
  std::vector<uint32_t> primes;
  primes.reserve(K);
  
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(
      (uint32_t)1 << 30,   // Min: 2^30
      ((uint32_t)1 << 31) - 1  // Max: 2^31 - 1
  );
  
  std::set<uint32_t> used;
  
  while ((int)primes.size() < K) {
    uint32_t candidate = dist(rng);
    
    // Make odd
    candidate |= 1;
    
    // Skip if already used
    if (used.count(candidate)) continue;
    
    // Check primality
    if (is_prime(candidate)) {
      primes.push_back(candidate);
      used.insert(candidate);
    }
  }
  
  return primes;
}

std::vector<uint32_t> generate_primes_near(int K, uint32_t target, uint32_t seed) {
  std::vector<uint32_t> primes;
  primes.reserve(K);
  
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> offset_dist(-1000000, 1000000);
  
  std::set<uint32_t> used;
  
  while ((int)primes.size() < K) {
    int32_t offset = offset_dist(rng);
    uint32_t candidate = (uint32_t)((int64_t)target + offset);
    
    // Make odd
    candidate |= 1;
    
    // Bounds check
    if (candidate < 3) continue;
    
    // Skip if already used
    if (used.count(candidate)) continue;
    
    if (is_prime(candidate)) {
      primes.push_back(candidate);
      used.insert(candidate);
    }
  }
  
  return primes;
}

std::vector<uint32_t> get_standard_primes_31bit(int K) {
  // Well-known 31-bit Mersenne-like and other primes
  static const std::vector<uint32_t> standard_primes = {
    2147483647u,  // 2^31 - 1 (Mersenne prime)
    2147483629u,
    2147483587u,
    2147483579u,
    2147483563u,
    2147483549u,
    2147483543u,
    2147483529u,
    2147483521u,
    2147483497u,
    2147483489u,
    2147483477u,
    2147483423u,
    2147483399u,
    2147483353u,
    2147483323u,
    2147483269u,
    2147483249u,
    2147483237u,
    2147483179u,
    2147483171u,
    2147483137u,
    2147483123u,
    2147483077u,
    2147483069u,
    2147483059u,
    2147483053u,
    2147483033u,
    2147483029u,
    2147482951u,
    2147482949u,
    2147482943u,
  };
  
  if (K > (int)standard_primes.size()) {
    // Generate additional primes if needed
    auto extra = generate_primes(K - (int)standard_primes.size(), 54321);
    std::vector<uint32_t> result = standard_primes;
    result.insert(result.end(), extra.begin(), extra.end());
    return std::vector<uint32_t>(result.begin(), result.begin() + K);
  }
  
  return std::vector<uint32_t>(standard_primes.begin(), standard_primes.begin() + K);
}

Modulus32 make_modulus(uint32_t p) {
  Modulus32 mod;
  mod.p = p;
  mod.mu = compute_barrett_mu(p);
  mod.r2 = 0;
  mod.p_inv = 0;
  return mod;
}

PrimeSet make_prime_set(const std::vector<uint32_t>& primes) {
  PrimeSet ps;
  ps.K = (int)primes.size();
  ps.mods.reserve(ps.K);
  
  for (uint32_t p : primes) {
    ps.mods.push_back(make_modulus(p));
  }
  
  return ps;
}

bool verify_primes(const std::vector<uint32_t>& primes) {
  std::set<uint32_t> seen;
  for (uint32_t p : primes) {
    if (!is_prime(p)) return false;
    if (seen.count(p)) return false;  // Duplicate
    seen.insert(p);
  }
  return true;
}

double PrimeSet::bit_capacity() const {
  double bits = 0;
  for (const auto& mod : mods) {
    bits += std::log2(mod.p);
  }
  return bits;
}

}  // namespace rns

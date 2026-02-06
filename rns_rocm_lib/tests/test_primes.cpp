#include <iostream>
#include <set>
#include <vector>
#include "rns/config.h"
#include "rns/modops.h"
#include "rns/barrett.h"

using namespace rns;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    tests_failed++; \
    return; \
  } \
} while(0)

#define TEST_PASS() tests_passed++

bool is_prime_simple(u32 n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if (n % 2 == 0) return false;
  for (u32 i = 3; i * i <= n; i += 2) {
    if (n % i == 0) return false;
  }
  return true;
}

std::vector<u32> generate_test_primes(int K, u64 seed) {
  std::vector<u32> primes;
  u32 candidate = (1u << 30) + 3;  // Start near 2^30
  
  // Simple LCG for determinism
  u64 state = seed;
  auto next_rand = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (u32)(state >> 32);
  };
  
  while ((int)primes.size() < K) {
    candidate = (next_rand() | (1u << 30)) | 1;  // Ensure 31-bit and odd
    if (candidate < (1u << 30)) candidate += (1u << 30);
    
    if (is_prime_simple(candidate)) {
      bool unique = true;
      for (u32 p : primes) {
        if (p == candidate) { unique = false; break; }
      }
      if (unique) primes.push_back(candidate);
    }
  }
  
  return primes;
}

void test_prime_generation() {
  auto primes = generate_test_primes(64, 12345);
  
  TEST_ASSERT(primes.size() == 64, "Generated 64 primes");
  
  // Check all are prime
  for (u32 p : primes) {
    TEST_ASSERT(is_prime_simple(p), "Each value is prime");
  }
  
  // Check all unique
  std::set<u32> unique_set(primes.begin(), primes.end());
  TEST_ASSERT(unique_set.size() == 64, "All primes unique");
  
  TEST_PASS();
  std::cout << "  test_prime_generation: PASS" << std::endl;
}

void test_prime_determinism() {
  auto primes1 = generate_test_primes(32, 99999);
  auto primes2 = generate_test_primes(32, 99999);
  
  TEST_ASSERT(primes1 == primes2, "Same seed produces same primes");
  
  auto primes3 = generate_test_primes(32, 12345);
  TEST_ASSERT(primes1 != primes3, "Different seeds produce different primes");
  
  TEST_PASS();
  std::cout << "  test_prime_determinism: PASS" << std::endl;
}

void test_prime_meta() {
  auto primes = generate_test_primes(8, 54321);
  
  std::vector<PrimeMeta> meta(primes.size());
  for (size_t i = 0; i < primes.size(); ++i) {
    meta[i].p = primes[i];
    meta[i].pad = 0;
    meta[i].mu = compute_barrett_mu(primes[i]);
    meta[i].pinv = 0;  // Future: Montgomery
    meta[i].r2 = 0;
  }
  
  // Verify Barrett mu is correct
  for (size_t i = 0; i < primes.size(); ++i) {
    u32 p = meta[i].p;
    u64 mu = meta[i].mu;
    
    // Test a few multiplications
    for (u32 a = 1; a < 1000; a += 7) {
      for (u32 b = 1; b < 1000; b += 11) {
        u64 prod = (u64)a * b;
        u32 expected = (u32)(prod % p);
        u32 got = barrett_reduce_u64(prod, p, mu);
        TEST_ASSERT(got == expected, "Barrett reduction correct");
      }
    }
  }
  
  TEST_PASS();
  std::cout << "  test_prime_meta: PASS" << std::endl;
}

void test_pairwise_coprime() {
  auto primes = generate_test_primes(16, 11111);
  
  // All distinct primes are pairwise coprime
  for (size_t i = 0; i < primes.size(); ++i) {
    for (size_t j = i + 1; j < primes.size(); ++j) {
      // GCD of two distinct primes is 1
      u32 a = primes[i], b = primes[j];
      while (b != 0) {
        u32 t = b;
        b = a % b;
        a = t;
      }
      TEST_ASSERT(a == 1, "Primes are coprime");
    }
  }
  
  TEST_PASS();
  std::cout << "  test_pairwise_coprime: PASS" << std::endl;
}

int main() {
  std::cout << "=== Prime Tests ===" << std::endl;
  
  test_prime_generation();
  test_prime_determinism();
  test_prime_meta();
  test_pairwise_coprime();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

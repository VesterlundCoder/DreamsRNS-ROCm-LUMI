#include <iostream>
#include <vector>
#include <random>
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

std::vector<PrimeMeta> make_test_primes() {
  std::vector<u32> ps = {
    1000000007, 1000000009, 2147483647, 
    1073741789, 1073741827, 2147483629
  };
  
  std::vector<PrimeMeta> meta(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) {
    meta[i].p = ps[i];
    meta[i].pad = 0;
    meta[i].mu = compute_barrett_mu(ps[i]);
    meta[i].pinv = 0;
    meta[i].r2 = 0;
  }
  return meta;
}

void test_add_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    
    TEST_ASSERT(add_mod(0, 0, p) == 0, "0+0");
    TEST_ASSERT(add_mod(1, 1, p) == 2, "1+1");
    TEST_ASSERT(add_mod(p-1, 1, p) == 0, "(p-1)+1");
    TEST_ASSERT(add_mod(p-1, p-1, p) == p-2, "(p-1)+(p-1)");
  }
  
  TEST_PASS();
  std::cout << "  test_add_mod: PASS" << std::endl;
}

void test_sub_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    
    TEST_ASSERT(sub_mod(5, 3, p) == 2, "5-3");
    TEST_ASSERT(sub_mod(0, 0, p) == 0, "0-0");
    TEST_ASSERT(sub_mod(0, 1, p) == p-1, "0-1");
    TEST_ASSERT(sub_mod(p-1, p-1, p) == 0, "(p-1)-(p-1)");
  }
  
  TEST_PASS();
  std::cout << "  test_sub_mod: PASS" << std::endl;
}

void test_mul_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    u64 mu = pm.mu;
    
    TEST_ASSERT(mul_mod(0, 100, p, mu) == 0, "0*x");
    TEST_ASSERT(mul_mod(1, 100, p, mu) == 100, "1*x");
    TEST_ASSERT(mul_mod(2, 3, p, mu) == 6, "2*3");
    TEST_ASSERT(mul_mod(p-1, 2, p, mu) == p-2, "(p-1)*2");
    TEST_ASSERT(mul_mod(p-1, p-1, p, mu) == 1, "(p-1)*(p-1)");
  }
  
  TEST_PASS();
  std::cout << "  test_mul_mod: PASS" << std::endl;
}

void test_mul_mod_random() {
  auto primes = make_test_primes();
  std::mt19937 rng(12345);
  
  int n_tests = 100000;
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    u64 mu = pm.mu;
    std::uniform_int_distribution<u32> dist(0, p-1);
    
    for (int i = 0; i < n_tests; ++i) {
      u32 a = dist(rng);
      u32 b = dist(rng);
      
      u64 prod = (u64)a * b;
      u32 expected = (u32)(prod % p);
      u32 got = mul_mod(a, b, p, mu);
      
      if (got != expected) {
        std::cerr << "FAIL: mul_mod(" << a << ", " << b << ") mod " << p 
                  << " = " << got << ", expected " << expected << std::endl;
        tests_failed++;
        return;
      }
    }
  }
  
  TEST_PASS();
  std::cout << "  test_mul_mod_random: PASS (" << n_tests << " tests per prime)" << std::endl;
}

void test_fma_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    u64 mu = pm.mu;
    
    TEST_ASSERT(fma_mod(2, 3, 4, p, mu) == 10, "2*3+4");
    TEST_ASSERT(fma_mod(0, 100, 50, p, mu) == 50, "0*x+c");
    
    // Large values
    u32 a = p - 2, b = p - 3, c = p - 1;
    u64 expected = (((u64)a * b) + c) % p;
    TEST_ASSERT(fma_mod(a, b, c, p, mu) == (u32)expected, "large fma");
  }
  
  TEST_PASS();
  std::cout << "  test_fma_mod: PASS" << std::endl;
}

void test_inv_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    u64 mu = pm.mu;
    
    for (u32 a : {2u, 3u, 7u, 123u, 456789u, p-1}) {
      u32 inv = inv_mod(a, p, mu);
      u32 product = mul_mod(a, inv, p, mu);
      TEST_ASSERT(product == 1, "a * inv(a) = 1");
    }
    
    TEST_ASSERT(inv_mod(0, p, mu) == 0, "inv(0) = 0");
  }
  
  TEST_PASS();
  std::cout << "  test_inv_mod: PASS" << std::endl;
}

void test_pow_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    u64 mu = pm.mu;
    
    TEST_ASSERT(pow_mod(2, 0, p, mu) == 1, "x^0 = 1");
    TEST_ASSERT(pow_mod(2, 1, p, mu) == 2, "x^1 = x");
    TEST_ASSERT(pow_mod(2, 10, p, mu) == 1024, "2^10");
    
    // Fermat's little theorem: a^(p-1) = 1 mod p
    TEST_ASSERT(pow_mod(2, p-1, p, mu) == 1, "Fermat");
    TEST_ASSERT(pow_mod(7, p-1, p, mu) == 1, "Fermat");
  }
  
  TEST_PASS();
  std::cout << "  test_pow_mod: PASS" << std::endl;
}

void test_neg_mod() {
  auto primes = make_test_primes();
  
  for (const auto& pm : primes) {
    u32 p = pm.p;
    
    TEST_ASSERT(neg_mod(0, p) == 0, "-0 = 0");
    TEST_ASSERT(neg_mod(1, p) == p-1, "-1 = p-1");
    TEST_ASSERT(add_mod(5, neg_mod(5, p), p) == 0, "x + (-x) = 0");
  }
  
  TEST_PASS();
  std::cout << "  test_neg_mod: PASS" << std::endl;
}

int main() {
  std::cout << "=== ModOps Tests ===" << std::endl;
  
  test_add_mod();
  test_sub_mod();
  test_mul_mod();
  test_mul_mod_random();
  test_fma_mod();
  test_inv_mod();
  test_pow_mod();
  test_neg_mod();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

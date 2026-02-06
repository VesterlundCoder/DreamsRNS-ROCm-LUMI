#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstdint>

#include "rns/rns_ops.h"
#include "rns/primes.h"

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

void test_add_mod_basic() {
  Modulus32 mod = make_modulus(1000000007);
  
  // Basic addition
  TEST_ASSERT(add_mod(5, 3, mod) == 8, "5 + 3 = 8");
  TEST_ASSERT(add_mod(0, 0, mod) == 0, "0 + 0 = 0");
  TEST_ASSERT(add_mod(mod.p - 1, 1, mod) == 0, "(p-1) + 1 = 0 mod p");
  TEST_ASSERT(add_mod(mod.p - 1, mod.p - 1, mod) == mod.p - 2, "(p-1) + (p-1) = p-2 mod p");
  
  TEST_PASS();
  std::cout << "  test_add_mod_basic: PASS" << std::endl;
}

void test_sub_mod_basic() {
  Modulus32 mod = make_modulus(1000000007);
  
  TEST_ASSERT(sub_mod(8, 3, mod) == 5, "8 - 3 = 5");
  TEST_ASSERT(sub_mod(0, 0, mod) == 0, "0 - 0 = 0");
  TEST_ASSERT(sub_mod(0, 1, mod) == mod.p - 1, "0 - 1 = p-1 mod p");
  TEST_ASSERT(sub_mod(5, 5, mod) == 0, "5 - 5 = 0");
  
  TEST_PASS();
  std::cout << "  test_sub_mod_basic: PASS" << std::endl;
}

void test_mul_mod_basic() {
  Modulus32 mod = make_modulus(1000000007);
  
  TEST_ASSERT(mul_mod(5, 3, mod) == 15, "5 * 3 = 15");
  TEST_ASSERT(mul_mod(0, 12345, mod) == 0, "0 * x = 0");
  TEST_ASSERT(mul_mod(1, 12345, mod) == 12345, "1 * x = x");
  TEST_ASSERT(mul_mod(mod.p - 1, 2, mod) == mod.p - 2, "(p-1) * 2 = p-2 mod p");
  
  // Large numbers
  uint32_t a = 123456789;
  uint32_t b = 987654321 % mod.p;
  uint64_t expected = ((uint64_t)a * b) % mod.p;
  TEST_ASSERT(mul_mod(a, b, mod) == (uint32_t)expected, "Large multiplication");
  
  TEST_PASS();
  std::cout << "  test_mul_mod_basic: PASS" << std::endl;
}

void test_fma_mod_basic() {
  Modulus32 mod = make_modulus(1000000007);
  
  // a*b + c
  TEST_ASSERT(fma_mod(5, 3, 2, mod) == 17, "5*3 + 2 = 17");
  TEST_ASSERT(fma_mod(0, 100, 50, mod) == 50, "0*100 + 50 = 50");
  TEST_ASSERT(fma_mod(mod.p - 1, mod.p - 1, 1, mod) == 2, "(p-1)*(p-1) + 1 = 2");
  
  TEST_PASS();
  std::cout << "  test_fma_mod_basic: PASS" << std::endl;
}

void test_inv_mod() {
  Modulus32 mod = make_modulus(1000000007);
  
  // Test inverse: a * inv(a) = 1
  for (uint32_t a : {2u, 3u, 7u, 123u, 456789u}) {
    uint32_t inv = inv_mod(a, mod);
    uint32_t product = mul_mod(a, inv, mod);
    TEST_ASSERT(product == 1, "a * inv(a) = 1");
  }
  
  // inv(0) should return 0 (not invertible)
  TEST_ASSERT(inv_mod(0, mod) == 0, "inv(0) = 0");
  
  TEST_PASS();
  std::cout << "  test_inv_mod: PASS" << std::endl;
}

void test_pow_mod() {
  Modulus32 mod = make_modulus(1000000007);
  
  TEST_ASSERT(pow_mod(2, 0, mod) == 1, "2^0 = 1");
  TEST_ASSERT(pow_mod(2, 1, mod) == 2, "2^1 = 2");
  TEST_ASSERT(pow_mod(2, 10, mod) == 1024, "2^10 = 1024");
  TEST_ASSERT(pow_mod(3, 20, mod) == 3486784401u % mod.p, "3^20");
  
  // Fermat: a^(p-1) = 1 mod p
  TEST_ASSERT(pow_mod(2, mod.p - 1, mod) == 1, "Fermat's little theorem");
  
  TEST_PASS();
  std::cout << "  test_pow_mod: PASS" << std::endl;
}

void test_barrett_random() {
  auto primes = get_standard_primes_31bit(8);
  std::mt19937 rng(12345);
  
  for (uint32_t p : primes) {
    Modulus32 mod = make_modulus(p);
    std::uniform_int_distribution<uint32_t> dist(0, p - 1);
    
    for (int i = 0; i < 1000; ++i) {
      uint32_t a = dist(rng);
      uint32_t b = dist(rng);
      
      // Verify mul_mod against naive
      uint64_t prod = (uint64_t)a * b;
      uint32_t expected = (uint32_t)(prod % p);
      uint32_t got = mul_mod(a, b, mod);
      
      if (got != expected) {
        std::cerr << "FAIL: Barrett reduction mismatch for p=" << p 
                  << " a=" << a << " b=" << b << std::endl;
        tests_failed++;
        return;
      }
    }
  }
  
  TEST_PASS();
  std::cout << "  test_barrett_random: PASS (8 primes, 1000 tests each)" << std::endl;
}

void test_boundary_values() {
  auto primes = get_standard_primes_31bit(4);
  
  for (uint32_t p : primes) {
    Modulus32 mod = make_modulus(p);
    
    // Boundary tests
    TEST_ASSERT(add_mod(0, 0, mod) == 0, "0+0");
    TEST_ASSERT(add_mod(p - 1, 0, mod) == p - 1, "(p-1)+0");
    TEST_ASSERT(add_mod(p - 1, 1, mod) == 0, "(p-1)+1 = 0");
    
    TEST_ASSERT(mul_mod(0, p - 1, mod) == 0, "0*(p-1)");
    TEST_ASSERT(mul_mod(1, p - 1, mod) == p - 1, "1*(p-1)");
    TEST_ASSERT(mul_mod(p - 1, p - 1, mod) == 1, "(p-1)*(p-1) = 1");
  }
  
  TEST_PASS();
  std::cout << "  test_boundary_values: PASS" << std::endl;
}

int main() {
  std::cout << "=== RNS Ops Tests ===" << std::endl;
  
  test_add_mod_basic();
  test_sub_mod_basic();
  test_mul_mod_basic();
  test_fma_mod_basic();
  test_inv_mod();
  test_pow_mod();
  test_barrett_random();
  test_boundary_values();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

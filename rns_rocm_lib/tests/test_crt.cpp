#include <iostream>
#include <vector>
#include <random>
#include <cassert>

#include "rns/crt.h"
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

void test_bigint_basic() {
  BigInt a(123);
  BigInt b(456);
  
  BigInt sum = a + b;
  TEST_ASSERT(sum == BigInt(579), "123 + 456 = 579");
  
  BigInt diff = b - a;
  TEST_ASSERT(diff == BigInt(333), "456 - 123 = 333");
  
  BigInt prod = a * b;
  TEST_ASSERT(prod == BigInt(56088), "123 * 456 = 56088");
  
  TEST_PASS();
  std::cout << "  test_bigint_basic: PASS" << std::endl;
}

void test_bigint_large() {
  // Test with numbers larger than 32 bits
  BigInt a(0xFFFFFFFFULL);
  BigInt b(0xFFFFFFFFULL);
  
  BigInt prod = a * b;
  // 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001
  TEST_ASSERT(prod.limbs.size() == 2, "Product has 2 limbs");
  TEST_ASSERT(prod.limbs[0] == 1, "Low limb correct");
  TEST_ASSERT(prod.limbs[1] == 0xFFFFFFFE, "High limb correct");
  
  TEST_PASS();
  std::cout << "  test_bigint_large: PASS" << std::endl;
}

void test_bigint_mod() {
  BigInt x(1000000);
  uint32_t p = 997;
  uint32_t result = x.mod_u32(p);
  TEST_ASSERT(result == (1000000 % 997), "mod_u32");
  
  TEST_PASS();
  std::cout << "  test_bigint_mod: PASS" << std::endl;
}

void test_crt_plan_creation() {
  auto primes = get_standard_primes_31bit(4);
  CrtPlan plan = CrtPlan::create(primes);
  
  TEST_ASSERT(plan.K == 4, "K = 4");
  TEST_ASSERT(plan.verify(), "Plan verification");
  TEST_ASSERT(plan.bit_capacity > 100, "Bit capacity > 100");
  
  TEST_PASS();
  std::cout << "  test_crt_plan_creation: PASS" << std::endl;
}

void test_crt_roundtrip_small() {
  auto primes = get_standard_primes_31bit(4);
  CrtPlan plan = CrtPlan::create(primes);
  
  // Test small values
  for (uint64_t x : {0ULL, 1ULL, 123ULL, 999999ULL}) {
    std::vector<uint32_t> residues(plan.K);
    rns_encode_u64(x, plan, residues.data());
    
    BigInt reconstructed = crt_reconstruct(residues.data(), plan);
    TEST_ASSERT(reconstructed == BigInt(x), "Roundtrip for " + std::to_string(x));
  }
  
  TEST_PASS();
  std::cout << "  test_crt_roundtrip_small: PASS" << std::endl;
}

void test_crt_roundtrip_random() {
  auto primes = get_standard_primes_31bit(8);
  CrtPlan plan = CrtPlan::create(primes);
  
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 60) - 1);
  
  for (int i = 0; i < 100; ++i) {
    uint64_t x = dist(rng);
    
    std::vector<uint32_t> residues(plan.K);
    rns_encode_u64(x, plan, residues.data());
    
    // Verify residues
    for (int k = 0; k < plan.K; ++k) {
      uint32_t expected = (uint32_t)(x % primes[k]);
      if (residues[k] != expected) {
        std::cerr << "FAIL: Encoding mismatch at k=" << k << std::endl;
        tests_failed++;
        return;
      }
    }
    
    BigInt reconstructed = crt_reconstruct(residues.data(), plan);
    if (reconstructed != BigInt(x)) {
      std::cerr << "FAIL: Reconstruction mismatch for x=" << x << std::endl;
      tests_failed++;
      return;
    }
  }
  
  TEST_PASS();
  std::cout << "  test_crt_roundtrip_random: PASS (100 random u64)" << std::endl;
}

void test_crt_bigint_roundtrip() {
  auto primes = get_standard_primes_31bit(8);
  CrtPlan plan = CrtPlan::create(primes);
  
  // Create a large BigInt
  BigInt x = BigInt::from_string("123456789012345678901234567890");
  
  std::vector<uint32_t> residues(plan.K);
  rns_encode(x, plan, residues.data());
  
  BigInt reconstructed = crt_reconstruct(residues.data(), plan);
  BigInt expected = x % plan.M_total;
  
  TEST_ASSERT(reconstructed == expected, "BigInt roundtrip (mod M)");
  
  TEST_PASS();
  std::cout << "  test_crt_bigint_roundtrip: PASS" << std::endl;
}

void test_crt_signed() {
  auto primes = get_standard_primes_31bit(4);
  CrtPlan plan = CrtPlan::create(primes);
  
  // Test negative values
  BigInt neg_one(-1);
  neg_one.negative = true;
  neg_one.limbs = {1};
  
  std::vector<uint32_t> residues(plan.K);
  rns_encode_signed(neg_one, plan, residues.data());
  
  BigInt reconstructed = crt_reconstruct_signed(residues.data(), plan);
  TEST_ASSERT(reconstructed.negative == true, "Negative flag preserved");
  TEST_ASSERT(reconstructed.limbs.size() == 1 && reconstructed.limbs[0] == 1, "Value is 1");
  
  TEST_PASS();
  std::cout << "  test_crt_signed: PASS" << std::endl;
}

int main() {
  std::cout << "=== CRT Tests ===" << std::endl;
  
  test_bigint_basic();
  test_bigint_large();
  test_bigint_mod();
  test_crt_plan_creation();
  test_crt_roundtrip_small();
  test_crt_roundtrip_random();
  test_crt_bigint_roundtrip();
  test_crt_signed();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

#include <iostream>
#include <vector>
#include <random>

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

void test_matrix_roundtrip_small() {
  auto primes = get_standard_primes_31bit(8);
  CrtPlan plan = CrtPlan::create(primes);
  
  int m = 4;
  int E = m * m;
  
  // Create a simple matrix
  std::vector<BigInt> matrix(E);
  for (int i = 0; i < E; ++i) {
    matrix[i] = BigInt((uint64_t)(i + 1) * 1000);
  }
  
  // Encode to RNS
  std::vector<uint32_t> residues(plan.K * E);
  rns_encode_matrix(matrix, plan, m, residues.data());
  
  // Verify encoding
  for (int k = 0; k < plan.K; ++k) {
    for (int e = 0; e < E; ++e) {
      uint32_t expected = matrix[e].mod_u32(primes[k]);
      uint32_t got = residues[k * E + e];
      TEST_ASSERT(got == expected, "Encoding verification");
    }
  }
  
  // Decode
  auto decoded = crt_reconstruct_matrix(residues.data(), plan, m);
  
  // Verify
  for (int e = 0; e < E; ++e) {
    TEST_ASSERT(decoded[e] == matrix[e], "Matrix element roundtrip");
  }
  
  TEST_PASS();
  std::cout << "  test_matrix_roundtrip_small: PASS" << std::endl;
}

void test_matrix_roundtrip_random() {
  auto primes = get_standard_primes_31bit(16);
  CrtPlan plan = CrtPlan::create(primes);
  
  std::mt19937_64 rng(12345);
  
  for (int m : {4, 6, 8}) {
    int E = m * m;
    
    // Create random matrix with values up to 2^48
    std::vector<BigInt> matrix(E);
    std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 48) - 1);
    
    for (int e = 0; e < E; ++e) {
      matrix[e] = BigInt(dist(rng));
    }
    
    // Encode
    std::vector<uint32_t> residues(plan.K * E);
    rns_encode_matrix(matrix, plan, m, residues.data());
    
    // Decode
    auto decoded = crt_reconstruct_matrix(residues.data(), plan, m);
    
    // Verify
    for (int e = 0; e < E; ++e) {
      if (decoded[e] != matrix[e]) {
        std::cerr << "FAIL: Matrix element mismatch at e=" << e 
                  << " for m=" << m << std::endl;
        tests_failed++;
        return;
      }
    }
  }
  
  TEST_PASS();
  std::cout << "  test_matrix_roundtrip_random: PASS (m=4,6,8)" << std::endl;
}

void test_batch_reconstruction() {
  auto primes = get_standard_primes_31bit(8);
  CrtPlan plan = CrtPlan::create(primes);
  
  int N = 100;
  
  // Create random values
  std::mt19937_64 rng(99999);
  std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 50));
  
  std::vector<uint64_t> values(N);
  for (int i = 0; i < N; ++i) {
    values[i] = dist(rng);
  }
  
  // Encode all
  std::vector<uint32_t> residues(plan.K * N);
  for (int k = 0; k < plan.K; ++k) {
    for (int i = 0; i < N; ++i) {
      residues[k * N + i] = (uint32_t)(values[i] % primes[k]);
    }
  }
  
  // Batch decode
  auto decoded = crt_reconstruct_batch(residues.data(), plan, N);
  
  // Verify
  for (int i = 0; i < N; ++i) {
    if (decoded[i] != BigInt(values[i])) {
      std::cerr << "FAIL: Batch reconstruction mismatch at i=" << i << std::endl;
      tests_failed++;
      return;
    }
  }
  
  TEST_PASS();
  std::cout << "  test_batch_reconstruction: PASS (N=100)" << std::endl;
}

void test_identity_matrix() {
  auto primes = get_standard_primes_31bit(8);
  CrtPlan plan = CrtPlan::create(primes);
  
  int m = 4;
  int E = m * m;
  
  // Create identity matrix
  std::vector<BigInt> identity(E);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      identity[i * m + j] = BigInt(i == j ? 1 : 0);
    }
  }
  
  // Encode
  std::vector<uint32_t> residues(plan.K * E);
  rns_encode_matrix(identity, plan, m, residues.data());
  
  // Verify all residues
  for (int k = 0; k < plan.K; ++k) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        uint32_t expected = (i == j) ? 1 : 0;
        uint32_t got = residues[k * E + i * m + j];
        TEST_ASSERT(got == expected, "Identity element");
      }
    }
  }
  
  // Decode
  auto decoded = crt_reconstruct_matrix(residues.data(), plan, m);
  
  for (int e = 0; e < E; ++e) {
    TEST_ASSERT(decoded[e] == identity[e], "Identity roundtrip");
  }
  
  TEST_PASS();
  std::cout << "  test_identity_matrix: PASS" << std::endl;
}

int main() {
  std::cout << "=== CRT Matrix Tests ===" << std::endl;
  
  test_matrix_roundtrip_small();
  test_matrix_roundtrip_random();
  test_batch_reconstruction();
  test_identity_matrix();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

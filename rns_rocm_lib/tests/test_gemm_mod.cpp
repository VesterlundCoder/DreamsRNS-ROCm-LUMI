#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "rns/rns_api.h"
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

void test_gemm_cpu_reference(int m) {
  auto primes = get_standard_primes_31bit(4);
  std::vector<Modulus32> mods;
  for (auto p : primes) {
    mods.push_back(make_modulus(p));
  }
  
  int K = (int)primes.size();
  int B = 2;  // 2 batches
  int E = m * m;
  int N = B * E;
  
  std::mt19937 rng(12345 + m);
  
  std::vector<uint32_t> A(K * N), Bmat(K * N), C(K * N);
  
  // Initialize random
  for (int k = 0; k < K; ++k) {
    std::uniform_int_distribution<uint32_t> dist(0, primes[k] - 1);
    for (int i = 0; i < N; ++i) {
      A[k * N + i] = dist(rng);
      Bmat[k * N + i] = dist(rng);
    }
  }
  
  // CPU reference
  cpu_gemm_mod(C.data(), A.data(), Bmat.data(), mods, K, B, m);
  
  // Verify against naive
  for (int k = 0; k < K; ++k) {
    Modulus32 mod = mods[k];
    
    for (int b = 0; b < B; ++b) {
      int base = k * N + b * E;
      
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
          uint32_t expected = 0;
          for (int t = 0; t < m; ++t) {
            uint64_t prod = (uint64_t)A[base + i * m + t] * Bmat[base + t * m + j];
            expected = (uint32_t)((expected + prod) % mod.p);
          }
          
          uint32_t got = C[base + i * m + j];
          if (got != expected) {
            std::cerr << "FAIL: GEMM mismatch m=" << m 
                      << " k=" << k << " b=" << b 
                      << " i=" << i << " j=" << j << std::endl;
            tests_failed++;
            return;
          }
        }
      }
    }
  }
  
  TEST_PASS();
  std::cout << "  test_gemm_cpu_reference(m=" << m << "): PASS" << std::endl;
}

#ifdef RNS_HAS_GPU
void test_gemm_gpu(int m) {
  auto primes = get_standard_primes_31bit(8);
  auto ctx = create_context(primes);
  
  int K = ctx.K;
  int B = 64;
  int E = m * m;
  int N = B * E;
  size_t total = (size_t)K * N;
  
  std::mt19937 rng(54321 + m);
  
  std::vector<uint32_t> hA(total), hB(total), hC(total), hC_ref(total);
  
  // Initialize random
  for (int k = 0; k < K; ++k) {
    std::uniform_int_distribution<uint32_t> dist(0, primes[k] - 1);
    for (int i = 0; i < N; ++i) {
      hA[k * N + i] = dist(rng);
      hB[k * N + i] = dist(rng);
    }
  }
  
  // CPU reference
  std::vector<Modulus32> mods;
  for (auto p : primes) mods.push_back(make_modulus(p));
  cpu_gemm_mod(hC_ref.data(), hA.data(), hB.data(), mods, K, B, m);
  
  // GPU
  uint32_t* dA = device_alloc_u32(total);
  uint32_t* dB = device_alloc_u32(total);
  uint32_t* dC = device_alloc_u32(total);
  
  h2d_u32(dA, hA.data(), total);
  h2d_u32(dB, hB.data(), total);
  
  gemm_mod(ctx, dC, dA, dB, B, m);
  device_sync();
  
  d2h_u32(hC.data(), dC, total);
  
  // Compare
  int mismatches = 0;
  for (size_t i = 0; i < total; ++i) {
    if (hC[i] != hC_ref[i]) {
      if (mismatches < 5) {
        std::cerr << "Mismatch at i=" << i << ": got=" << hC[i] 
                  << " expected=" << hC_ref[i] << std::endl;
      }
      mismatches++;
    }
  }
  
  device_free(dA);
  device_free(dB);
  device_free(dC);
  destroy_context(ctx);
  
  TEST_ASSERT(mismatches == 0, "GPU GEMM matches CPU reference");
  
  TEST_PASS();
  std::cout << "  test_gemm_gpu(m=" << m << "): PASS" << std::endl;
}
#endif

int main() {
  std::cout << "=== GEMM Mod Tests ===" << std::endl;
  
  // CPU reference tests
  for (int m : {4, 6, 8, 10}) {
    test_gemm_cpu_reference(m);
  }
  
  // Also test a non-specialized size
  test_gemm_cpu_reference(5);
  test_gemm_cpu_reference(7);
  
#ifdef RNS_HAS_GPU
  std::cout << "\n--- GPU Tests ---" << std::endl;
  for (int m : {4, 6, 8, 10}) {
    test_gemm_gpu(m);
  }
  test_gemm_gpu(7);  // Generic fallback
#else
  std::cout << "\n(GPU tests skipped - no GPU support)" << std::endl;
#endif
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

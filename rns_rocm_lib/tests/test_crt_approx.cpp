#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include "rns/crt_approx.h"
#include "rns/modops.h"
#include "rns/barrett.h"

using namespace rns;

std::vector<PrimeMeta> generate_test_primes(int K, u64 seed) {
  std::vector<PrimeMeta> pm(K);
  std::mt19937_64 rng(seed);
  
  auto is_prime = [](u32 n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (u32 i = 3; i * i <= n; i += 2) {
      if (n % i == 0) return false;
    }
    return true;
  };
  
  std::vector<u32> primes;
  while ((int)primes.size() < K) {
    u32 candidate = (rng() % ((1u << 31) - (1u << 30))) + (1u << 30);
    candidate |= 1;
    if (is_prime(candidate)) {
      bool unique = true;
      for (u32 p : primes) if (p == candidate) unique = false;
      if (unique) primes.push_back(candidate);
    }
  }
  
  for (int i = 0; i < K; ++i) {
    pm[i].p = primes[i];
    pm[i].mu = compute_barrett_mu(primes[i]);
  }
  
  return pm;
}

int main() {
  std::cout << "=== K_small CRT Approximation Tests ===" << std::endl;
  
  int K_small = 3;
  auto pm = generate_test_primes(K_small, 12345);
  
  std::cout << "Using " << K_small << " primes:" << std::endl;
  for (int i = 0; i < K_small; ++i) {
    std::cout << "  p[" << i << "] = " << pm[i].p << std::endl;
  }
  
  auto plan = create_crt_approx_plan(pm.data(), K_small);
  std::cout << "Plan created, log_M_total = " << plan.log_M_total << std::endl;
  
  // Test 1: Small known value
  std::cout << "\nTest 1: Small value (x = 12345)" << std::endl;
  {
    u64 x = 12345;
    std::vector<u32> residues(K_small);
    for (int i = 0; i < K_small; ++i) {
      residues[i] = x % pm[i].p;
    }
    
    double log_mag = crt_approx_log_magnitude(residues.data(), plan);
    double approx_x = std::exp(log_mag);
    double error = std::abs(approx_x - x) / x;
    
    std::cout << "  Actual: " << x << std::endl;
    std::cout << "  Approx: " << approx_x << std::endl;
    std::cout << "  Relative error: " << error << std::endl;
    
    if (error < 1e-6) {
      std::cout << "  PASSED" << std::endl;
    } else {
      std::cout << "  FAILED (error too large)" << std::endl;
    }
  }
  
  // Test 2: Larger value
  std::cout << "\nTest 2: Larger value (x = 2^40)" << std::endl;
  {
    u64 x = 1ULL << 40;
    std::vector<u32> residues(K_small);
    for (int i = 0; i < K_small; ++i) {
      residues[i] = x % pm[i].p;
    }
    
    double log_mag = crt_approx_log_magnitude(residues.data(), plan);
    double expected_log = std::log((double)x);
    double error = std::abs(log_mag - expected_log) / expected_log;
    
    std::cout << "  Actual log: " << expected_log << std::endl;
    std::cout << "  Approx log: " << log_mag << std::endl;
    std::cout << "  Relative error: " << error << std::endl;
    
    if (error < 1e-6) {
      std::cout << "  PASSED" << std::endl;
    } else {
      std::cout << "  FAILED" << std::endl;
    }
  }
  
  // Test 3: Ratio computation
  std::cout << "\nTest 3: Ratio (a=1000, b=100, ratio=10)" << std::endl;
  {
    u64 a = 1000, b = 100;
    std::vector<u32> residues_a(K_small), residues_b(K_small);
    for (int i = 0; i < K_small; ++i) {
      residues_a[i] = a % pm[i].p;
      residues_b[i] = b % pm[i].p;
    }
    
    double ratio = crt_approx_ratio(residues_a.data(), residues_b.data(), plan);
    double expected = 10.0;
    double error = std::abs(ratio - expected) / expected;
    
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Computed: " << ratio << std::endl;
    std::cout << "  Relative error: " << error << std::endl;
    
    if (error < 1e-6) {
      std::cout << "  PASSED" << std::endl;
    } else {
      std::cout << "  FAILED" << std::endl;
    }
  }
  
  // Test 4: Batch scoring
  std::cout << "\nTest 4: Batch scoring" << std::endl;
  {
    int B = 10;
    int m = 2;
    int E = m * m;
    
    // Create test matrices (identity * scalar)
    std::vector<u32> P_residues(K_small * B * E);
    std::vector<double> expected_est(B);
    
    for (int b = 0; b < B; ++b) {
      u64 scale = (b + 1) * 100;  // 100, 200, ..., 1000
      
      // Identity matrix * scale
      // [[scale, 0], [0, scale]]
      for (int k = 0; k < K_small; ++k) {
        u32 s_mod = scale % pm[k].p;
        P_residues[k * B * E + b * E + 0] = s_mod;  // P[0,0]
        P_residues[k * B * E + b * E + 1] = 0;       // P[0,1]
        P_residues[k * B * E + b * E + 2] = 0;       // P[1,0]
        P_residues[k * B * E + b * E + 3] = s_mod;  // P[1,1]
      }
      
      // Expected Frobenius norm: sqrt(2 * scale^2) = scale * sqrt(2)
      expected_est[b] = scale * std::sqrt(2.0);
    }
    
    std::vector<float> delta(B), est(B);
    crt_approx_score_batch(P_residues.data(), delta.data(), est.data(),
                           K_small, B, m, plan);
    
    bool all_passed = true;
    for (int b = 0; b < B; ++b) {
      double error = std::abs(est[b] - expected_est[b]) / expected_est[b];
      // delta should be 0 (or inf if P[0,0]=0)
      bool delta_ok = (delta[b] == 0.0f || std::isinf(delta[b]));
      bool est_ok = (error < 0.01);  // 1% tolerance
      
      if (!delta_ok || !est_ok) {
        std::cout << "  b=" << b << ": delta=" << delta[b] 
                  << ", est=" << est[b] << " (expected " << expected_est[b] << ")"
                  << " error=" << error << std::endl;
        all_passed = false;
      }
    }
    
    if (all_passed) {
      std::cout << "  PASSED (all " << B << " batches)" << std::endl;
    } else {
      std::cout << "  FAILED" << std::endl;
    }
  }
  
  // Test 5: Random values
  std::cout << "\nTest 5: Random values" << std::endl;
  {
    std::mt19937 rng(42);
    std::uniform_int_distribution<u64> dist(1, 1ULL << 50);
    
    int n_tests = 100;
    int passed = 0;
    double max_error = 0;
    
    for (int t = 0; t < n_tests; ++t) {
      u64 x = dist(rng);
      std::vector<u32> residues(K_small);
      for (int i = 0; i < K_small; ++i) {
        residues[i] = x % pm[i].p;
      }
      
      double log_mag = crt_approx_log_magnitude(residues.data(), plan);
      double expected_log = std::log((double)x);
      double error = std::abs(log_mag - expected_log) / std::abs(expected_log);
      
      max_error = std::max(max_error, error);
      if (error < 1e-6) passed++;
    }
    
    std::cout << "  Passed: " << passed << "/" << n_tests << std::endl;
    std::cout << "  Max relative error: " << max_error << std::endl;
    
    if (passed == n_tests) {
      std::cout << "  PASSED" << std::endl;
    } else {
      std::cout << "  FAILED" << std::endl;
    }
  }
  
  std::cout << "\n=== All K_small CRT Approximation Tests Complete ===" << std::endl;
  return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "rns/rns_api.h"
#include "rns/primes.h"

using namespace rns;

int main() {
  std::cout << "=== RNS GEMM Demo ===" << std::endl;
  
#ifndef RNS_HAS_GPU
  std::cout << "GPU support not enabled. Exiting." << std::endl;
  return 0;
#else
  
  // Configuration
  int K = 8;       // Number of primes
  int B = 2048;    // Batch size
  int iters = 200; // Timing iterations
  
  auto primes = get_standard_primes_31bit(K);
  auto ctx = create_context(primes);
  
  std::cout << "Primes: " << K << std::endl;
  std::cout << "Batch:  " << B << std::endl;
  std::cout << "Iters:  " << iters << std::endl;
  std::cout << std::endl;
  
  for (int m : {4, 6, 8, 10}) {
    int E = m * m;
    int N = B * E;
    size_t total = (size_t)K * N;
    
    // Allocate
    std::vector<uint32_t> hA(total), hB(total);
    
    std::mt19937 rng(12345 + m);
    for (int k = 0; k < K; ++k) {
      std::uniform_int_distribution<uint32_t> dist(0, primes[k] - 1);
      for (int i = 0; i < N; ++i) {
        hA[k * N + i] = dist(rng);
        hB[k * N + i] = dist(rng);
      }
    }
    
    uint32_t* dA = device_alloc_u32(total);
    uint32_t* dB = device_alloc_u32(total);
    uint32_t* dC = device_alloc_u32(total);
    
    h2d_u32(dA, hA.data(), total);
    h2d_u32(dB, hB.data(), total);
    
    // Warmup
    gemm_mod(ctx, dC, dA, dB, B, m);
    device_sync();
    
    // Timed loop
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      gemm_mod(ctx, dC, dA, dB, B, m);
    }
    device_sync();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per = ms / iters;
    
    // Ops estimate: K * B * m^3 * 2 (mul + add per element)
    double ops = (double)K * B * m * m * m * 2.0;
    double gops = (ops / (ms_per * 1e-3)) / 1e9;
    
    std::cout << "m=" << m 
              << " | " << ms_per << " ms/iter"
              << " | ~" << gops << " GOPS"
              << std::endl;
    
    device_free(dA);
    device_free(dB);
    device_free(dC);
  }
  
  destroy_context(ctx);
  std::cout << "\nDone." << std::endl;
  
  return 0;
#endif
}

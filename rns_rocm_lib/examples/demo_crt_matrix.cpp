#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "rns/crt.h"
#include "rns/primes.h"
#include "rns/utils.h"

using namespace rns;

int main() {
  std::cout << "=== CRT Matrix Reconstruction Demo ===" << std::endl;
  
  // Configuration
  int K = 16;  // Number of primes (gives ~496 bits capacity)
  int m = 4;   // Matrix dimension
  
  auto primes = get_standard_primes_31bit(K);
  CrtPlan plan = CrtPlan::create(primes);
  
  std::cout << "Primes: " << K << std::endl;
  std::cout << "Matrix: " << m << "x" << m << std::endl;
  std::cout << "Bit capacity: " << plan.bit_capacity << " bits" << std::endl;
  std::cout << std::endl;
  
  // Create a random matrix with large entries
  int E = m * m;
  std::vector<BigInt> matrix(E);
  
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 60) - 1);
  
  std::cout << "Original matrix:" << std::endl;
  for (int i = 0; i < m; ++i) {
    std::cout << "  [";
    for (int j = 0; j < m; ++j) {
      matrix[i * m + j] = BigInt(dist(rng));
      std::cout << std::setw(20) << matrix[i * m + j].to_string();
      if (j < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;
  
  // Encode to RNS
  std::vector<uint32_t> residues(K * E);
  
  Timer t_encode;
  rns_encode_matrix(matrix, plan, m, residues.data());
  double encode_us = t_encode.elapsed_us();
  
  std::cout << "Encoding time: " << encode_us << " us" << std::endl;
  
  // Show a few residues
  std::cout << "\nResidue samples (element [0,0]):" << std::endl;
  for (int k = 0; k < std::min(K, 4); ++k) {
    std::cout << "  p[" << k << "]=" << primes[k] 
              << " -> r=" << residues[k * E + 0] << std::endl;
  }
  std::cout << std::endl;
  
  // Decode
  Timer t_decode;
  auto decoded = crt_reconstruct_matrix(residues.data(), plan, m);
  double decode_us = t_decode.elapsed_us();
  
  std::cout << "Decoding time: " << decode_us << " us" << std::endl;
  
  // Verify
  std::cout << "\nDecoded matrix:" << std::endl;
  bool match = true;
  for (int i = 0; i < m; ++i) {
    std::cout << "  [";
    for (int j = 0; j < m; ++j) {
      std::cout << std::setw(20) << decoded[i * m + j].to_string();
      if (j < m - 1) std::cout << ", ";
      if (decoded[i * m + j] != matrix[i * m + j]) {
        match = false;
      }
    }
    std::cout << "]" << std::endl;
  }
  
  std::cout << "\nVerification: " << (match ? "PASS" : "FAIL") << std::endl;
  
  // Benchmark batch decoding
  std::cout << "\n--- Batch Benchmark ---" << std::endl;
  int N = 1000;
  std::vector<uint32_t> batch_residues(K * N);
  
  // Fill with random residues
  for (int k = 0; k < K; ++k) {
    std::uniform_int_distribution<uint32_t> rdist(0, primes[k] - 1);
    for (int n = 0; n < N; ++n) {
      batch_residues[k * N + n] = rdist(rng);
    }
  }
  
  Timer t_batch;
  auto batch_decoded = crt_reconstruct_batch(batch_residues.data(), plan, N);
  double batch_ms = t_batch.elapsed_ms();
  
  std::cout << "Decoded " << N << " values in " << batch_ms << " ms" << std::endl;
  std::cout << "Throughput: " << (N / batch_ms * 1000) << " values/sec" << std::endl;
  
  return 0;
}

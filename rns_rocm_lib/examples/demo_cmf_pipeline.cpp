#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include "rns/config.h"
#include "rns/modops.h"
#include "rns/barrett.h"
#include "rns/rns_tensor.h"
#include "rns/rns_eval.h"
#include "rns/rns_walk.h"
#include "rns/topk.h"
#include "rns/io.h"

using namespace rns;

std::vector<PrimeMeta> generate_primes(int K, u64 seed) {
  std::vector<PrimeMeta> meta(K);
  std::vector<u32> primes;
  
  u64 state = seed;
  auto next_rand = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (u32)(state >> 32);
  };
  
  auto is_prime = [](u32 n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (u32 i = 3; i * i <= n; i += 2) {
      if (n % i == 0) return false;
    }
    return true;
  };
  
  while ((int)primes.size() < K) {
    u32 candidate = (next_rand() | (1u << 30)) | 1;
    if (candidate < (1u << 30)) candidate += (1u << 30);
    
    if (is_prime(candidate)) {
      bool unique = true;
      for (u32 p : primes) {
        if (p == candidate) { unique = false; break; }
      }
      if (unique) primes.push_back(candidate);
    }
  }
  
  for (int i = 0; i < K; ++i) {
    meta[i].p = primes[i];
    meta[i].pad = 0;
    meta[i].mu = compute_barrett_mu(primes[i]);
    meta[i].pinv = 0;
    meta[i].r2 = 0;
  }
  
  return meta;
}

int main() {
  std::cout << "=== RNS CMF Pipeline Demo ===" << std::endl;
  
  // Configuration
  int K = 32;           // Number of primes
  int B = 1000;         // Batch size (number of shifts)
  int m = 4;            // Matrix dimension
  int dim = 2;          // Number of x-variables
  int depth = 200;      // Walk depth
  int depth1 = 50;      // First snapshot
  int depth2 = depth;   // Second snapshot
  int Kkeep = 20;       // TopK to keep
  
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Primes (K):   " << K << std::endl;
  std::cout << "  Batch (B):    " << B << std::endl;
  std::cout << "  Matrix (m):   " << m << std::endl;
  std::cout << "  Dimension:    " << dim << std::endl;
  std::cout << "  Walk depth:   " << depth << std::endl;
  std::cout << "  TopK:         " << Kkeep << std::endl;
  std::cout << std::endl;
  
  // Generate primes
  std::cout << "Generating " << K << " primes..." << std::endl;
  auto pm = generate_primes(K, 12345);
  std::cout << "  First prime: " << pm[0].p << std::endl;
  std::cout << "  Last prime:  " << pm[K-1].p << std::endl;
  
  // Build a simple test program
  // Matrix: [[x0+1, x1], [1, x0*x1]]
  int E = m * m;
  std::vector<Instr> instrs;
  std::vector<u32> consts(K * 2);
  for (int k = 0; k < K; ++k) {
    consts[k * 2 + 0] = 1;  // const 0 = 1
    consts[k * 2 + 1] = 0;  // const 1 = 0
  }
  
  // r0 = x[0]
  // r1 = x[1]
  // r2 = 1
  // r3 = 0
  // r4 = x0 + 1
  // r5 = x0 * x1
  instrs.push_back({OP_LOAD_X, 0, 0, 0});
  instrs.push_back({OP_LOAD_X, 1, 1, 0});
  instrs.push_back({OP_LOAD_C, 2, 0, 0});
  instrs.push_back({OP_LOAD_C, 3, 1, 0});
  instrs.push_back({OP_ADD, 4, 0, 2});      // r4 = x0 + 1
  instrs.push_back({OP_MUL, 5, 0, 1});      // r5 = x0 * x1
  
  // 4x4 matrix with pattern:
  // [x0+1, x1,   0,    0   ]
  // [1,    x0*x1, 0,   0   ]
  // [0,    0,    x0+1, x1  ]
  // [0,    0,    1,    x0*x1]
  std::vector<uint16_t> out_reg = {
    4, 1, 3, 3,
    2, 5, 3, 3,
    3, 3, 4, 1,
    3, 3, 2, 5
  };
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = (int)instrs.size();
  prog.n_reg = 6;
  prog.n_const = 2;
  prog.instr = instrs.data();
  prog.const_table = consts.data();
  prog.out_reg = out_reg.data();
  
  // Generate random shifts
  std::cout << "\nGenerating " << B << " random shifts..." << std::endl;
  std::mt19937 rng(42);
  std::uniform_int_distribution<i32> shift_dist(-1000000, 1000000);
  
  std::vector<i32> shifts(B * dim);
  for (int i = 0; i < B * dim; ++i) {
    shifts[i] = shift_dist(rng);
  }
  
  // Direction vector: all ones
  std::vector<i32> dirs(dim, 1);
  
  // Allocate outputs
  std::vector<u32> P_final(B * E);
  std::vector<uint8_t> alive(B);
  std::vector<float> est1(B), est2(B), delta1(B), delta2(B);
  
  WalkConfig cfg;
  cfg.depth = depth;
  cfg.depth1 = depth1;
  cfg.depth2 = depth2;
  cfg.m = m;
  cfg.dim = dim;
  cfg.K = K;
  cfg.B = B;
  
  WalkOutputs out;
  out.P_final = P_final.data();
  out.alive = alive.data();
  out.est1 = est1.data();
  out.est2 = est2.data();
  out.delta1 = delta1.data();
  out.delta2 = delta2.data();
  
  // Run walk
  std::cout << "\nRunning walk (depth=" << depth << ")..." << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  
  walk_fused_cpu(cfg, prog, shifts.data(), dirs.data(), pm.data(), out);
  
  auto t1 = std::chrono::high_resolution_clock::now();
  double walk_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Walk time: " << walk_ms << " ms" << std::endl;
  
  // Count alive
  int n_alive = 0;
  for (int b = 0; b < B; ++b) {
    if (alive[b]) n_alive++;
  }
  std::cout << "  Alive: " << n_alive << "/" << B << std::endl;
  
  // TopK on delta2 (ascending = smallest delta is best)
  std::cout << "\nRunning TopK (K=" << Kkeep << ")..." << std::endl;
  std::vector<TopKItem> topk(Kkeep);
  
  TopKConfig topk_cfg;
  topk_cfg.B = B;
  topk_cfg.Kkeep = Kkeep;
  topk_cfg.ascending = true;
  
  topk_reduce_cpu(delta2.data(), est2.data(), topk.data(), topk_cfg);
  
  // Display top results
  std::cout << "\nTop " << Kkeep << " candidates:" << std::endl;
  std::cout << "  Rank | Shift Idx | Delta        | Est" << std::endl;
  std::cout << "  -----|-----------|--------------|-------------" << std::endl;
  for (int i = 0; i < std::min(10, Kkeep); ++i) {
    printf("  %4d | %9d | %12.6e | %12.6e\n",
           i + 1, topk[i].shift_idx, topk[i].score, topk[i].est);
  }
  
  // Write outputs
  std::cout << "\nWriting outputs..." << std::endl;
  
  try {
    append_hits_jsonl("demo_hits.jsonl", "demo_cmf", topk.data(), Kkeep);
    std::cout << "  Written: demo_hits.jsonl" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "  Warning: " << e.what() << std::endl;
  }
  
  try {
    write_summary_csv("demo_summary.csv", "demo_cmf",
                      topk[0].score, topk[0].shift_idx, topk[0].est);
    std::cout << "  Written: demo_summary.csv" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "  Warning: " << e.what() << std::endl;
  }
  
  // Stats
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Best candidate:" << std::endl;
  std::cout << "  Shift index: " << topk[0].shift_idx << std::endl;
  std::cout << "  Delta:       " << topk[0].score << std::endl;
  std::cout << "  Est:         " << topk[0].est << std::endl;
  
  // Print final matrix for best candidate
  int best_b = topk[0].shift_idx;
  std::cout << "\nFinal P matrix (mod p0=" << pm[0].p << "):" << std::endl;
  for (int i = 0; i < m; ++i) {
    std::cout << "  [";
    for (int j = 0; j < m; ++j) {
      printf("%12u", P_final[best_b * E + i * m + j]);
      if (j < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
  
  std::cout << "\nDemo complete." << std::endl;
  return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include "rns/config.h"
#include "rns/modops.h"
#include "rns/rns_walk.h"
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

std::vector<PrimeMeta> make_test_primes(int K) {
  std::vector<u32> ps = {1000000007, 1000000009, 2147483647, 1073741789};
  std::vector<PrimeMeta> meta(K);
  for (int i = 0; i < K; ++i) {
    meta[i].p = ps[i % ps.size()];
    meta[i].pad = 0;
    meta[i].mu = compute_barrett_mu(meta[i].p);
  }
  return meta;
}

void test_identity_walk() {
  // Walk with identity step matrix should give identity
  int K = 1, B = 2, m = 4, dim = 1;
  int E = m * m;
  int depth = 10;
  
  auto pm = make_test_primes(K);
  u32 p = pm[0].p;
  
  // Program that produces identity matrix
  std::vector<Instr> instrs;
  std::vector<u32> consts(K * 2);
  consts[0] = 1;  // const 0 = 1
  consts[1] = 0;  // const 1 = 0
  
  instrs.push_back({OP_LOAD_C, 0, 0, 0});  // r0 = 1
  instrs.push_back({OP_LOAD_C, 1, 1, 0});  // r1 = 0
  
  // Identity 4x4: 1 on diagonal, 0 elsewhere
  std::vector<uint16_t> out_reg(E);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      out_reg[i * m + j] = (i == j) ? 0 : 1;  // r0=1, r1=0
    }
  }
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = (int)instrs.size();
  prog.n_reg = 2;
  prog.n_const = 2;
  prog.instr = instrs.data();
  prog.const_table = consts.data();
  prog.out_reg = out_reg.data();
  
  WalkConfig cfg;
  cfg.depth = depth;
  cfg.depth1 = 5;
  cfg.depth2 = depth;
  cfg.m = m;
  cfg.dim = dim;
  cfg.K = K;
  cfg.B = B;
  
  std::vector<i32> shifts(B * dim, 0);
  std::vector<i32> dirs(dim, 1);
  
  std::vector<u32> P_final(B * E);
  std::vector<uint8_t> alive(B);
  std::vector<float> est1(B), est2(B), delta1(B), delta2(B);
  
  WalkOutputs out;
  out.P_final = P_final.data();
  out.alive = alive.data();
  out.est1 = est1.data();
  out.est2 = est2.data();
  out.delta1 = delta1.data();
  out.delta2 = delta2.data();
  
  walk_fused_cpu(cfg, prog, shifts.data(), dirs.data(), pm.data(), out);
  
  // Check result is identity (I^depth = I)
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        u32 expected = (i == j) ? 1 : 0;
        u32 got = P_final[b * E + i * m + j];
        TEST_ASSERT(got == expected, "Identity walk result");
      }
    }
    TEST_ASSERT(alive[b] == 1, "Lane alive");
  }
  
  TEST_PASS();
  std::cout << "  test_identity_walk: PASS" << std::endl;
}

void test_constant_matrix_walk() {
  // Walk with constant 2x2 matrix [[2,1],[1,1]] (Fibonacci-like)
  int K = 1, B = 1, m = 2, dim = 1;
  int E = m * m;
  int depth = 5;
  
  auto pm = make_test_primes(K);
  u32 p = pm[0].p;
  u64 mu = pm[0].mu;
  
  // Constants: 2, 1
  std::vector<u32> consts = {2, 1};
  
  std::vector<Instr> instrs = {
    {OP_LOAD_C, 0, 0, 0},  // r0 = 2
    {OP_LOAD_C, 1, 1, 0},  // r1 = 1
  };
  
  // Matrix [[2,1],[1,1]] -> out_reg = {0,1,1,1}
  std::vector<uint16_t> out_reg = {0, 1, 1, 1};
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = (int)instrs.size();
  prog.n_reg = 2;
  prog.n_const = 2;
  prog.instr = instrs.data();
  prog.const_table = consts.data();
  prog.out_reg = out_reg.data();
  
  WalkConfig cfg;
  cfg.depth = depth;
  cfg.depth1 = 3;
  cfg.depth2 = depth;
  cfg.m = m;
  cfg.dim = dim;
  cfg.K = K;
  cfg.B = B;
  
  std::vector<i32> shifts(B * dim, 0);
  std::vector<i32> dirs(dim, 1);
  
  std::vector<u32> P_final(B * E);
  std::vector<uint8_t> alive(B);
  std::vector<float> est1(B), est2(B), delta1(B), delta2(B);
  
  WalkOutputs out;
  out.P_final = P_final.data();
  out.alive = alive.data();
  out.est1 = est1.data();
  out.est2 = est2.data();
  out.delta1 = delta1.data();
  out.delta2 = delta2.data();
  
  walk_fused_cpu(cfg, prog, shifts.data(), dirs.data(), pm.data(), out);
  
  // Compute expected: M^5 manually
  // M = [[2,1],[1,1]]
  // M^2 = [[5,3],[3,2]]
  // M^3 = [[13,8],[8,5]]
  // M^4 = [[34,21],[21,13]]
  // M^5 = [[89,55],[55,34]]
  
  u32 expected[4] = {89, 55, 55, 34};
  for (int e = 0; e < E; ++e) {
    TEST_ASSERT(P_final[e] == expected[e], "M^5 element " + std::to_string(e));
  }
  
  TEST_PASS();
  std::cout << "  test_constant_matrix_walk: PASS" << std::endl;
}

void test_walk_snapshots() {
  // Verify snapshots are taken at correct depths
  int K = 1, B = 2, m = 2, dim = 1;
  int E = m * m;
  int depth = 20;
  
  auto pm = make_test_primes(K);
  
  // Simple 2x identity
  std::vector<u32> consts = {2, 0};
  std::vector<Instr> instrs = {
    {OP_LOAD_C, 0, 0, 0},
    {OP_LOAD_C, 1, 1, 0},
  };
  std::vector<uint16_t> out_reg = {0, 1, 1, 0};  // [[2,0],[0,2]]
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = 2;
  prog.n_reg = 2;
  prog.n_const = 2;
  prog.instr = instrs.data();
  prog.const_table = consts.data();
  prog.out_reg = out_reg.data();
  
  WalkConfig cfg;
  cfg.depth = depth;
  cfg.depth1 = 5;
  cfg.depth2 = depth;
  cfg.m = m;
  cfg.dim = dim;
  cfg.K = K;
  cfg.B = B;
  
  std::vector<i32> shifts(B * dim, 0);
  std::vector<i32> dirs(dim, 1);
  
  std::vector<u32> P_final(B * E);
  std::vector<uint8_t> alive(B);
  std::vector<float> est1(B), est2(B), delta1(B), delta2(B);
  
  WalkOutputs out;
  out.P_final = P_final.data();
  out.alive = alive.data();
  out.est1 = est1.data();
  out.est2 = est2.data();
  out.delta1 = delta1.data();
  out.delta2 = delta2.data();
  
  walk_fused_cpu(cfg, prog, shifts.data(), dirs.data(), pm.data(), out);
  
  // Snapshots should be non-zero
  for (int b = 0; b < B; ++b) {
    TEST_ASSERT(est1[b] > 0, "est1 populated");
    TEST_ASSERT(est2[b] > 0, "est2 populated");
    TEST_ASSERT(alive[b] == 1, "alive");
  }
  
  TEST_PASS();
  std::cout << "  test_walk_snapshots: PASS" << std::endl;
}

int main() {
  std::cout << "=== Walk Tests ===" << std::endl;
  
  test_identity_walk();
  test_constant_matrix_walk();
  test_walk_snapshots();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

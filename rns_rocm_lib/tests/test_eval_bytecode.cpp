#include <iostream>
#include <vector>
#include <cstring>
#include "rns/config.h"
#include "rns/modops.h"
#include "rns/rns_eval.h"
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
    meta[i].pinv = 0;
    meta[i].r2 = 0;
  }
  return meta;
}

void test_simple_load() {
  // Program: out[0] = x[0], out[1] = const[0]
  int K = 2, B = 4, m = 2, dim = 1;
  int E = m * m;
  
  auto pm = make_test_primes(K);
  
  // Instructions
  std::vector<Instr> instrs = {
    {OP_LOAD_X, 0, 0, 0},   // r0 = x[0]
    {OP_LOAD_C, 1, 0, 0},   // r1 = const[0]
  };
  
  // Constants per prime
  std::vector<u32> consts(K * 1);
  for (int k = 0; k < K; ++k) {
    consts[k] = 42;
  }
  
  // Output register map
  std::vector<uint16_t> out_reg = {0, 1, 0, 1};  // 2x2 matrix
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = (int)instrs.size();
  prog.n_reg = 2;
  prog.n_const = 1;
  prog.instr = instrs.data();
  prog.const_table = consts.data();
  prog.out_reg = out_reg.data();
  
  // Input x values
  std::vector<u32> x_vals(K * B * dim);
  for (int k = 0; k < K; ++k) {
    for (int b = 0; b < B; ++b) {
      x_vals[k * B * dim + b * dim + 0] = 100 + b;
    }
  }
  
  // Output
  std::vector<u32> out_matrix(K * B * E, 0);
  std::vector<uint8_t> alive(K * B, 0);
  
  eval_program_to_matrix_cpu(prog, x_vals.data(), out_matrix.data(), alive.data(),
                             pm.data(), K, B);
  
  // Verify
  for (int k = 0; k < K; ++k) {
    for (int b = 0; b < B; ++b) {
      int base = k * B * E + b * E;
      TEST_ASSERT(out_matrix[base + 0] == (u32)(100 + b), "x[0] loaded");
      TEST_ASSERT(out_matrix[base + 1] == 42, "const loaded");
      TEST_ASSERT(alive[k * B + b] == 1, "lane alive");
    }
  }
  
  TEST_PASS();
  std::cout << "  test_simple_load: PASS" << std::endl;
}

void test_arithmetic() {
  // Program: compute x[0] + x[1], x[0] * x[1], x[0]^2
  int K = 2, B = 3, m = 2, dim = 2;
  int E = m * m;
  
  auto pm = make_test_primes(K);
  
  std::vector<Instr> instrs = {
    {OP_LOAD_X, 0, 0, 0},   // r0 = x[0]
    {OP_LOAD_X, 1, 1, 0},   // r1 = x[1]
    {OP_ADD, 2, 0, 1},      // r2 = r0 + r1
    {OP_MUL, 3, 0, 1},      // r3 = r0 * r1
    {OP_POW2, 4, 0, 0},     // r4 = r0^2
    {OP_SUB, 5, 0, 1},      // r5 = r0 - r1
  };
  
  std::vector<u32> consts;  // no constants
  std::vector<uint16_t> out_reg = {2, 3, 4, 5};
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = (int)instrs.size();
  prog.n_reg = 6;
  prog.n_const = 0;
  prog.instr = instrs.data();
  prog.const_table = consts.data();
  prog.out_reg = out_reg.data();
  
  std::vector<u32> x_vals(K * B * dim);
  for (int k = 0; k < K; ++k) {
    u32 p = pm[k].p;
    for (int b = 0; b < B; ++b) {
      x_vals[k * B * dim + b * dim + 0] = (10 + b) % p;
      x_vals[k * B * dim + b * dim + 1] = (5 + b) % p;
    }
  }
  
  std::vector<u32> out_matrix(K * B * E, 0);
  std::vector<uint8_t> alive(K * B, 0);
  
  eval_program_to_matrix_cpu(prog, x_vals.data(), out_matrix.data(), alive.data(),
                             pm.data(), K, B);
  
  // Verify
  for (int k = 0; k < K; ++k) {
    u32 p = pm[k].p;
    u64 mu = pm[k].mu;
    
    for (int b = 0; b < B; ++b) {
      u32 x0 = x_vals[k * B * dim + b * dim + 0];
      u32 x1 = x_vals[k * B * dim + b * dim + 1];
      
      u32 exp_add = add_mod(x0, x1, p);
      u32 exp_mul = mul_mod(x0, x1, p, mu);
      u32 exp_pow2 = mul_mod(x0, x0, p, mu);
      u32 exp_sub = sub_mod(x0, x1, p);
      
      int base = k * B * E + b * E;
      TEST_ASSERT(out_matrix[base + 0] == exp_add, "add");
      TEST_ASSERT(out_matrix[base + 1] == exp_mul, "mul");
      TEST_ASSERT(out_matrix[base + 2] == exp_pow2, "pow2");
      TEST_ASSERT(out_matrix[base + 3] == exp_sub, "sub");
    }
  }
  
  TEST_PASS();
  std::cout << "  test_arithmetic: PASS" << std::endl;
}

void test_inv_alive_mask() {
  // Test that INV of 0 kills the lane
  int K = 1, B = 3, m = 2, dim = 1;
  int E = m * m;
  
  auto pm = make_test_primes(K);
  
  std::vector<Instr> instrs = {
    {OP_LOAD_X, 0, 0, 0},   // r0 = x[0]
    {OP_INV, 1, 0, 0},      // r1 = 1/r0
  };
  
  std::vector<uint16_t> out_reg = {1, 1, 1, 1};
  
  Program prog;
  prog.m = m;
  prog.dim = dim;
  prog.n_instr = (int)instrs.size();
  prog.n_reg = 2;
  prog.n_const = 0;
  prog.instr = instrs.data();
  prog.const_table = nullptr;
  prog.out_reg = out_reg.data();
  
  // x_vals: [5, 0, 7] - batch 1 has x=0
  std::vector<u32> x_vals = {5, 0, 7};
  
  std::vector<u32> out_matrix(K * B * E, 0);
  std::vector<uint8_t> alive(K * B, 0);
  
  eval_program_to_matrix_cpu(prog, x_vals.data(), out_matrix.data(), alive.data(),
                             pm.data(), K, B);
  
  TEST_ASSERT(alive[0] == 1, "batch 0 alive");
  TEST_ASSERT(alive[1] == 0, "batch 1 dead (inv 0)");
  TEST_ASSERT(alive[2] == 1, "batch 2 alive");
  
  TEST_PASS();
  std::cout << "  test_inv_alive_mask: PASS" << std::endl;
}

int main() {
  std::cout << "=== Bytecode Eval Tests ===" << std::endl;
  
  test_simple_load();
  test_arithmetic();
  test_inv_alive_mask();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}

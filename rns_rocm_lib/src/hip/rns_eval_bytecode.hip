#include <hip/hip_runtime.h>
#include "rns/rns_eval.h"

namespace rns {

__global__ void k_eval_program(
    const Instr* instr, int n_instr,
    const u32* const_table, int n_const,
    const uint16_t* out_reg, int E,
    const u32* x_vals, int dim,
    u32* out_matrix,
    uint8_t* alive,
    const PrimeMeta* pm,
    int K, int B)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y;
  
  if (b >= B || k >= K) return;
  
  u32 p = pm[k].p;
  u64 mu = pm[k].mu;
  
  // Local register file
  u32 regs[MAX_REGISTERS];
  uint8_t lane_alive = 1;
  
  // Pointer to x-values for this (k, b)
  const u32* x = x_vals + k * B * dim + b * dim;
  
  // Pointer to constants for this prime k
  const u32* consts = const_table + k * n_const;
  
  // Execute instructions
  for (int i = 0; i < n_instr && lane_alive; ++i) {
    Instr ins = instr[i];
    u32 va = regs[ins.a];
    u32 vb = regs[ins.b];
    u32 result = 0;
    
    switch (ins.op) {
      case OP_NOP:
        break;
      case OP_LOAD_X:
        result = x[ins.a];
        break;
      case OP_LOAD_C:
        result = consts[ins.a];
        break;
      case OP_ADD:
        result = add_mod(va, vb, p);
        break;
      case OP_SUB:
        result = sub_mod(va, vb, p);
        break;
      case OP_MUL:
        result = mul_mod(va, vb, p, mu);
        break;
      case OP_NEG:
        result = neg_mod(va, p);
        break;
      case OP_POW2:
        result = pow2_mod(va, p, mu);
        break;
      case OP_POW3:
        result = pow3_mod(va, p, mu);
        break;
      case OP_INV:
        if (va == 0) {
          lane_alive = 0;
          result = 0;
        } else {
          result = inv_mod(va, p, mu);
        }
        break;
      case OP_MULINV:
        if (vb == 0) {
          lane_alive = 0;
          result = 0;
        } else {
          result = mul_mod(va, inv_mod(vb, p, mu), p, mu);
        }
        break;
      case OP_COPY:
        result = va;
        break;
    }
    regs[ins.dst] = result;
  }
  
  // Write output matrix
  int out_base = k * B * E + b * E;
  for (int e = 0; e < E; ++e) {
    out_matrix[out_base + e] = regs[out_reg[e]];
  }
  
  // Write alive mask
  alive[k * B + b] = lane_alive;
}

void eval_program_to_matrix(
    const Program& prog,
    const u32* x_vals,
    u32* out_matrix,
    uint8_t* alive,
    const PrimeMeta* pm,
    int K, int B)
{
#ifdef RNS_HAS_GPU
  int E = prog.m * prog.m;
  
  int blockSize = 256;
  int gridX = (B + blockSize - 1) / blockSize;
  dim3 grid(gridX, K);
  dim3 block(blockSize);
  
  hipLaunchKernelGGL(k_eval_program, grid, block, 0, 0,
                     prog.instr, prog.n_instr,
                     prog.const_table, prog.n_const,
                     prog.out_reg, E,
                     x_vals, prog.dim,
                     out_matrix, alive, pm, K, B);
#else
  eval_program_to_matrix_cpu(prog, x_vals, out_matrix, alive, pm, K, B);
#endif
}

void eval_program_to_matrix_cpu(
    const Program& prog,
    const u32* x_vals,
    u32* out_matrix,
    uint8_t* alive,
    const PrimeMeta* pm,
    int K, int B)
{
  int E = prog.m * prog.m;
  
  for (int k = 0; k < K; ++k) {
    u32 p = pm[k].p;
    u64 mu = pm[k].mu;
    const u32* consts = prog.const_table + k * prog.n_const;
    
    for (int b = 0; b < B; ++b) {
      u32 regs[MAX_REGISTERS] = {0};
      uint8_t lane_alive = 1;
      
      const u32* x = x_vals + k * B * prog.dim + b * prog.dim;
      
      for (int i = 0; i < prog.n_instr && lane_alive; ++i) {
        Instr ins = prog.instr[i];
        u32 va = regs[ins.a];
        u32 vb = regs[ins.b];
        u32 result = 0;
        
        switch (ins.op) {
          case OP_NOP: break;
          case OP_LOAD_X: result = x[ins.a]; break;
          case OP_LOAD_C: result = consts[ins.a]; break;
          case OP_ADD: result = add_mod(va, vb, p); break;
          case OP_SUB: result = sub_mod(va, vb, p); break;
          case OP_MUL: result = mul_mod(va, vb, p, mu); break;
          case OP_NEG: result = neg_mod(va, p); break;
          case OP_POW2: result = pow2_mod(va, p, mu); break;
          case OP_POW3: result = pow3_mod(va, p, mu); break;
          case OP_INV:
            if (va == 0) { lane_alive = 0; result = 0; }
            else { result = inv_mod(va, p, mu); }
            break;
          case OP_MULINV:
            if (vb == 0) { lane_alive = 0; result = 0; }
            else { result = mul_mod(va, inv_mod(vb, p, mu), p, mu); }
            break;
          case OP_COPY: result = va; break;
        }
        regs[ins.dst] = result;
      }
      
      int out_base = k * B * E + b * E;
      for (int e = 0; e < E; ++e) {
        out_matrix[out_base + e] = regs[prog.out_reg[e]];
      }
      alive[k * B + b] = lane_alive;
    }
  }
}

} // namespace rns

#include "rns/rns_eval.h"
#include <vector>

namespace rns {

#ifndef RNS_HAS_GPU
void eval_program_to_matrix(
    const Program& prog,
    const u32* x_vals,
    u32* out_matrix,
    uint8_t* alive,
    const PrimeMeta* pm,
    int K, int B)
{
  eval_program_to_matrix_cpu(prog, x_vals, out_matrix, alive, pm, K, B);
}
#endif

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

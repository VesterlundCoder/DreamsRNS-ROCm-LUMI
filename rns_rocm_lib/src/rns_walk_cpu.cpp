#include "rns/rns_walk.h"
#include <vector>
#include <cmath>

namespace rns {

#ifndef RNS_HAS_GPU
void walk_fused(
    const WalkConfig& cfg,
    const Program& prog_step,
    const i32* shifts,
    const i32* dirs,
    const PrimeMeta* pm,
    WalkOutputs out)
{
  walk_fused_cpu(cfg, prog_step, shifts, dirs, pm, out);
}
#endif

void walk_fused_cpu(
    const WalkConfig& cfg,
    const Program& prog_step,
    const i32* shifts,
    const i32* dirs,
    const PrimeMeta* pm,
    WalkOutputs out)
{
  int m = cfg.m;
  int E = m * m;
  int dim = cfg.dim;
  
  u32 p0 = pm[0].p;
  u64 mu0 = pm[0].mu;
  const u32* consts = prog_step.const_table;
  
  for (int b = 0; b < cfg.B; ++b) {
    const i32* shift_b = shifts + b * dim;
    
    std::vector<u32> P(E), M(E);
    std::vector<float> Pf(E), Mf(E);
    
    // Initialize P to identity
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        P[i * m + j] = (i == j) ? 1 : 0;
        Pf[i * m + j] = (i == j) ? 1.0f : 0.0f;
      }
    }
    
    uint8_t alive = 1;
    std::vector<u32> x(dim);
    
    for (int t = 0; t < cfg.depth && alive; ++t) {
      // Compute x
      for (int j = 0; j < dim; ++j) {
        i64 val = (i64)shift_b[j] + (i64)t * dirs[j];
        val = ((val % (i64)p0) + p0) % p0;
        x[j] = (u32)val;
      }
      
      // Eval step matrix
      std::vector<u32> regs(MAX_REGISTERS, 0);
      for (int i = 0; i < prog_step.n_instr && alive; ++i) {
        Instr ins = prog_step.instr[i];
        u32 va = regs[ins.a];
        u32 vb = regs[ins.b];
        u32 result = 0;
        
        switch (ins.op) {
          case OP_NOP: break;
          case OP_LOAD_X: result = x[ins.a]; break;
          case OP_LOAD_C: result = consts[ins.a]; break;
          case OP_ADD: result = add_mod(va, vb, p0); break;
          case OP_SUB: result = sub_mod(va, vb, p0); break;
          case OP_MUL: result = mul_mod(va, vb, p0, mu0); break;
          case OP_NEG: result = neg_mod(va, p0); break;
          case OP_POW2: result = pow2_mod(va, p0, mu0); break;
          case OP_POW3: result = pow3_mod(va, p0, mu0); break;
          case OP_INV:
            if (va == 0) alive = 0;
            else result = inv_mod(va, p0, mu0);
            break;
          case OP_MULINV:
            if (vb == 0) alive = 0;
            else result = mul_mod(va, inv_mod(vb, p0, mu0), p0, mu0);
            break;
          case OP_COPY: result = va; break;
        }
        regs[ins.dst] = result;
      }
      
      for (int e = 0; e < E; ++e) {
        M[e] = regs[prog_step.out_reg[e]];
        Mf[e] = (float)M[e];
      }
      
      // P = P @ M
      std::vector<u32> Pnew(E);
      std::vector<float> Pfnew(E);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
          u32 acc = 0;
          float accf = 0.0f;
          for (int tt = 0; tt < m; ++tt) {
            acc = fma_mod(P[i * m + tt], M[tt * m + j], acc, p0, mu0);
            accf += Pf[i * m + tt] * Mf[tt * m + j];
          }
          Pnew[i * m + j] = acc;
          Pfnew[i * m + j] = accf;
        }
      }
      P = Pnew;
      Pf = Pfnew;
      
      // Snapshots
      if (t + 1 == cfg.depth1) {
        float norm = 0.0f;
        for (int e = 0; e < E; ++e) norm += Pf[e] * Pf[e];
        out.est1[b] = sqrtf(norm);
        out.delta1[b] = (Pf[0] != 0.0f) ? fabsf(Pf[1] / Pf[0]) : 1e30f;
      }
      if (t + 1 == cfg.depth2) {
        float norm = 0.0f;
        for (int e = 0; e < E; ++e) norm += Pf[e] * Pf[e];
        out.est2[b] = sqrtf(norm);
        out.delta2[b] = (Pf[0] != 0.0f) ? fabsf(Pf[1] / Pf[0]) : 1e30f;
      }
    }
    
    int base = b * E;
    for (int e = 0; e < E; ++e) {
      out.P_final[base + e] = P[e];
    }
    out.alive[b] = alive;
  }
}

void init_identity_batch(u32* P, int K, int B, int m) {
  int E = m * m;
  for (int k = 0; k < K; ++k) {
    for (int b = 0; b < B; ++b) {
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
          P[k * B * E + b * E + i * m + j] = (i == j) ? 1 : 0;
        }
      }
    }
  }
}

} // namespace rns

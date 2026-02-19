#include <hip/hip_runtime.h>
#include <vector>
#include "rns/rns_walk.h"
#include "rns/rns_matmul.h"

namespace rns {

// Helper: set matrix to identity
__device__ void set_identity(u32* M, int m) {
  int E = m * m;
  for (int i = 0; i < E; ++i) {
    int row = i / m;
    int col = i % m;
    M[i] = (row == col) ? 1 : 0;
  }
}

// Helper: matrix multiply C = A @ B mod p (in-place friendly via temp)
__device__ void matmul_local(
    const u32* A, const u32* B, u32* C,
    int m, u32 p, u64 mu)
{
  u32 temp[100];  // max m=10 -> 100 elements
  int E = m * m;
  
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      u32 acc = 0;
      for (int t = 0; t < m; ++t) {
        acc = fma_mod(A[i * m + t], B[t * m + j], acc, p, mu);
      }
      temp[i * m + j] = acc;
    }
  }
  
  for (int e = 0; e < E; ++e) {
    C[e] = temp[e];
  }
}

// Evaluate x-values: x_j = (shift[j] + t * dir[j]) mod p
__device__ void compute_x_vals(
    const i32* shift, const i32* dir, int dim,
    int t, u32 p, u32* x_out)
{
  for (int j = 0; j < dim; ++j) {
    i64 val = (i64)shift[j] + (i64)t * dir[j];
    val = ((val % (i64)p) + p) % p;
    x_out[j] = (u32)val;
  }
}

// Simplified bytecode eval for walk kernel (uses shared program)
__device__ void eval_step_matrix(
    const Instr* instr, int n_instr,
    const u32* consts, int n_const,
    const uint16_t* out_reg,
    const u32* x, int dim, int m,
    u32 p, u64 mu,
    u32* M_out,
    uint8_t& alive)
{
  u32 regs[MAX_REGISTERS] = {0};
  int E = m * m;
  
  for (int i = 0; i < n_instr && alive; ++i) {
    Instr ins = instr[i];
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
        if (va == 0) { alive = 0; }
        else { result = inv_mod(va, p, mu); }
        break;
      case OP_MULINV:
        if (vb == 0) { alive = 0; }
        else { result = mul_mod(va, inv_mod(vb, p, mu), p, mu); }
        break;
      case OP_COPY: result = va; break;
    }
    regs[ins.dst] = result;
  }
  
  for (int e = 0; e < E; ++e) {
    M_out[e] = regs[out_reg[e]];
  }
}

// Shadow float matrix multiply for approximate scoring
__device__ void matmul_float_local(
    const float* A, const float* B, float* C, int m)
{
  float temp[100];
  
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < m; ++t) {
        acc += A[i * m + t] * B[t * m + j];
      }
      temp[i * m + j] = acc;
    }
  }
  
  int E = m * m;
  for (int e = 0; e < E; ++e) {
    C[e] = temp[e];
  }
}

__global__ void k_walk_fused(
    int depth, int depth1, int depth2, int m, int dim, int K, int B,
    const Instr* instr, int n_instr,
    const u32* const_table, int n_const,
    const uint16_t* out_reg,
    const i32* shifts,
    const i32* dirs,
    const PrimeMeta* pm,
    u32* P_final,
    uint8_t* alive,
    float* est1, float* est2,
    float* delta1, float* delta2)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  
  int E = m * m;
  const i32* shift_b = shifts + b * dim;
  
  // Use prime k=0 for shadow float (approximate)
  u32 p0 = pm[0].p;
  u64 mu0 = pm[0].mu;
  
  // Local storage for P matrices (RNS) and shadow float
  u32 P_local[100];      // max 10x10 = 100
  float P_float[100];
  u32 M_step[100];
  float M_float[100];
  u32 x_vals[16];        // max dim
  
  // Initialize P to identity
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      int idx = i * m + j;
      P_local[idx] = (i == j) ? 1 : 0;
      P_float[idx] = (i == j) ? 1.0f : 0.0f;
    }
  }
  
  uint8_t lane_alive = 1;
  const u32* consts_k0 = const_table;
  
  // Walk loop
  for (int t = 0; t < depth && lane_alive; ++t) {
    // Compute x-values for this step
    compute_x_vals(shift_b, dirs, dim, t, p0, x_vals);
    
    // Evaluate step matrix
    eval_step_matrix(instr, n_instr, consts_k0, n_const, out_reg,
                     x_vals, dim, m, p0, mu0, M_step, lane_alive);
    
    if (!lane_alive) break;
    
    // Float version of M_step
    for (int e = 0; e < E; ++e) {
      M_float[e] = (float)M_step[e];
    }
    
    // P = P @ M_step
    matmul_local(P_local, M_step, P_local, m, p0, mu0);
    matmul_float_local(P_float, M_float, P_float, m);
    
    // Snapshots
    if (t + 1 == depth1) {
      float norm = 0.0f;
      for (int e = 0; e < E; ++e) {
        norm += P_float[e] * P_float[e];
      }
      est1[b] = sqrtf(norm);
      delta1[b] = (P_float[0] != 0.0f) ? fabsf(P_float[1] / P_float[0]) : 1e30f;
    }
    
    if (t + 1 == depth2) {
      float norm = 0.0f;
      for (int e = 0; e < E; ++e) {
        norm += P_float[e] * P_float[e];
      }
      est2[b] = sqrtf(norm);
      delta2[b] = (P_float[0] != 0.0f) ? fabsf(P_float[1] / P_float[0]) : 1e30f;
    }
  }
  
  // Store final P (only for k=0 in this simplified version)
  int out_base = b * E;  // For full version: k * B * E + b * E
  for (int e = 0; e < E; ++e) {
    P_final[out_base + e] = P_local[e];
  }
  alive[b] = lane_alive;
}

void walk_fused(
    const WalkConfig& cfg,
    const Program& prog_step,
    const i32* shifts,
    const i32* dirs,
    const PrimeMeta* pm,
    WalkOutputs out)
{
#ifdef RNS_HAS_GPU
  int blockSize = 64;
  int gridSize = (cfg.B + blockSize - 1) / blockSize;
  
  hipLaunchKernelGGL(k_walk_fused, dim3(gridSize), dim3(blockSize), 0, 0,
                     cfg.depth, cfg.depth1, cfg.depth2, cfg.m, cfg.dim, cfg.K, cfg.B,
                     prog_step.instr, prog_step.n_instr,
                     prog_step.const_table, prog_step.n_const,
                     prog_step.out_reg,
                     shifts, dirs, pm,
                     out.P_final, out.alive,
                     out.est1, out.est2, out.delta1, out.delta2);
#else
  walk_fused_cpu(cfg, prog_step, shifts, dirs, pm, out);
#endif
}

} // namespace rns

#ifndef RNS_GPU_CRT_H
#define RNS_GPU_CRT_H

#include "config.h"
#include "modops.h"
#include <vector>
#include <cmath>

namespace rns {

// GPU CRT approximation for fast scoring without CPU roundtrip
// Uses floating-point weights precomputed on CPU

struct GpuCrtPlan {
  int K_small;                      // Number of primes for scoring
  std::vector<float> weights;       // CRT weights (CPU)
  std::vector<float> log_weights;   // Log-space weights (CPU)
  
#ifdef RNS_HAS_GPU
  float* d_weights;                 // Device weights
  float* d_log_weights;             // Device log weights
#endif
};

// Precompute CRT weights for K_small primes
// weight[i] ≈ M_i * y_i where M_i = prod(p_j, j≠i), y_i = M_i^(-1) mod p_i
inline GpuCrtPlan create_gpu_crt_plan(const PrimeMeta* pm, int K_small) {
  GpuCrtPlan plan;
  plan.K_small = K_small;
  plan.weights.resize(K_small);
  plan.log_weights.resize(K_small);
  
  // Compute M_total = prod(p_i) in log space to avoid overflow
  double log_M_total = 0.0;
  for (int i = 0; i < K_small; ++i) {
    log_M_total += std::log((double)pm[i].p);
  }
  
  // For each prime, compute the CRT coefficient weight
  for (int i = 0; i < K_small; ++i) {
    // M_i = M_total / p_i (in log space: log_M_total - log(p_i))
    double log_Mi = log_M_total - std::log((double)pm[i].p);
    
    // y_i = M_i^(-1) mod p_i
    // Compute M_i mod p_i first
    u64 Mi_mod_pi = 1;
    for (int j = 0; j < K_small; ++j) {
      if (j != i) {
        Mi_mod_pi = (Mi_mod_pi * pm[j].p) % pm[i].p;
      }
    }
    u32 yi = pow_mod((u32)Mi_mod_pi, pm[i].p - 2, pm[i].p, pm[i].mu);
    
    // Weight = M_i * y_i (normalized to avoid overflow)
    // We store log(M_i) + offset for stable computation
    plan.log_weights[i] = (float)log_Mi;
    
    // Linear weight (may overflow for large K_small, use log_weights instead)
    // For K_small <= 3, this is safe
    if (K_small <= 3) {
      double Mi = std::exp(log_Mi);
      plan.weights[i] = (float)(Mi * yi);
    } else {
      plan.weights[i] = 0.0f;  // Use log_weights for K_small > 3
    }
  }
  
#ifdef RNS_HAS_GPU
  // Allocate and copy to device
  hipMalloc(&plan.d_weights, K_small * sizeof(float));
  hipMalloc(&plan.d_log_weights, K_small * sizeof(float));
  hipMemcpy(plan.d_weights, plan.weights.data(), K_small * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(plan.d_log_weights, plan.log_weights.data(), K_small * sizeof(float), hipMemcpyHostToDevice);
#endif
  
  return plan;
}

inline void destroy_gpu_crt_plan(GpuCrtPlan& plan) {
#ifdef RNS_HAS_GPU
  if (plan.d_weights) hipFree(plan.d_weights);
  if (plan.d_log_weights) hipFree(plan.d_log_weights);
  plan.d_weights = nullptr;
  plan.d_log_weights = nullptr;
#endif
}

#ifdef RNS_HAS_GPU
// GPU kernel: compute approximate log-magnitude for each batch element
__global__ void k_crt_approx_log_mag(
    const u32* __restrict__ residues,  // [K_small, B]
    float* __restrict__ log_mags,      // [B]
    const float* __restrict__ log_weights,
    int K_small, int B)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  
  // Log-sum-exp for numerical stability
  float max_log = -1e30f;
  for (int k = 0; k < K_small; ++k) {
    u32 r = residues[k * B + b];
    if (r > 0) {
      float log_term = logf((float)r) + log_weights[k];
      max_log = fmaxf(max_log, log_term);
    }
  }
  
  float sum = 0.0f;
  for (int k = 0; k < K_small; ++k) {
    u32 r = residues[k * B + b];
    if (r > 0) {
      float log_term = logf((float)r) + log_weights[k];
      sum += expf(log_term - max_log);
    }
  }
  
  log_mags[b] = max_log + logf(sum + 1e-30f);
}

// GPU kernel: compute delta = |P[0,1]| / |P[0,0]| for each batch
__global__ void k_crt_approx_delta(
    const u32* __restrict__ P_residues,  // [K_small, B, E]
    float* __restrict__ deltas,          // [B]
    const float* __restrict__ log_weights,
    int K_small, int B, int E)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  
  // Gather P[0,0] and P[0,1] across primes
  float log_p00 = -1e30f, log_p01 = -1e30f;
  float sum00 = 0.0f, sum01 = 0.0f;
  float max00 = -1e30f, max01 = -1e30f;
  
  // First pass: find max for stability
  for (int k = 0; k < K_small; ++k) {
    u32 r00 = P_residues[k * B * E + b * E + 0];
    u32 r01 = P_residues[k * B * E + b * E + 1];
    
    if (r00 > 0) {
      float log_term = logf((float)r00) + log_weights[k];
      max00 = fmaxf(max00, log_term);
    }
    if (r01 > 0) {
      float log_term = logf((float)r01) + log_weights[k];
      max01 = fmaxf(max01, log_term);
    }
  }
  
  // Second pass: accumulate
  for (int k = 0; k < K_small; ++k) {
    u32 r00 = P_residues[k * B * E + b * E + 0];
    u32 r01 = P_residues[k * B * E + b * E + 1];
    
    if (r00 > 0) {
      float log_term = logf((float)r00) + log_weights[k];
      sum00 += expf(log_term - max00);
    }
    if (r01 > 0) {
      float log_term = logf((float)r01) + log_weights[k];
      sum01 += expf(log_term - max01);
    }
  }
  
  log_p00 = max00 + logf(sum00 + 1e-30f);
  log_p01 = max01 + logf(sum01 + 1e-30f);
  
  // delta = exp(log_p01 - log_p00)
  deltas[b] = expf(log_p01 - log_p00);
}
#endif

// Host function to compute approximate delta scores on GPU
inline void gpu_approx_delta(
    const u32* d_P_residues,  // Device: [K_small, B, E]
    float* d_deltas,          // Device: [B]
    const GpuCrtPlan& plan,
    int B, int E)
{
#ifdef RNS_HAS_GPU
  dim3 block(256);
  dim3 grid((B + 255) / 256);
  hipLaunchKernelGGL(k_crt_approx_delta, grid, block, 0, 0,
                     d_P_residues, d_deltas, plan.d_log_weights,
                     plan.K_small, B, E);
#else
  // CPU fallback - not implemented here, use crt_approx.h functions
  (void)d_P_residues;
  (void)d_deltas;
  (void)plan;
  (void)B;
  (void)E;
#endif
}

// Host function to compute approximate log magnitudes on GPU
inline void gpu_approx_log_mags(
    const u32* d_residues,    // Device: [K_small, B]
    float* d_log_mags,        // Device: [B]
    const GpuCrtPlan& plan,
    int B)
{
#ifdef RNS_HAS_GPU
  dim3 block(256);
  dim3 grid((B + 255) / 256);
  hipLaunchKernelGGL(k_crt_approx_log_mag, grid, block, 0, 0,
                     d_residues, d_log_mags, plan.d_log_weights,
                     plan.K_small, B);
#else
  (void)d_residues;
  (void)d_log_mags;
  (void)plan;
  (void)B;
#endif
}

} // namespace rns

#endif // RNS_GPU_CRT_H

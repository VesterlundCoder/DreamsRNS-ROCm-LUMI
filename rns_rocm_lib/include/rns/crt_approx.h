#ifndef RNS_CRT_APPROX_H
#define RNS_CRT_APPROX_H

#include "config.h"
#include "modops.h"
#include <cmath>
#include <vector>

namespace rns {

// K_small CRT scoring: Use first K_small primes for approximate magnitude
// This avoids float overflow that occurs with shadow float after many matrix muls

struct CrtApproxPlan {
  int K_small;                    // Number of primes to use (typically 3-4)
  std::vector<u32> primes;        // p[0..K_small-1]
  std::vector<u64> M_partial;     // M_i = prod(p_j for j != i)
  std::vector<u32> y_partial;     // y_i = M_i^(-1) mod p_i
  double log_M_total;             // log(M_total) for normalization
};

// Create plan for K_small primes
inline CrtApproxPlan create_crt_approx_plan(const PrimeMeta* pm, int K_small) {
  CrtApproxPlan plan;
  plan.K_small = K_small;
  plan.primes.resize(K_small);
  plan.M_partial.resize(K_small);
  plan.y_partial.resize(K_small);
  
  // Collect primes
  for (int i = 0; i < K_small; ++i) {
    plan.primes[i] = pm[i].p;
  }
  
  // Compute M_total and M_partial
  // M_total = prod(p_i)
  // M_i = M_total / p_i
  
  // For K_small=3 with 31-bit primes: M_total ~ 2^93
  // We compute M_i mod each prime and also keep log for normalization
  
  plan.log_M_total = 0.0;
  for (int i = 0; i < K_small; ++i) {
    plan.log_M_total += std::log((double)plan.primes[i]);
  }
  
  // Compute M_i as product of all primes except p_i
  // Store only mod-reduced values since we can't store 2^93 exactly
  for (int i = 0; i < K_small; ++i) {
    u64 Mi = 1;
    for (int j = 0; j < K_small; ++j) {
      if (i != j) {
        Mi = (Mi * plan.primes[j]) % plan.primes[i];
      }
    }
    plan.M_partial[i] = Mi;
    
    // y_i = M_i^(-1) mod p_i
    // Use Fermat's little theorem: M_i^(p_i-2) mod p_i
    u64 mu = compute_barrett_mu(plan.primes[i]);
    plan.y_partial[i] = pow_mod((u32)Mi, plan.primes[i] - 2, plan.primes[i], mu);
  }
  
  return plan;
}

// Compute approximate log-magnitude from K_small residues
// Uses Garner's algorithm but converts to double early
inline double crt_approx_log_magnitude(
    const u32* residues,  // residues[k] for k=0..K_small-1
    const CrtApproxPlan& plan)
{
  int K = plan.K_small;
  
  // Garner's algorithm with floating-point accumulation
  // x = sum_{i=0}^{K-1} v_i * prod_{j<i} p_j
  // where v_i = (r_i - x_{i-1}) * y_i mod p_i
  
  double log_result = -std::numeric_limits<double>::infinity();
  double x_approx = 0.0;
  double prod_p = 1.0;
  
  std::vector<u32> v(K);
  
  for (int i = 0; i < K; ++i) {
    // Compute x mod p_i so far
    u64 x_mod_pi = 0;
    u64 pp = 1;
    for (int j = 0; j < i; ++j) {
      x_mod_pi = (x_mod_pi + (u64)v[j] * pp) % plan.primes[i];
      pp = (pp * plan.primes[j]) % plan.primes[i];
    }
    
    // v_i = (r_i - x_mod_pi) * y_i mod p_i
    u32 diff = (residues[i] >= x_mod_pi) 
               ? residues[i] - (u32)x_mod_pi 
               : residues[i] + plan.primes[i] - (u32)x_mod_pi;
    
    u64 mu = compute_barrett_mu(plan.primes[i]);
    v[i] = mul_mod(diff, plan.y_partial[i], plan.primes[i], mu);
    
    // Accumulate in floating point
    x_approx += (double)v[i] * prod_p;
    prod_p *= (double)plan.primes[i];
  }
  
  // Return log of magnitude
  if (x_approx > 0) {
    return std::log(x_approx);
  }
  return -std::numeric_limits<double>::infinity();
}

// Compute approximate ratio |a/b| from K_small residues
// Useful for delta scoring: delta = |P[0,1]| / |P[0,0]|
inline double crt_approx_ratio(
    const u32* residues_a,  // Numerator residues
    const u32* residues_b,  // Denominator residues
    const CrtApproxPlan& plan)
{
  double log_a = crt_approx_log_magnitude(residues_a, plan);
  double log_b = crt_approx_log_magnitude(residues_b, plan);
  
  if (std::isinf(log_b) && log_b < 0) {
    // Denominator is zero
    return std::numeric_limits<double>::infinity();
  }
  
  return std::exp(log_a - log_b);
}

// Compute approximate Frobenius norm from matrix residues
// ||M||_F = sqrt(sum_ij |M_ij|^2)
inline double crt_approx_frobenius_norm(
    const u32* matrix_residues,  // [K_small][E] layout
    int E,                       // m*m elements
    const CrtApproxPlan& plan)
{
  double sum_sq = 0.0;
  
  for (int e = 0; e < E; ++e) {
    // Gather residues for element e across K_small primes
    std::vector<u32> elem_residues(plan.K_small);
    for (int k = 0; k < plan.K_small; ++k) {
      elem_residues[k] = matrix_residues[k * E + e];
    }
    
    double log_mag = crt_approx_log_magnitude(elem_residues.data(), plan);
    if (!std::isinf(log_mag)) {
      double mag = std::exp(log_mag);
      sum_sq += mag * mag;
    }
  }
  
  return std::sqrt(sum_sq);
}

// Batch scoring: compute delta and est for B matrices
// Input: P_residues[K_small][B][E]
// Output: delta[B], est[B]
inline void crt_approx_score_batch(
    const u32* P_residues,  // [K_small, B, E]
    float* delta,           // [B]
    float* est,             // [B]
    int K_small, int B, int m,
    const CrtApproxPlan& plan)
{
  int E = m * m;
  
  for (int b = 0; b < B; ++b) {
    // Gather P[0,0] and P[0,1] residues across primes
    std::vector<u32> r00(K_small), r01(K_small);
    for (int k = 0; k < K_small; ++k) {
      r00[k] = P_residues[k * B * E + b * E + 0];      // P[0,0]
      r01[k] = P_residues[k * B * E + b * E + 1];      // P[0,1]
    }
    
    // delta = |P[0,1]| / |P[0,0]|
    delta[b] = (float)crt_approx_ratio(r01.data(), r00.data(), plan);
    
    // est = Frobenius norm (approximate)
    std::vector<u32> matrix_residues(K_small * E);
    for (int k = 0; k < K_small; ++k) {
      for (int e = 0; e < E; ++e) {
        matrix_residues[k * E + e] = P_residues[k * B * E + b * E + e];
      }
    }
    est[b] = (float)crt_approx_frobenius_norm(matrix_residues.data(), E, plan);
  }
}

} // namespace rns

#endif // RNS_CRT_APPROX_H

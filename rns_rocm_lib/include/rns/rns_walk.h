#ifndef RNS_WALK_H
#define RNS_WALK_H

#include "config.h"
#include "modops.h"
#include "rns_eval.h"

namespace rns {

struct WalkConfig {
  int depth;       // total walk depth
  int depth1;      // snapshot 1 position
  int depth2;      // snapshot 2 position (usually == depth)
  int m;           // matrix dimension
  int dim;         // number of x-variables
  int K;           // number of primes
  int B;           // batch size
};

struct WalkOutputs {
  u32* P_final;      // [K][B][E] final RNS matrix product
  uint8_t* alive;    // [K][B] alive mask
  float* est1;       // [B] estimate at depth1 (from shadow float)
  float* est2;       // [B] estimate at depth2 (from shadow float)
  float* delta1;     // [B] delta proxy at depth1
  float* delta2;     // [B] delta proxy at depth2
};

// Fused walk kernel: computes P = prod_{t=0}^{depth-1} M_t where M_t = prog(x + t*dir)
// Maintains parallel shadow float for approximate scoring
// shifts: [B][dim] base shift values
// dirs: [dim] direction vector
void walk_fused(
    const WalkConfig& cfg,
    const Program& prog_step,
    const i32* shifts,
    const i32* dirs,
    const PrimeMeta* pm,
    WalkOutputs out);

// CPU reference implementation
void walk_fused_cpu(
    const WalkConfig& cfg,
    const Program& prog_step,
    const i32* shifts,
    const i32* dirs,
    const PrimeMeta* pm,
    WalkOutputs out);

// Initialize P to identity matrix for all batches
void init_identity_batch(u32* P, int K, int B, int m);

} // namespace rns

#endif // RNS_WALK_H

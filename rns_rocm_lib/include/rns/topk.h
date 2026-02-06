#ifndef RNS_TOPK_H
#define RNS_TOPK_H

#include "config.h"

namespace rns {

struct TopKItem {
  float score;
  int shift_idx;
  float est;
};

struct TopKConfig {
  int B;           // total number of items
  int Kkeep;       // number of items to keep
  bool ascending;  // true for smallest scores, false for largest
};

// GPU TopK reduction: select Kkeep items with best scores
// scores: [B] input scores
// est: [B] corresponding estimates (carried through)
// out_topk: [Kkeep] output items
void topk_reduce(
    const float* scores,
    const float* est,
    TopKItem* out_topk,
    TopKConfig cfg);

// CPU reference implementation
void topk_reduce_cpu(
    const float* scores,
    const float* est,
    TopKItem* out_topk,
    TopKConfig cfg);

} // namespace rns

#endif // RNS_TOPK_H

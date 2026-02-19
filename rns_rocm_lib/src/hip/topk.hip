#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>
#include "rns/topk.h"

namespace rns {

// Simple parallel reduction for TopK
// For v1, we use a straightforward approach
// Future: use radix select or bitonic sort

__global__ void k_topk_init(
    const float* scores, const float* est,
    TopKItem* items, int B)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B) return;
  
  items[idx].score = scores[idx];
  items[idx].shift_idx = idx;
  items[idx].est = est[idx];
}

void topk_reduce(
    const float* scores,
    const float* est,
    TopKItem* out_topk,
    TopKConfig cfg)
{
#ifdef RNS_HAS_GPU
  // For GPU: copy to host and use CPU sort (simple v1 approach)
  // Future: implement GPU bitonic sort or radix select
  
  std::vector<float> h_scores(cfg.B);
  std::vector<float> h_est(cfg.B);
  
  hipMemcpy(h_scores.data(), scores, cfg.B * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(h_est.data(), est, cfg.B * sizeof(float), hipMemcpyDeviceToHost);
  
  std::vector<TopKItem> items(cfg.B);
  for (int i = 0; i < cfg.B; ++i) {
    items[i].score = h_scores[i];
    items[i].shift_idx = i;
    items[i].est = h_est[i];
  }
  
  if (cfg.ascending) {
    std::partial_sort(items.begin(), items.begin() + cfg.Kkeep, items.end(),
                      [](const TopKItem& a, const TopKItem& b) {
                        return a.score < b.score;
                      });
  } else {
    std::partial_sort(items.begin(), items.begin() + cfg.Kkeep, items.end(),
                      [](const TopKItem& a, const TopKItem& b) {
                        return a.score > b.score;
                      });
  }
  
  hipMemcpy(out_topk, items.data(), cfg.Kkeep * sizeof(TopKItem), hipMemcpyHostToDevice);
#else
  topk_reduce_cpu(scores, est, out_topk, cfg);
#endif
}

} // namespace rns

#include "rns/topk.h"
#include <vector>
#include <algorithm>

namespace rns {

#ifndef RNS_HAS_GPU
void topk_reduce(
    const float* scores,
    const float* est,
    TopKItem* out_topk,
    TopKConfig cfg)
{
  topk_reduce_cpu(scores, est, out_topk, cfg);
}
#endif

void topk_reduce_cpu(
    const float* scores,
    const float* est,
    TopKItem* out_topk,
    TopKConfig cfg)
{
  std::vector<TopKItem> items(cfg.B);
  for (int i = 0; i < cfg.B; ++i) {
    items[i].score = scores[i];
    items[i].shift_idx = i;
    items[i].est = est[i];
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
  
  for (int i = 0; i < cfg.Kkeep; ++i) {
    out_topk[i] = items[i];
  }
}

} // namespace rns

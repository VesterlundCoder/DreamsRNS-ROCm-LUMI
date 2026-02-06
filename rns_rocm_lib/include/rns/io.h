#ifndef RNS_IO_H
#define RNS_IO_H

#include <string>
#include <vector>
#include "topk.h"

namespace rns {

// Append hits to JSONL file
void append_hits_jsonl(
    const std::string& path,
    const std::string& cmf_id,
    const TopKItem* items,
    int n_items);

// Write summary CSV
void write_summary_csv(
    const std::string& path,
    const std::string& cmf_id,
    float best_delta,
    int best_shift,
    float best_est);

// Append row to CSV (creates header if file doesn't exist)
void append_summary_csv(
    const std::string& path,
    const std::string& cmf_id,
    float best_delta,
    int best_shift,
    float best_est);

} // namespace rns

#endif // RNS_IO_H

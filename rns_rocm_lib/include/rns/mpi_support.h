#ifndef RNS_MPI_SUPPORT_H
#define RNS_MPI_SUPPORT_H

#include "config.h"
#include "rns_walk.h"
#include "topk.h"
#include <string>
#include <vector>

#ifdef RNS_HAS_MPI
#include <mpi.h>
#endif

namespace rns {

struct MpiConfig {
  int world_size;      // Total number of ranks
  int rank;            // This rank's ID
  int gpu_id;          // GPU assigned to this rank (usually = rank)
  std::string log_dir; // Directory for per-rank logs
  bool is_root;        // True if rank == 0
};

// Initialize MPI and configure for RNS
inline bool init_mpi_rns(MpiConfig& cfg, int* argc = nullptr, char*** argv = nullptr) {
#ifdef RNS_HAS_MPI
  int provided;
  MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
  
  MPI_Comm_size(MPI_COMM_WORLD, &cfg.world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &cfg.rank);
  
  cfg.gpu_id = cfg.rank;  // Default: 1 GPU per rank
  cfg.is_root = (cfg.rank == 0);
  cfg.log_dir = ".";
  
#ifdef RNS_HAS_GPU
  // Set GPU for this rank
  int num_gpus;
  hipGetDeviceCount(&num_gpus);
  if (num_gpus > 0) {
    cfg.gpu_id = cfg.rank % num_gpus;
    hipSetDevice(cfg.gpu_id);
  }
#endif
  
  return true;
#else
  // Single-process fallback
  cfg.world_size = 1;
  cfg.rank = 0;
  cfg.gpu_id = 0;
  cfg.is_root = true;
  cfg.log_dir = ".";
  return true;
#endif
}

inline void finalize_mpi_rns() {
#ifdef RNS_HAS_MPI
  MPI_Finalize();
#endif
}

// Get per-rank log filename
inline std::string get_rank_log_path(const MpiConfig& cfg, const std::string& base) {
  if (cfg.world_size == 1) {
    return cfg.log_dir + "/" + base;
  }
  return cfg.log_dir + "/" + base + "_rank" + std::to_string(cfg.rank);
}

// Distributed walk: each rank processes a portion of the batch
struct DistributedWalkResult {
  std::vector<TopKItem> global_topk;  // Only valid on root
  int total_alive;                     // Sum across all ranks
  double total_time_ms;                // Max time across ranks
};

inline DistributedWalkResult run_distributed_walk(
    const MpiConfig& cfg,
    const WalkConfig& walk_cfg,
    const Program& prog,
    const i32* all_shifts,     // Full shift array (root only, others can be null)
    const i32* dirs,           // Direction vector (all ranks need this)
    const PrimeMeta* pm,       // Prime metadata (all ranks need this)
    int total_B,               // Total batch size
    int Kkeep)                 // TopK to keep
{
  DistributedWalkResult result;
  
#ifdef RNS_HAS_MPI
  // Calculate local batch size
  int base_batch = total_B / cfg.world_size;
  int remainder = total_B % cfg.world_size;
  int local_B = base_batch + (cfg.rank < remainder ? 1 : 0);
  int start_idx = cfg.rank * base_batch + std::min(cfg.rank, remainder);
  
  // Scatter shifts to all ranks
  std::vector<int> sendcounts(cfg.world_size);
  std::vector<int> displs(cfg.world_size);
  int offset = 0;
  for (int r = 0; r < cfg.world_size; ++r) {
    int r_batch = base_batch + (r < remainder ? 1 : 0);
    sendcounts[r] = r_batch * walk_cfg.dim;
    displs[r] = offset;
    offset += sendcounts[r];
  }
  
  std::vector<i32> local_shifts(local_B * walk_cfg.dim);
  MPI_Scatterv(all_shifts, sendcounts.data(), displs.data(), MPI_INT,
               local_shifts.data(), local_B * walk_cfg.dim, MPI_INT,
               0, MPI_COMM_WORLD);
  
  // Allocate local outputs
  int E = walk_cfg.m * walk_cfg.m;
  std::vector<u32> local_P(local_B * E);
  std::vector<uint8_t> local_alive(local_B);
  std::vector<float> local_est1(local_B), local_est2(local_B);
  std::vector<float> local_delta1(local_B), local_delta2(local_B);
  
  WalkConfig local_cfg = walk_cfg;
  local_cfg.B = local_B;
  
  WalkOutputs local_out;
  local_out.P_final = local_P.data();
  local_out.alive = local_alive.data();
  local_out.est1 = local_est1.data();
  local_out.est2 = local_est2.data();
  local_out.delta1 = local_delta1.data();
  local_out.delta2 = local_delta2.data();
  
  // Run local walk
  auto t0 = MPI_Wtime();
  walk_fused_cpu(local_cfg, prog, local_shifts.data(), dirs, pm, local_out);
  auto t1 = MPI_Wtime();
  double local_time = (t1 - t0) * 1000.0;
  
  // Count local alive
  int local_alive_count = 0;
  for (int b = 0; b < local_B; ++b) {
    if (local_alive[b]) local_alive_count++;
  }
  
  // Local TopK
  std::vector<TopKItem> local_topk(Kkeep);
  TopKConfig topk_cfg;
  topk_cfg.B = local_B;
  topk_cfg.Kkeep = Kkeep;
  topk_cfg.ascending = true;
  topk_reduce_cpu(local_delta2.data(), local_est2.data(), local_topk.data(), topk_cfg);
  
  // Adjust indices to global
  for (auto& item : local_topk) {
    item.shift_idx += start_idx;
  }
  
  // Gather all local TopK to root
  std::vector<TopKItem> all_topk;
  if (cfg.is_root) {
    all_topk.resize(cfg.world_size * Kkeep);
  }
  
  MPI_Gather(local_topk.data(), Kkeep * sizeof(TopKItem), MPI_BYTE,
             all_topk.data(), Kkeep * sizeof(TopKItem), MPI_BYTE,
             0, MPI_COMM_WORLD);
  
  // Reduce alive counts and max time
  MPI_Reduce(&local_alive_count, &result.total_alive, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_time, &result.total_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
  // Root: final TopK selection
  if (cfg.is_root) {
    std::partial_sort(all_topk.begin(), all_topk.begin() + Kkeep, all_topk.end(),
                      [](const TopKItem& a, const TopKItem& b) {
                        return a.score < b.score;
                      });
    result.global_topk.assign(all_topk.begin(), all_topk.begin() + Kkeep);
  }
  
#else
  // Single-process fallback
  int E = walk_cfg.m * walk_cfg.m;
  std::vector<u32> P_final(total_B * E);
  std::vector<uint8_t> alive(total_B);
  std::vector<float> est1(total_B), est2(total_B);
  std::vector<float> delta1(total_B), delta2(total_B);
  
  WalkConfig cfg_copy = walk_cfg;
  cfg_copy.B = total_B;
  
  WalkOutputs out;
  out.P_final = P_final.data();
  out.alive = alive.data();
  out.est1 = est1.data();
  out.est2 = est2.data();
  out.delta1 = delta1.data();
  out.delta2 = delta2.data();
  
  auto t0 = std::chrono::high_resolution_clock::now();
  walk_fused_cpu(cfg_copy, prog, all_shifts, dirs, pm, out);
  auto t1 = std::chrono::high_resolution_clock::now();
  result.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  
  result.total_alive = 0;
  for (int b = 0; b < total_B; ++b) {
    if (alive[b]) result.total_alive++;
  }
  
  result.global_topk.resize(Kkeep);
  TopKConfig topk_cfg;
  topk_cfg.B = total_B;
  topk_cfg.Kkeep = Kkeep;
  topk_cfg.ascending = true;
  topk_reduce_cpu(delta2.data(), est2.data(), result.global_topk.data(), topk_cfg);
#endif
  
  return result;
}

// Broadcast program from root to all ranks
inline void broadcast_program(Program& prog, const MpiConfig& cfg) {
#ifdef RNS_HAS_MPI
  // Broadcast scalar fields
  MPI_Bcast(&prog.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&prog.dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&prog.n_instr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&prog.n_reg, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&prog.n_const, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Note: actual instruction/constant data needs separate handling
  // This is a simplified version; full implementation would serialize the program
#endif
}

} // namespace rns

#endif // RNS_MPI_SUPPORT_H

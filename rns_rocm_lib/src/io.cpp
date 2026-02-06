#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include "rns/io.h"

namespace rns {

static bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

void append_hits_jsonl(
    const std::string& path,
    const std::string& cmf_id,
    const TopKItem* items,
    int n_items)
{
  std::ofstream ofs(path, std::ios::app);
  if (!ofs) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  
  for (int i = 0; i < n_items; ++i) {
    ofs << "{\"cmf_id\":\"" << cmf_id << "\""
        << ",\"shift_idx\":" << items[i].shift_idx
        << ",\"score\":" << std::scientific << std::setprecision(8) << items[i].score
        << ",\"est\":" << std::scientific << std::setprecision(8) << items[i].est
        << "}\n";
  }
  
  ofs.close();
}

void write_summary_csv(
    const std::string& path,
    const std::string& cmf_id,
    float best_delta,
    int best_shift,
    float best_est)
{
  std::ofstream ofs(path);
  if (!ofs) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  
  ofs << "cmf_id,best_delta,best_shift,best_est\n";
  ofs << cmf_id << ","
      << std::scientific << std::setprecision(8) << best_delta << ","
      << best_shift << ","
      << std::scientific << std::setprecision(8) << best_est << "\n";
  
  ofs.close();
}

void append_summary_csv(
    const std::string& path,
    const std::string& cmf_id,
    float best_delta,
    int best_shift,
    float best_est)
{
  bool needs_header = !file_exists(path);
  
  std::ofstream ofs(path, std::ios::app);
  if (!ofs) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  
  if (needs_header) {
    ofs << "cmf_id,best_delta,best_shift,best_est\n";
  }
  
  ofs << cmf_id << ","
      << std::scientific << std::setprecision(8) << best_delta << ","
      << best_shift << ","
      << std::scientific << std::setprecision(8) << best_est << "\n";
  
  ofs.close();
}

} // namespace rns

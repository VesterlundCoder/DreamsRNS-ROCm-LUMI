#pragma once
#include <cstdint>
#include <string>
#include <chrono>
#include <stdexcept>

namespace rns {

// ============================================================================
// Error Handling
// ============================================================================

#ifdef RNS_HAS_GPU
void hip_check(int error, const char* file, int line);
#define HIP_CHECK(e) rns::hip_check((int)(e), __FILE__, __LINE__)
#else
#define HIP_CHECK(e) ((void)0)
#endif

// ============================================================================
// Timing
// ============================================================================

class Timer {
public:
  Timer() : start_(std::chrono::high_resolution_clock::now()) {}
  
  void reset() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  
  double elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }
  
  double elapsed_us() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now - start_).count();
  }
  
private:
  std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Math Utilities
// ============================================================================

/**
 * Check if n is prime using Miller-Rabin.
 */
bool is_prime(uint32_t n);

/**
 * Greatest common divisor.
 */
uint32_t gcd(uint32_t a, uint32_t b);

/**
 * Extended Euclidean algorithm.
 * Returns gcd(a, b) and sets x, y such that a*x + b*y = gcd(a,b).
 */
int64_t extended_gcd(int64_t a, int64_t b, int64_t& x, int64_t& y);

/**
 * Modular inverse using extended GCD.
 * Returns a^{-1} mod m, or 0 if not invertible.
 */
uint32_t mod_inverse(uint32_t a, uint32_t m);

/**
 * Compute Barrett constant mu = floor(2^64 / p).
 */
uint64_t compute_barrett_mu(uint32_t p);

// ============================================================================
// Debug Printing
// ============================================================================

/**
 * Print array for debugging.
 */
void print_array(const uint32_t* arr, int n, const char* name = nullptr);

/**
 * Format number with commas.
 */
std::string format_number(uint64_t n);

}  // namespace rns

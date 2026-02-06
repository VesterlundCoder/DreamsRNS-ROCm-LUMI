#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <random>

#include "rns/utils.h"

#ifdef RNS_HAS_GPU
#include <hip/hip_runtime.h>
#endif

namespace rns {

// ============================================================================
// Error Handling
// ============================================================================

#ifdef RNS_HAS_GPU
void hip_check(int error, const char* file, int line) {
  if (error != hipSuccess) {
    std::ostringstream oss;
    oss << "HIP error at " << file << ":" << line << ": "
        << hipGetErrorString((hipError_t)error);
    throw std::runtime_error(oss.str());
  }
}
#endif

// ============================================================================
// Math Utilities
// ============================================================================

bool is_prime(uint32_t n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if (n % 2 == 0) return false;
  if (n < 9) return true;
  if (n % 3 == 0) return false;
  
  // Miller-Rabin with deterministic witnesses for 32-bit
  uint32_t d = n - 1;
  int r = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    r++;
  }
  
  // Witnesses sufficient for n < 2^32
  uint32_t witnesses[] = {2, 7, 61};
  
  for (uint32_t a : witnesses) {
    if (a >= n) continue;
    
    // Compute a^d mod n
    uint64_t x = 1;
    uint64_t base = a;
    uint32_t exp = d;
    while (exp > 0) {
      if (exp & 1) {
        x = (x * base) % n;
      }
      base = (base * base) % n;
      exp >>= 1;
    }
    
    if (x == 1 || x == n - 1) continue;
    
    bool composite = true;
    for (int i = 0; i < r - 1; ++i) {
      x = (x * x) % n;
      if (x == n - 1) {
        composite = false;
        break;
      }
    }
    
    if (composite) return false;
  }
  
  return true;
}

uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

int64_t extended_gcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
  if (b == 0) {
    x = 1;
    y = 0;
    return a;
  }
  
  int64_t x1, y1;
  int64_t g = extended_gcd(b, a % b, x1, y1);
  x = y1;
  y = x1 - (a / b) * y1;
  return g;
}

uint32_t mod_inverse(uint32_t a, uint32_t m) {
  if (a == 0) return 0;
  
  int64_t x, y;
  int64_t g = extended_gcd((int64_t)a, (int64_t)m, x, y);
  
  if (g != 1) {
    return 0;  // Not invertible
  }
  
  // Make x positive
  x = ((x % (int64_t)m) + (int64_t)m) % (int64_t)m;
  return (uint32_t)x;
}

uint64_t compute_barrett_mu(uint32_t p) {
  __uint128_t one = ((__uint128_t)1 << 64);
  return (uint64_t)(one / p);
}

// ============================================================================
// Debug Printing
// ============================================================================

void print_array(const uint32_t* arr, int n, const char* name) {
  if (name) {
    std::cout << name << ": ";
  }
  std::cout << "[";
  for (int i = 0; i < n; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << arr[i];
    if (i >= 10) {
      std::cout << ", ...";
      break;
    }
  }
  std::cout << "]" << std::endl;
}

std::string format_number(uint64_t n) {
  std::string s = std::to_string(n);
  std::string result;
  int count = 0;
  for (int i = (int)s.length() - 1; i >= 0; --i) {
    if (count > 0 && count % 3 == 0) {
      result = "," + result;
    }
    result = s[i] + result;
    count++;
  }
  return result;
}

}  // namespace rns

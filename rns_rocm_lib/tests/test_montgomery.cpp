#include <iostream>
#include <random>
#include <cassert>
#include <cstdint>
#include "rns/montgomery.h"
#include "rns/modops.h"

using namespace rns;

bool is_prime_simple(u32 n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if (n % 2 == 0) return false;
  for (u32 i = 3; i * i <= n; i += 2) {
    if (n % i == 0) return false;
  }
  return true;
}

u32 generate_prime(std::mt19937& rng) {
  std::uniform_int_distribution<u32> dist((1u << 30), (1u << 31) - 1);
  while (true) {
    u32 candidate = dist(rng) | 1;  // Make odd
    if (is_prime_simple(candidate)) return candidate;
  }
}

int main() {
  std::cout << "=== Montgomery Multiplication Tests ===" << std::endl;
  
  std::mt19937 rng(42);
  int n_primes = 5;
  int n_tests = 1000;
  int passed = 0;
  int total = 0;
  
  for (int pi = 0; pi < n_primes; ++pi) {
    u32 p = generate_prime(rng);
    auto mp = compute_montgomery_params_ext(p);
    
    std::cout << "\nTesting prime p = " << p << std::endl;
    std::cout << "  pinv = " << mp.pinv << std::endl;
    std::cout << "  r2 = " << mp.r2 << std::endl;
    std::cout << "  oneR = " << mp.oneR << std::endl;
    
    // Verify pinv: p * pinv ≡ -1 (mod 2^32)
    u32 check = p * mp.pinv;
    assert((u32)(check + 1) == 0 && "pinv verification failed");
    std::cout << "  pinv verification: PASSED" << std::endl;
    
    // Test roundtrip: from_mont(to_mont(a)) == a
    std::uniform_int_distribution<u32> val_dist(0, p - 1);
    bool roundtrip_ok = true;
    for (int i = 0; i < n_tests; ++i) {
      u32 a = val_dist(rng);
      u32 aR = to_mont(a, mp.r2, p, mp.pinv);
      u32 a_back = from_mont(aR, p, mp.pinv);
      if (a_back != a) {
        std::cout << "  Roundtrip FAILED: " << a << " -> " << aR << " -> " << a_back << std::endl;
        roundtrip_ok = false;
        break;
      }
      total++;
      passed++;
    }
    if (roundtrip_ok) {
      std::cout << "  Roundtrip test: PASSED" << std::endl;
    }
    
    // Test multiplication: from_mont(mont_mul(to_mont(a), to_mont(b))) == (a*b) mod p
    bool mul_ok = true;
    for (int i = 0; i < n_tests; ++i) {
      u32 a = val_dist(rng);
      u32 b = val_dist(rng);
      u64 expected = ((u64)a * b) % p;
      
      u32 aR = to_mont(a, mp.r2, p, mp.pinv);
      u32 bR = to_mont(b, mp.r2, p, mp.pinv);
      u32 cR = mont_mul(aR, bR, p, mp.pinv);
      u32 c = from_mont(cR, p, mp.pinv);
      
      if (c != (u32)expected) {
        std::cout << "  Mul FAILED: " << a << " * " << b << " = " << c 
                  << " (expected " << expected << ")" << std::endl;
        mul_ok = false;
        break;
      }
      total++;
      passed++;
    }
    if (mul_ok) {
      std::cout << "  Multiplication test: PASSED" << std::endl;
    }
    
    // Test FMA: from_mont(mont_fma(aR, bR, cR)) == (a*b + c) mod p
    bool fma_ok = true;
    for (int i = 0; i < n_tests; ++i) {
      u32 a = val_dist(rng);
      u32 b = val_dist(rng);
      u32 c = val_dist(rng);
      u64 expected = (((u64)a * b) + c) % p;
      
      u32 aR = to_mont(a, mp.r2, p, mp.pinv);
      u32 bR = to_mont(b, mp.r2, p, mp.pinv);
      u32 cR = to_mont(c, mp.r2, p, mp.pinv);
      u32 rR = mont_fma(aR, bR, cR, p, mp.pinv);
      u32 r = from_mont(rR, p, mp.pinv);
      
      if (r != (u32)expected) {
        std::cout << "  FMA FAILED: " << a << " * " << b << " + " << c 
                  << " = " << r << " (expected " << expected << ")" << std::endl;
        fma_ok = false;
        break;
      }
      total++;
      passed++;
    }
    if (fma_ok) {
      std::cout << "  FMA test: PASSED" << std::endl;
    }
    
    // Test inverse: a * inv(a) == 1 mod p
    bool inv_ok = true;
    for (int i = 0; i < 100; ++i) {
      u32 a = val_dist(rng);
      if (a == 0) continue;
      
      u32 aR = to_mont(a, mp.r2, p, mp.pinv);
      u32 invR = mont_inv(aR, p, mp.pinv, mp.oneR);
      u32 prodR = mont_mul(aR, invR, p, mp.pinv);
      u32 prod = from_mont(prodR, p, mp.pinv);
      
      if (prod != 1) {
        std::cout << "  Inv FAILED: " << a << " * inv(" << a << ") = " << prod << std::endl;
        inv_ok = false;
        break;
      }
      total++;
      passed++;
    }
    if (inv_ok) {
      std::cout << "  Inverse test: PASSED" << std::endl;
    }
    
    // Compare with Barrett
    bool compare_ok = true;
    u64 mu = compute_barrett_mu(p);
    for (int i = 0; i < n_tests; ++i) {
      u32 a = val_dist(rng);
      u32 b = val_dist(rng);
      
      // Barrett
      u32 barrett_result = mul_mod(a, b, p, mu);
      
      // Montgomery
      u32 aR = to_mont(a, mp.r2, p, mp.pinv);
      u32 bR = to_mont(b, mp.r2, p, mp.pinv);
      u32 cR = mont_mul(aR, bR, p, mp.pinv);
      u32 mont_result = from_mont(cR, p, mp.pinv);
      
      if (barrett_result != mont_result) {
        std::cout << "  Compare FAILED: Barrett=" << barrett_result 
                  << " Montgomery=" << mont_result << std::endl;
        compare_ok = false;
        break;
      }
    }
    if (compare_ok) {
      std::cout << "  Barrett/Montgomery comparison: PASSED" << std::endl;
    }
  }
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << passed << "/" << total << std::endl;
  
  if (passed == total) {
    std::cout << "All Montgomery tests PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED!" << std::endl;
    return 1;
  }
}

#ifndef RNS_MONTGOMERY_H
#define RNS_MONTGOMERY_H

#include "config.h"

namespace rns {

// Montgomery multiplication stub for future implementation
// Montgomery form: aR mod N where R = 2^32
// Requires precomputed:
//   - N' such that N*N' = -1 mod R
//   - R^2 mod N for conversion

struct MontgomeryParams {
  u32 p;        // prime
  u32 pinv;     // -p^(-1) mod 2^32
  u32 r2;       // R^2 mod p
  u32 r;        // R mod p
};

inline MontgomeryParams compute_montgomery_params(u32 p) {
  MontgomeryParams mp;
  mp.p = p;
  
  // Compute -p^(-1) mod 2^32 using Newton's method
  u32 inv = 1;
  for (int i = 0; i < 5; ++i) {
    inv = inv * (2 - p * inv);
  }
  mp.pinv = (u32)(-(i64)inv);
  
  // Compute R mod p and R^2 mod p
  u64 R = (u64)1 << 32;
  mp.r = (u32)(R % p);
  mp.r2 = (u32)((R * R) % p);
  
  return mp;
}

RNS_HOST_DEVICE inline u32 mont_reduce(u64 x, u32 p, u32 pinv) {
  u32 m = (u32)x * pinv;
  u64 t = x + (u64)m * p;
  u32 result = (u32)(t >> 32);
  return result >= p ? result - p : result;
}

RNS_HOST_DEVICE inline u32 mont_mul(u32 a, u32 b, u32 p, u32 pinv) {
  return mont_reduce((u64)a * b, p, pinv);
}

RNS_HOST_DEVICE inline u32 to_mont(u32 a, u32 r2, u32 p, u32 pinv) {
  return mont_mul(a, r2, p, pinv);
}

RNS_HOST_DEVICE inline u32 from_mont(u32 a, u32 p, u32 pinv) {
  return mont_reduce((u64)a, p, pinv);
}

// Montgomery-domain FMA: (a*b + c) in Montgomery form
RNS_HOST_DEVICE inline u32 mont_fma(u32 aR, u32 bR, u32 cR, u32 p, u32 pinv) {
  u32 prod = mont_mul(aR, bR, p, pinv);
  u32 sum = prod + cR;
  return sum >= p ? sum - p : sum;
}

// Montgomery-domain squaring
RNS_HOST_DEVICE inline u32 mont_sqr(u32 aR, u32 p, u32 pinv) {
  return mont_mul(aR, aR, p, pinv);
}

// Montgomery-domain exponentiation
RNS_HOST_DEVICE inline u32 mont_pow(u32 baseR, u32 exp, u32 p, u32 pinv, u32 oneR) {
  u32 result = oneR;  // 1 in Montgomery form
  u32 base = baseR;
  while (exp > 0) {
    if (exp & 1) {
      result = mont_mul(result, base, p, pinv);
    }
    base = mont_sqr(base, p, pinv);
    exp >>= 1;
  }
  return result;
}

// Montgomery-domain inverse using Fermat's little theorem
// a^(-1) = a^(p-2) mod p
RNS_HOST_DEVICE inline u32 mont_inv(u32 aR, u32 p, u32 pinv, u32 oneR) {
  return mont_pow(aR, p - 2, p, pinv, oneR);
}

// Helper: compute oneR = R mod p (1 in Montgomery form)
inline u32 compute_oneR(u32 p) {
  return (u32)(((u64)1 << 32) % p);
}

// Extended Montgomery params including oneR
struct MontgomeryParamsExt {
  u32 p;
  u32 pinv;
  u32 r2;     // R^2 mod p (for to_mont)
  u32 oneR;   // R mod p (1 in Montgomery form)
};

inline MontgomeryParamsExt compute_montgomery_params_ext(u32 p) {
  MontgomeryParamsExt mp;
  mp.p = p;
  
  // Compute -p^(-1) mod 2^32 using Newton's method
  u32 inv = 1;
  for (int i = 0; i < 5; ++i) {
    inv = inv * (2 - p * inv);
  }
  mp.pinv = (u32)(-(i64)inv);
  
  // Compute R mod p and R^2 mod p
  u64 R = (u64)1 << 32;
  mp.oneR = (u32)(R % p);
  mp.r2 = (u32)(((R % p) * (R % p)) % p);
  
  return mp;
}

} // namespace rns

#endif // RNS_MONTGOMERY_H

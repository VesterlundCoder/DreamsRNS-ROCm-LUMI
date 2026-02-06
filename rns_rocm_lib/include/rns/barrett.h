#ifndef RNS_BARRETT_H
#define RNS_BARRETT_H

#include "config.h"

namespace rns {

// ============================================================================
// Barrett reduction: computes x mod p using precomputed mu = floor(2^64 / p)
//
// BUG FIX for ROCm/HIP:
//   The original code used `unsigned __int128` for the 128-bit multiply
//   x * mu.  This type is NOT supported in HIP device code on AMD GPUs.
//   On the device we use the __umul64hi() intrinsic which returns the
//   upper 64 bits of a 64×64→128 product — exactly what we need.
//   On the host we keep the __int128 path since GCC/Clang support it.
// ============================================================================

RNS_HOST_DEVICE inline u64 mulhi64(u64 a, u64 b) {
#if defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__)
  return __umul64hi(a, b);
#else
  unsigned __int128 prod = (unsigned __int128)a * b;
  return (u64)(prod >> 64);
#endif
}

RNS_HOST_DEVICE inline u32 barrett_reduce_u64(u64 x, u32 p, u64 mu) {
  // q ≈ floor(x * mu / 2^64)  — upper 64 bits of (x * mu)
  u64 q = mulhi64(x, mu);
  u64 r = x - q * (u64)p;
  // At most one correction step needed
  if (r >= (u64)p) r -= (u64)p;
  return (u32)r;
}

RNS_HOST_DEVICE inline u32 barrett_reduce_u32(u32 x, u32 p) {
  return x >= p ? x - p : x;
}

// compute_barrett_mu is host-only, so __int128 is fine here
inline u64 compute_barrett_mu(u32 p) {
  unsigned __int128 one = ((unsigned __int128)1 << 64);
  return (u64)(one / (u64)p);
}

} // namespace rns

#endif // RNS_BARRETT_H

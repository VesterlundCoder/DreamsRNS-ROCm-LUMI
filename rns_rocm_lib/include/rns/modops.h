#ifndef RNS_MODOPS_H
#define RNS_MODOPS_H

#include "config.h"
#include "barrett.h"

namespace rns {

struct PrimeMeta {
  u32 p;
  u32 pad;
  u64 mu;
  u32 pinv;  // for Montgomery (future)
  u32 r2;    // for Montgomery (future)
};

RNS_HOST_DEVICE inline u32 add_mod(u32 a, u32 b, u32 p) {
  u32 sum = a + b;
  return sum >= p ? sum - p : sum;
}

RNS_HOST_DEVICE inline u32 sub_mod(u32 a, u32 b, u32 p) {
  return a >= b ? a - b : a + p - b;
}

RNS_HOST_DEVICE inline u32 mul_mod(u32 a, u32 b, u32 p, u64 mu) {
  u64 prod = (u64)a * b;
  return barrett_reduce_u64(prod, p, mu);
}

RNS_HOST_DEVICE inline u32 mul_mod(u32 a, u32 b, const PrimeMeta& pm) {
  return mul_mod(a, b, pm.p, pm.mu);
}

RNS_HOST_DEVICE inline u32 fma_mod(u32 a, u32 b, u32 c, u32 p, u64 mu) {
  u64 prod = (u64)a * b + c;
  return barrett_reduce_u64(prod, p, mu);
}

RNS_HOST_DEVICE inline u32 fma_mod(u32 a, u32 b, u32 c, const PrimeMeta& pm) {
  return fma_mod(a, b, c, pm.p, pm.mu);
}

RNS_HOST_DEVICE inline u32 neg_mod(u32 a, u32 p) {
  return a == 0 ? 0 : p - a;
}

RNS_HOST_DEVICE inline u32 pow2_mod(u32 a, u32 p, u64 mu) {
  return mul_mod(a, a, p, mu);
}

RNS_HOST_DEVICE inline u32 pow3_mod(u32 a, u32 p, u64 mu) {
  u32 a2 = mul_mod(a, a, p, mu);
  return mul_mod(a2, a, p, mu);
}

RNS_HOST_DEVICE inline u32 pow_mod(u32 base, u32 exp, u32 p, u64 mu) {
  u32 result = 1;
  base = base % p;
  while (exp > 0) {
    if (exp & 1) {
      result = mul_mod(result, base, p, mu);
    }
    exp >>= 1;
    base = mul_mod(base, base, p, mu);
  }
  return result;
}

RNS_HOST_DEVICE inline u32 inv_mod(u32 a, u32 p, u64 mu) {
  if (a == 0) return 0;
  return pow_mod(a, p - 2, p, mu);
}

} // namespace rns

#endif // RNS_MODOPS_H

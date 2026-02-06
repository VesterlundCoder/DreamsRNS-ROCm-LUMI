#ifndef RNS_CONFIG_H
#define RNS_CONFIG_H

#include <cstdint>

#ifdef __HIPCC__
#define RNS_HOST_DEVICE __host__ __device__
#define RNS_DEVICE __device__
#define RNS_HOST __host__
#else
#define RNS_HOST_DEVICE
#define RNS_DEVICE
#define RNS_HOST
#endif

namespace rns {

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr int SUPPORTED_M_SIZES[] = {4, 6, 8, 10};
constexpr int NUM_SUPPORTED_M = 4;

constexpr int MAX_PRIMES = 128;
constexpr int MAX_REGISTERS = 64;
constexpr int MAX_INSTRUCTIONS = 256;
constexpr int MAX_CONSTANTS = 64;

constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 64;

RNS_HOST_DEVICE constexpr bool is_supported_m(int m) {
  return m == 4 || m == 6 || m == 8 || m == 10;
}

RNS_HOST_DEVICE constexpr int m_to_E(int m) {
  return m * m;
}

} // namespace rns

#endif // RNS_CONFIG_H

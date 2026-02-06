#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstring>

#include "rns/rns_api.h"
#include "rns/rns_ops.h"
#include "rns/utils.h"

#ifdef RNS_HAS_GPU
#include <hip/hip_runtime.h>
#include "rns/rns_kernels.h"
#endif

namespace rns {

// ============================================================================
// Device Context
// ============================================================================

DeviceContext create_context(const std::vector<uint32_t>& primes) {
  DeviceContext ctx;
  ctx.K = (int)primes.size();
  
  if (ctx.K <= 0) {
    throw std::runtime_error("No primes provided");
  }
  
  // Build host modulus array
  std::vector<Modulus32> hmods(ctx.K);
  for (int i = 0; i < ctx.K; ++i) {
    uint32_t p = primes[i];
    hmods[i].p = p;
    hmods[i].mu = compute_barrett_mu(p);
    hmods[i].r2 = 0;      // Reserved for Montgomery
    hmods[i].p_inv = 0;   // Reserved for Montgomery
  }
  
#ifdef RNS_HAS_GPU
  hipError_t err = hipMalloc((void**)&ctx.d_mods, ctx.K * sizeof(Modulus32));
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipMalloc failed: ") + hipGetErrorString(err));
  }
  
  err = hipMemcpy(ctx.d_mods, hmods.data(), ctx.K * sizeof(Modulus32), hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    hipFree(ctx.d_mods);
    throw std::runtime_error(std::string("hipMemcpy failed: ") + hipGetErrorString(err));
  }
#else
  // CPU-only: allocate host copy
  ctx.d_mods = new Modulus32[ctx.K];
  std::memcpy(ctx.d_mods, hmods.data(), ctx.K * sizeof(Modulus32));
#endif
  
  ctx.valid = true;
  return ctx;
}

void destroy_context(DeviceContext& ctx) {
  if (ctx.d_mods) {
#ifdef RNS_HAS_GPU
    hipFree(ctx.d_mods);
#else
    delete[] ctx.d_mods;
#endif
  }
  ctx.d_mods = nullptr;
  ctx.K = 0;
  ctx.valid = false;
}

// ============================================================================
// Memory Management
// ============================================================================

uint32_t* device_alloc_u32(size_t count) {
  uint32_t* p = nullptr;
#ifdef RNS_HAS_GPU
  hipError_t err = hipMalloc((void**)&p, count * sizeof(uint32_t));
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipMalloc failed: ") + hipGetErrorString(err));
  }
#else
  p = new uint32_t[count];
#endif
  return p;
}

void device_free(void* ptr) {
  if (ptr) {
#ifdef RNS_HAS_GPU
    hipFree(ptr);
#else
    delete[] static_cast<uint32_t*>(ptr);
#endif
  }
}

void h2d_u32(uint32_t* device, const uint32_t* host, size_t count) {
#ifdef RNS_HAS_GPU
  hipError_t err = hipMemcpy(device, host, count * sizeof(uint32_t), hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipMemcpy H2D failed: ") + hipGetErrorString(err));
  }
#else
  std::memcpy(device, host, count * sizeof(uint32_t));
#endif
}

void d2h_u32(uint32_t* host, const uint32_t* device, size_t count) {
#ifdef RNS_HAS_GPU
  hipError_t err = hipMemcpy(host, device, count * sizeof(uint32_t), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipMemcpy D2H failed: ") + hipGetErrorString(err));
  }
#else
  std::memcpy(host, device, count * sizeof(uint32_t));
#endif
}

void device_sync() {
#ifdef RNS_HAS_GPU
  hipDeviceSynchronize();
#endif
}

// ============================================================================
// RNS Operations (GPU dispatch)
// ============================================================================

void add(const DeviceContext& ctx, uint32_t* out,
         const uint32_t* a, const uint32_t* b, int N) {
#ifdef RNS_HAS_GPU
  rns_add_u32(out, a, b, ctx.d_mods, ctx.K, N);
#else
  // CPU fallback
  for (int k = 0; k < ctx.K; ++k) {
    Modulus32 mod = ctx.d_mods[k];
    for (int i = 0; i < N; ++i) {
      int idx = k * N + i;
      out[idx] = add_mod(a[idx], b[idx], mod);
    }
  }
#endif
}

void mul(const DeviceContext& ctx, uint32_t* out,
         const uint32_t* a, const uint32_t* b, int N) {
#ifdef RNS_HAS_GPU
  rns_mul_u32(out, a, b, ctx.d_mods, ctx.K, N);
#else
  for (int k = 0; k < ctx.K; ++k) {
    Modulus32 mod = ctx.d_mods[k];
    for (int i = 0; i < N; ++i) {
      int idx = k * N + i;
      out[idx] = mul_mod(a[idx], b[idx], mod);
    }
  }
#endif
}

void sub(const DeviceContext& ctx, uint32_t* out,
         const uint32_t* a, const uint32_t* b, int N) {
#ifdef RNS_HAS_GPU
  rns_sub_u32(out, a, b, ctx.d_mods, ctx.K, N);
#else
  for (int k = 0; k < ctx.K; ++k) {
    Modulus32 mod = ctx.d_mods[k];
    for (int i = 0; i < N; ++i) {
      int idx = k * N + i;
      out[idx] = sub_mod(a[idx], b[idx], mod);
    }
  }
#endif
}

void gemm_mod(const DeviceContext& ctx, uint32_t* C,
              const uint32_t* A, const uint32_t* B, int batch, int m) {
#ifdef RNS_HAS_GPU
  rns_gemm_mod_u32(C, A, B, ctx.d_mods, ctx.K, batch, m);
#else
  // CPU reference
  std::vector<Modulus32> mods(ctx.d_mods, ctx.d_mods + ctx.K);
  cpu_gemm_mod(C, A, B, mods, ctx.K, batch, m);
#endif
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

void cpu_gemm_mod(uint32_t* C, const uint32_t* A, const uint32_t* B,
                  const std::vector<Modulus32>& mods, int K, int batch, int m) {
  int E = m * m;
  int N = batch * E;
  
  for (int k = 0; k < K; ++k) {
    Modulus32 mod = mods[k];
    
    for (int b = 0; b < batch; ++b) {
      int base = k * N + b * E;
      
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
          uint32_t acc = 0;
          for (int t = 0; t < m; ++t) {
            uint32_t a_it = A[base + i * m + t];
            uint32_t b_tj = B[base + t * m + j];
            acc = fma_mod(a_it, b_tj, acc, mod);
          }
          C[base + i * m + j] = acc;
        }
      }
    }
  }
}

void cpu_add_mod(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const std::vector<Modulus32>& mods, int K, int N) {
  for (int k = 0; k < K; ++k) {
    Modulus32 mod = mods[k];
    for (int i = 0; i < N; ++i) {
      int idx = k * N + i;
      out[idx] = add_mod(a[idx], b[idx], mod);
    }
  }
}

void cpu_mul_mod(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const std::vector<Modulus32>& mods, int K, int N) {
  for (int k = 0; k < K; ++k) {
    Modulus32 mod = mods[k];
    for (int i = 0; i < N; ++i) {
      int idx = k * N + i;
      out[idx] = mul_mod(a[idx], b[idx], mod);
    }
  }
}

}  // namespace rns

#pragma once
#include <vector>
#include <cstdint>
#include "rns_types.h"

namespace rns {

// ============================================================================
// Device Context Management
// ============================================================================

/**
 * Create a device context with the given primes.
 * Uploads moduli to GPU memory.
 */
DeviceContext create_context(const std::vector<uint32_t>& primes);

/**
 * Destroy device context and free GPU memory.
 */
void destroy_context(DeviceContext& ctx);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate device memory for uint32 array.
 */
uint32_t* device_alloc_u32(size_t count);

/**
 * Free device memory.
 */
void device_free(void* ptr);

/**
 * Copy host to device.
 */
void h2d_u32(uint32_t* device, const uint32_t* host, size_t count);

/**
 * Copy device to host.
 */
void d2h_u32(uint32_t* host, const uint32_t* device, size_t count);

/**
 * Synchronize device.
 */
void device_sync();

// ============================================================================
// RNS Operations
// ============================================================================

/**
 * Elementwise addition: out = a + b (mod p) for each prime.
 * Arrays are [K][N] flattened.
 */
void add(const DeviceContext& ctx, uint32_t* out,
         const uint32_t* a, const uint32_t* b, int N);

/**
 * Elementwise multiplication: out = a * b (mod p) for each prime.
 */
void mul(const DeviceContext& ctx, uint32_t* out,
         const uint32_t* a, const uint32_t* b, int N);

/**
 * Elementwise subtraction: out = a - b (mod p) for each prime.
 */
void sub(const DeviceContext& ctx, uint32_t* out,
         const uint32_t* a, const uint32_t* b, int N);

/**
 * Batched matrix multiplication: C = A @ B (mod p) for each prime and batch.
 * 
 * @param ctx  Device context
 * @param C    Output [K][B*m*m]
 * @param A    Left input [K][B*m*m]
 * @param B    Right input [K][B*m*m]
 * @param batch Number of matrices per prime
 * @param m    Matrix dimension (specialized for 4,6,8,10)
 */
void gemm_mod(const DeviceContext& ctx, uint32_t* C,
              const uint32_t* A, const uint32_t* B, int batch, int m);

// ============================================================================
// CPU Reference Implementations
// ============================================================================

/**
 * CPU reference for modular matrix multiplication.
 * Used for correctness testing.
 */
void cpu_gemm_mod(uint32_t* C, const uint32_t* A, const uint32_t* B,
                  const std::vector<Modulus32>& mods, int K, int batch, int m);

/**
 * CPU reference for elementwise operations.
 */
void cpu_add_mod(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const std::vector<Modulus32>& mods, int K, int N);
void cpu_mul_mod(uint32_t* out, const uint32_t* a, const uint32_t* b,
                 const std::vector<Modulus32>& mods, int K, int N);

}  // namespace rns

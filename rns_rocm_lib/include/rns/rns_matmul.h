#ifndef RNS_MATMUL_H
#define RNS_MATMUL_H

#include "config.h"
#include "modops.h"

namespace rns {

// Batched modular matrix multiply: C = A @ B mod p for each prime k and batch b
// Layout: [K][B][E] where E = m*m, row-major within each matrix

void matmul_mod_m4(const u32* A, const u32* B, u32* C, 
                   const PrimeMeta* pm, int K, int B);

void matmul_mod_m6(const u32* A, const u32* B, u32* C,
                   const PrimeMeta* pm, int K, int B);

void matmul_mod_m8(const u32* A, const u32* B, u32* C,
                   const PrimeMeta* pm, int K, int B);

void matmul_mod_m10(const u32* A, const u32* B, u32* C,
                    const PrimeMeta* pm, int K, int B);

// Generic dispatcher
void matmul_mod(const u32* A, const u32* B, u32* C,
                const PrimeMeta* pm, int K, int B, int m);

// CPU reference implementation
void matmul_mod_cpu(const u32* A, const u32* B, u32* C,
                    const PrimeMeta* pm, int K, int B, int m);

} // namespace rns

#endif // RNS_MATMUL_H

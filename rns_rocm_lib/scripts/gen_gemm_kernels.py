#!/usr/bin/env python3
"""
Generate specialized GEMM kernels for various matrix sizes.

Usage:
    python gen_gemm_kernels.py --max-m 20 --output-dir generated/
"""

import argparse
import os
from pathlib import Path


def generate_kernel_source(m: int) -> str:
    """Generate HIP kernel source for m×m GEMM."""
    E = m * m
    
    # Generate register declarations
    reg_decls = []
    for i in range(m):
        for j in range(m):
            reg_decls.append(f"    u32 c{i}_{j} = 0;")
    
    # Generate FMA operations (unrolled)
    fma_ops = []
    for i in range(m):
        for j in range(m):
            for t in range(m):
                a_idx = i * m + t
                b_idx = t * m + j
                fma_ops.append(
                    f"    c{i}_{j} = fma_mod(A[base + {a_idx}], "
                    f"B[base + {b_idx}], c{i}_{j}, p, mu);"
                )
    
    # Generate stores
    stores = []
    for i in range(m):
        for j in range(m):
            idx = i * m + j
            stores.append(f"    C[base + {idx}] = c{i}_{j};")
    
    return f'''// Auto-generated GEMM kernel for m={m}
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#include "rns/config.h"
#include "rns/modops.h"

#ifdef RNS_HAS_GPU
#include <hip/hip_runtime.h>
#endif

namespace rns {{

#ifdef RNS_HAS_GPU
__global__ void k_gemm_mod_m{m}(
    u32* __restrict__ C,
    const u32* __restrict__ A,
    const u32* __restrict__ B,
    const u32* __restrict__ primes,
    const u64* __restrict__ mus,
    int K, int batch)
{{
    int k = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= K || b >= batch) return;
    
    u32 p = primes[k];
    u64 mu = mus[k];
    constexpr int E = {E};
    int base = k * batch * E + b * E;
    
    // Load A and B into registers
{chr(10).join(reg_decls)}
    
    // Unrolled matrix multiplication
{chr(10).join(fma_ops)}
    
    // Store results
{chr(10).join(stores)}
}}
#endif

void rns_gemm_mod_m{m}(
    u32* C, const u32* A, const u32* B,
    const u32* d_primes, const u64* d_mus,
    int K, int batch)
{{
#ifdef RNS_HAS_GPU
    dim3 grid(K, (batch + 255) / 256);
    dim3 block(256);
    hipLaunchKernelGGL(k_gemm_mod_m{m}, grid, block, 0, 0,
                       C, A, B, d_primes, d_mus, K, batch);
#else
    // CPU fallback
    constexpr int m = {m};
    constexpr int E = m * m;
    for (int k = 0; k < K; ++k) {{
        u32 p = d_primes[k];
        u64 mu = d_mus[k];
        for (int b = 0; b < batch; ++b) {{
            int base = k * batch * E + b * E;
            for (int i = 0; i < m; ++i) {{
                for (int j = 0; j < m; ++j) {{
                    u32 acc = 0;
                    for (int t = 0; t < m; ++t) {{
                        acc = fma_mod(A[base + i*m + t], 
                                     B[base + t*m + j], 
                                     acc, p, mu);
                    }}
                    C[base + i*m + j] = acc;
                }}
            }}
        }}
    }}
#endif
}}

}} // namespace rns
'''


def generate_dispatcher(max_m: int) -> str:
    """Generate dispatcher that routes to specialized kernels."""
    cases = []
    for m in range(2, max_m + 1):
        cases.append(f"        case {m}: rns_gemm_mod_m{m}(C, A, B, d_primes, d_mus, K, batch); break;")
    
    return f'''// Auto-generated GEMM dispatcher
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#ifndef RNS_GEMM_DISPATCH_H
#define RNS_GEMM_DISPATCH_H

#include "rns/config.h"

namespace rns {{

// Forward declarations for all specialized kernels
{chr(10).join(f"void rns_gemm_mod_m{m}(u32* C, const u32* A, const u32* B, const u32* d_primes, const u64* d_mus, int K, int batch);" for m in range(2, max_m + 1))}

// Dispatcher function
inline void rns_gemm_dispatch(
    u32* C, const u32* A, const u32* B,
    const u32* d_primes, const u64* d_mus,
    int K, int batch, int m)
{{
    switch (m) {{
{chr(10).join(cases)}
        default:
            // Generic fallback for unsupported sizes
            break;
    }}
}}

}} // namespace rns

#endif // RNS_GEMM_DISPATCH_H
'''


def generate_header(max_m: int) -> str:
    """Generate header with all kernel declarations."""
    decls = []
    for m in range(2, max_m + 1):
        decls.append(
            f"void rns_gemm_mod_m{m}(u32* C, const u32* A, const u32* B, "
            f"const u32* d_primes, const u64* d_mus, int K, int batch);"
        )
    
    return f'''// Auto-generated GEMM kernel declarations
// DO NOT EDIT - regenerate with gen_gemm_kernels.py

#ifndef RNS_GEMM_GENERATED_H
#define RNS_GEMM_GENERATED_H

#include "rns/config.h"

namespace rns {{

constexpr int GEMM_MAX_M = {max_m};
constexpr int GEMM_MIN_M = 2;

// Specialized kernel declarations
{chr(10).join(decls)}

// Check if m has a specialized kernel
inline bool has_specialized_gemm(int m) {{
    return m >= GEMM_MIN_M && m <= GEMM_MAX_M;
}}

}} // namespace rns

#endif // RNS_GEMM_GENERATED_H
'''


def main():
    parser = argparse.ArgumentParser(description='Generate GEMM kernels')
    parser.add_argument('--max-m', type=int, default=12, 
                        help='Maximum matrix size (default: 12)')
    parser.add_argument('--output-dir', type=str, default='generated',
                        help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating GEMM kernels for m=2..{args.max_m}")
    
    # Generate individual kernel files
    for m in range(2, args.max_m + 1):
        source = generate_kernel_source(m)
        filename = output_dir / f"rns_gemm_m{m}.hip"
        with open(filename, 'w') as f:
            f.write(source)
        print(f"  Generated {filename}")
    
    # Generate header
    header = generate_header(args.max_m)
    header_file = output_dir / "rns_gemm_generated.h"
    with open(header_file, 'w') as f:
        f.write(header)
    print(f"  Generated {header_file}")
    
    # Generate dispatcher
    dispatcher = generate_dispatcher(args.max_m)
    dispatcher_file = output_dir / "rns_gemm_dispatch.h"
    with open(dispatcher_file, 'w') as f:
        f.write(dispatcher)
    print(f"  Generated {dispatcher_file}")
    
    print(f"\nGenerated {args.max_m - 1} kernel files + 2 headers")
    print(f"To use: include generated/*.hip in your build")


if __name__ == '__main__':
    main()

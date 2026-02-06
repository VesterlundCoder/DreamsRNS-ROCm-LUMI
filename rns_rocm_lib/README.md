# RNS-ROCm - Residue Number System Library For Matrix Multiplications on ROCm/HIP based GPU

A high-performance ROCm/HIP C++ library for exact modular arithmetic using the **Residue Number System (RNS)**, focused on batched small matrix multiplications and CRT reconstruction.

## What is RNS?

The Residue Number System represents large integers as a tuple of residues modulo a set of coprime moduli. This enables **exact** arithmetic on very large integers using only 32-bit operations per prime, with results reconstructed via the Chinese Remainder Theorem (CRT). RNS is ideal for GPU computation because each prime lane is independent—enabling massive parallelism.

## Features

- **Batched modular GEMM** for small matrices (m = 4, 6, 8, 10) with specialized unrolled kernels
- **Barrett reduction** for fast modular multiplication without division
- **CRT reconstruction** using Garner's algorithm for reconstructing integers from residues
- **Bytecode evaluator** for GPU-side matrix expression evaluation
- **Fused walk kernel** for CMF-style iterated matrix products with shadow float scoring
- **TopK reduction** for candidate selection
- **CPU reference implementations** for correctness testing
- **Flexible prime generation** with verified primality
- **SoA data layout** optimized for GPU memory access: `[K][BATCH*m*m]`

## Data Layout

Arrays are stored in Structure-of-Arrays (SoA) format:
```
data[k * (B * E) + b * E + (i * m + j)]
```
Where:
- `K` = number of primes
- `B` = batch size (number of matrices per prime)
- `E` = m × m (elements per matrix)
- `k` = prime index
- `b` = batch index
- `i, j` = matrix row/column

## Building

### Prerequisites

- CMake 3.21+
- C++17 compiler
- ROCm 5.0+ (for GPU support)
- (Optional) Boost for extended bigint support

### CPU-only build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRNS_ENABLE_GPU=OFF
cmake --build . -j
```

### GPU build (ROCm)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Running tests

```bash
cd build
ctest --output-on-failure
```

Or run individual tests:
```bash
./test_rns_ops       # Modular arithmetic
./test_crt           # CRT reconstruction
./test_crt_matrix    # Matrix CRT
./test_primes        # Prime generation
./test_modops        # Barrett reduction
./test_eval_bytecode # Bytecode evaluator
./test_walk          # Fused walk kernel
./test_topk          # TopK reduction
./test_gemm_mod      # GPU GEMM (requires GPU)
```

### Running demos

```bash
./demo_gemm          # GPU GEMM benchmark (requires GPU)
./demo_crt_matrix    # CRT encoding/decoding demo
./demo_cmf_pipeline  # End-to-end CMF walk pipeline
```

## API Overview

### Device Context

```cpp
#include "rns/rns_api.h"

auto primes = rns::get_standard_primes_31bit(K);
auto ctx = rns::create_context(primes);

// ... use ctx ...

rns::destroy_context(ctx);
```

### Batched GEMM

```cpp
// Allocate device buffers
uint32_t* dA = rns::device_alloc_u32(K * B * m * m);
uint32_t* dB = rns::device_alloc_u32(K * B * m * m);
uint32_t* dC = rns::device_alloc_u32(K * B * m * m);

// Upload data
rns::h2d_u32(dA, hostA, K * B * m * m);
rns::h2d_u32(dB, hostB, K * B * m * m);

// Matrix multiply: C = A @ B (mod p) for all primes and batches
rns::gemm_mod(ctx, dC, dA, dB, B, m);
rns::device_sync();

// Download results
rns::d2h_u32(hostC, dC, K * B * m * m);
```

### CRT Reconstruction

```cpp
#include "rns/crt.h"

auto primes = rns::get_standard_primes_31bit(16);
auto plan = rns::CrtPlan::create(primes);

// Encode a BigInt to residues
rns::BigInt x("123456789012345678901234567890");
std::vector<uint32_t> residues(plan.K);
rns::rns_encode(x, plan, residues.data());

// Decode back
rns::BigInt y = rns::crt_reconstruct(residues.data(), plan);
assert(y == x % plan.M_total);
```

## Integer Range / Capacity

With K primes of ~31 bits each, the total capacity is approximately:
- **K=8**: ~248 bits
- **K=16**: ~496 bits  
- **K=32**: ~992 bits
- **K=64**: ~1984 bits

Ensure your intermediate results stay within capacity to avoid wraparound.

## Correctness Guarantees

- All operations are **exact** (no floating-point approximation)
- Results are correct modulo M = ∏ p_k
- CRT reconstruction is unique for values in [0, M)
- All primes are verified for primality and distinctness

## Architecture

```
rocm-rns-matmul/
├── include/rns/
│   ├── config.h        # Types and compile-time config
│   ├── modops.h        # PrimeMeta, modular arithmetic
│   ├── barrett.h       # Barrett reduction
│   ├── montgomery.h    # Montgomery (stub for future)
│   ├── rns_tensor.h    # SoA tensor views
│   ├── rns_matmul.h    # Batched GEMM declarations
│   ├── rns_eval.h      # Bytecode evaluator
│   ├── rns_walk.h      # Fused walk kernel
│   ├── topk.h          # TopK reduction
│   ├── crt.h           # CRT reconstruction
│   ├── io.h            # JSONL/CSV output
│   └── ...             # Legacy headers
├── src/
│   ├── hip/            # HIP/GPU kernels
│   │   ├── rns_matmul_m{4,6,8,10}.hip
│   │   ├── rns_eval_bytecode.hip
│   │   ├── rns_walk_fused.hip
│   │   └── topk.hip
│   ├── rns_eval_cpu.cpp
│   ├── rns_walk_cpu.cpp
│   ├── topk_cpu.cpp
│   └── ...
├── tests/              # Unit tests (8 test suites)
└── examples/
    ├── demo_gemm.cpp
    ├── demo_crt_matrix.cpp
    └── demo_cmf_pipeline.cpp
```

## Bytecode Evaluator

The bytecode evaluator allows dynamic matrix construction from expressions:

```cpp
#include "rns/rns_eval.h"

// Available opcodes:
// OP_LOAD_X, OP_LOAD_C, OP_ADD, OP_SUB, OP_MUL,
// OP_NEG, OP_POW2, OP_POW3, OP_INV, OP_MULINV

std::vector<Instr> instrs = {
  {OP_LOAD_X, 0, 0, 0},   // r0 = x[0]
  {OP_LOAD_C, 1, 0, 0},   // r1 = const[0]
  {OP_ADD, 2, 0, 1},      // r2 = r0 + r1
};

Program prog;
prog.instr = instrs.data();
// ...
eval_program_to_matrix(prog, x_vals, out_matrix, alive, pm, K, B);
```

## Python Bindings

The library includes optional Python bindings via pybind11.

### Installation

```bash
# From source (CPU-only)
cd python
pip install .

# With GPU support
RNS_ENABLE_GPU=ON pip install .

# Development mode
pip install -e .
```

### Python Usage

```python
import numpy as np
from rns import (
    generate_primes, mul_mod, inv_mod,
    Instruction, Program, OpCodes,
    run_walk, select_topk
)

# Generate primes
primes = np.array(generate_primes(32), dtype=np.uint32)

# Define a program (step matrix)
instrs = [
    Instruction(OpCodes.LOAD_X, 0, 0),     # r0 = x[0]
    Instruction(OpCodes.LOAD_C, 1, 0),     # r1 = const[0] = 1
    Instruction(OpCodes.ADD, 2, 0, 1),     # r2 = x[0] + 1
]
const_values = np.ones((32, 1), dtype=np.uint32)
out_reg = [2, 0, 0, 2]  # 2x2 matrix [[x+1, x], [x, x+1]]

prog = Program(m=2, dim=1, instructions=instrs,
               out_reg=out_reg, n_reg=3, const_values=const_values)

# Run walk
shifts = np.random.randint(-1000, 1000, (100, 1), dtype=np.int32)
dirs = np.ones(1, dtype=np.int32)

result = run_walk(prog, shifts, dirs, primes,
                  depth=200, depth1=50, depth2=200)

# Select best candidates
topk = select_topk(result['delta2'], result['est2'], k=10, ascending=True)
print(f"Best indices: {topk['indices']}")
```

### Pure Python Fallback

If the native module is not available, the library falls back to pure Python implementations. Check `rns.HAS_NATIVE` to see which mode is active.

## New in v0.3.0

### Montgomery Multiplication
Fast modular multiplication using Montgomery reduction:
```cpp
#include "rns/montgomery.h"
auto mp = compute_montgomery_params_ext(prime);
u32 aR = to_mont(a, mp.r2, mp.p, mp.pinv);
u32 bR = to_mont(b, mp.r2, mp.p, mp.pinv);
u32 cR = mont_mul(aR, bR, mp.p, mp.pinv);
u32 c = from_mont(cR, mp.p, mp.pinv);  // c = a*b mod p
```

### SymPy Bytecode Compiler
Compile SymPy matrix expressions to GPU bytecode:
```python
import sympy as sp
from rns import compile_matrix, disassemble

x0, x1 = sp.symbols('x0 x1')
M = sp.Matrix([[x0 + 1, x1], [1, x0 * x1]])
prog = compile_matrix(M, [x0, x1], K=32)
print(disassemble(prog))
```

### K_small CRT Scoring
Approximate magnitudes using partial CRT (avoids float overflow):
```cpp
#include "rns/crt_approx.h"
auto plan = create_crt_approx_plan(pm, K_small=3);
double log_mag = crt_approx_log_magnitude(residues, plan);
double ratio = crt_approx_ratio(residues_a, residues_b, plan);
```

### Auto-Generated Kernels
Generate optimized GEMM kernels for any matrix size:
```bash
python scripts/gen_gemm_kernels.py --max-m 20 --output-dir src/generated
```

### MPI Multi-GPU Support
Distributed walks across multiple GPUs:
```cpp
#include "rns/mpi_support.h"
MpiConfig cfg;
init_mpi_rns(cfg);
auto result = run_distributed_walk(cfg, walk_cfg, prog, shifts, dirs, pm, B, Kkeep);
```

## Future Roadmap

1. **GPU-native walk kernel**: Full walk on GPU without CPU roundtrip
2. **Streaming pipeline**: Overlap compute and I/O
3. **NTT/FFT support**: For polynomial arithmetic

## Comparison with CUDA Ecosystem

| Feature | CUDA | RNS-ROCm |
|---------|------|----------|
| Big integers | CGBN library | RNS + CRT |
| Modular reduction | Various | Barrett (Montgomery planned) |
| NTT/FFT | cuFFT | Not yet (future) |
| Fused kernels | Manual | walk_fused kernel |

## License

MIT License - see LICENSE file.

# RNS-ROCm Roadmap: Implementation Details

This document provides detailed implementation plans for each roadmap item.

---

## 1. Montgomery Backend

### Overview
Montgomery multiplication is an efficient algorithm for modular multiplication that avoids expensive division operations. Unlike Barrett reduction (which we currently use), Montgomery works in a transformed "Montgomery domain" where multiplication is faster.

### Theory

For a prime `p`, Montgomery multiplication works as follows:

1. **Choose R**: R = 2^32 (power of 2 > p)
2. **Precompute**:
   - `R mod p` (for converting to Montgomery form)
   - `R² mod p` (for efficient conversion)
   - `p' = -p⁻¹ mod R` (Montgomery constant)

3. **Montgomery Representation**: 
   - To convert `a` to Montgomery form: `aR mod p`
   - To convert back: multiply by 1 in Montgomery domain

4. **Montgomery Multiplication**:
   ```
   MonMul(aR, bR) = abR mod p
   ```
   
   Algorithm (REDC):
   ```
   function REDC(T):
       m = (T * p') mod R      # Low bits only, no division
       t = (T + m*p) / R       # Exact division by R (right shift)
       if t >= p: t = t - p
       return t
   ```

### Implementation Plan

**Files to modify/create**:
- `include/rns/montgomery.h` - Complete the stub
- `include/rns/modops.h` - Add Montgomery variants
- `src/hip/montgomery.hip` - GPU kernels

**Data structures**:
```cpp
struct MontgomeryParams {
    u32 p;          // Prime
    u32 p_inv;      // -p^(-1) mod 2^32
    u32 R_mod_p;    // 2^32 mod p
    u32 R2_mod_p;   // 2^64 mod p (for fast conversion)
};
```

**Key functions**:
```cpp
// Precomputation (CPU, once per prime)
MontgomeryParams compute_montgomery_params(u32 p);

// Convert to/from Montgomery domain
RNS_HOST_DEVICE u32 to_mont(u32 a, MontgomeryParams params);
RNS_HOST_DEVICE u32 from_mont(u32 aR, MontgomeryParams params);

// Montgomery multiplication
RNS_HOST_DEVICE u32 mont_mul(u32 aR, u32 bR, MontgomeryParams params);

// Montgomery reduction (REDC)
RNS_HOST_DEVICE u32 mont_reduce(u64 T, MontgomeryParams params);
```

**Performance considerations**:
- Montgomery is faster when doing many multiplications on the same values
- Conversion overhead means it's best for sequences (like matrix walks)
- For single multiplications, Barrett may still be faster
- Consider hybrid: use Montgomery for walk kernels, Barrett for one-offs

**Testing**:
- Verify `from_mont(to_mont(a)) == a` for all a < p
- Verify `from_mont(mont_mul(to_mont(a), to_mont(b))) == (a*b) mod p`
- Benchmark vs Barrett on walk kernel

---

## 2. SymPy to Bytecode Compiler

### Overview
Compile SymPy matrix expressions into our bytecode format for GPU evaluation.

### Input Format
```python
import sympy as sp

x0, x1 = sp.symbols('x0 x1')
M = sp.Matrix([
    [x0 + 1, x1],
    [1, x0 * x1]
])
```

### Output Format
Our `Program` struct with:
- List of `Instr` (op, dst, a, b)
- Output register mapping
- Constant table

### Compilation Algorithm

1. **Expression DAG Construction**:
   - Parse SymPy matrix
   - Build directed acyclic graph of operations
   - Identify common subexpressions (CSE)

2. **Register Allocation**:
   - Topological sort of DAG
   - Linear scan register allocation
   - Minimize register count

3. **Instruction Generation**:
   - Map SymPy operations to opcodes:
     - `Add` → `OP_ADD`
     - `Mul` → `OP_MUL`
     - `Pow(x, -1)` → `OP_INV`
     - `Pow(x, 2)` → `OP_POW2`
     - `Integer` → `OP_LOAD_C`
     - `Symbol` → `OP_LOAD_X`
   - Handle rational coefficients: `a/b` → `a * inv(b)`

4. **Constant Extraction**:
   - Collect all integer constants
   - Build constant table (per-prime reduction done at runtime)

### Implementation Plan

**New file**: `python/rns/sympy_compiler.py`

```python
from sympy import Matrix, symbols, Add, Mul, Pow, Integer, Symbol
from rns import Instruction, Program, OpCodes

class BytecodeCompiler:
    def __init__(self, matrix: Matrix, var_symbols: list):
        self.matrix = matrix
        self.symbols = var_symbols
        self.instructions = []
        self.constants = []
        self.reg_map = {}  # expr -> register
        self.next_reg = 0
    
    def compile(self) -> Program:
        # 1. CSE optimization
        replacements, reduced = sympy.cse(self.matrix)
        
        # 2. Allocate symbol registers
        for i, sym in enumerate(self.symbols):
            self._emit(OpCodes.LOAD_X, sym, a=i)
        
        # 3. Compile CSE replacements
        for var, expr in replacements:
            self._compile_expr(expr)
            self.reg_map[var] = self.reg_map[expr]
        
        # 4. Compile matrix entries
        out_reg = []
        for entry in reduced[0]:
            self._compile_expr(entry)
            out_reg.append(self.reg_map[entry])
        
        return Program(...)
    
    def _compile_expr(self, expr):
        if expr in self.reg_map:
            return  # Already computed
        
        if isinstance(expr, Symbol):
            # Should already be loaded
            pass
        elif isinstance(expr, Integer):
            idx = self._add_constant(int(expr))
            self._emit(OpCodes.LOAD_C, expr, a=idx)
        elif isinstance(expr, Add):
            # Compile operands first
            for arg in expr.args:
                self._compile_expr(arg)
            # Chain additions
            result = self.reg_map[expr.args[0]]
            for arg in expr.args[1:]:
                new_reg = self._alloc_reg()
                self.instructions.append(
                    Instruction(OpCodes.ADD, new_reg, result, self.reg_map[arg]))
                result = new_reg
            self.reg_map[expr] = result
        # ... similar for Mul, Pow, etc.
```

### Testing
- Compile known CMF matrices
- Verify output matches hand-coded programs
- Round-trip test: compile → eval → compare to SymPy.subs()

---

## 3. K_small CRT Scoring

### Overview
Replace shadow float (which overflows) with partial CRT reconstruction using a small subset of primes to get an approximate magnitude.

### Problem with Shadow Float
After ~50-100 matrix multiplications, float64 values overflow to inf/nan, making delta scores useless.

### Solution: Partial CRT
Use the first K_small (e.g., 3-4) primes to reconstruct a reduced integer, then convert to float.

### Algorithm

```cpp
// Given residues r[0..K-1] for K primes
// Use first K_small primes for approximate reconstruction

float approx_from_crt(const u32* residues, const PrimeMeta* pm, int K_small) {
    // Product of first K_small primes
    // M_small ~ 2^(31*K_small) ~ 2^93 for K_small=3
    
    // Garner's algorithm for K_small primes
    BigInt x = crt_reconstruct_partial(residues, pm, K_small);
    
    // Convert to float (may lose precision but won't overflow)
    return bigint_to_float(x);
}
```

### Implementation Plan

**Modifications to `rns_walk.h`**:
```cpp
struct WalkConfig {
    // ... existing fields ...
    int K_small;  // Number of primes for scoring (default 3)
};
```

**New functions in `crt.h`**:
```cpp
// Partial CRT reconstruction
BigInt crt_reconstruct_partial(const u32* residues, const CrtPlan& plan, int K_use);

// Convert BigInt to double (approximate)
double bigint_to_double(const BigInt& x);
```

**GPU implementation**:
- For GPU scoring, can't easily do full CRT
- Option 1: Copy K_small residues to CPU, do CRT there
- Option 2: Precompute M_small⁻¹ mod p_i for mixed-radix conversion
- Option 3: Use floating-point approximation of CRT (less accurate)

### Testing
- Compare K_small CRT score vs exact ratio from full CRT
- Verify ranking correlation (Kendall's tau) is high
- Benchmark overhead of partial CRT

---

## 4. MPI Multi-GPU

### Overview
Scale to multiple GPUs using MPI with one rank per GPU.

### Architecture
```
Rank 0 (GPU 0)          Rank 1 (GPU 1)          Rank 2 (GPU 2)
     |                       |                       |
  shifts[0:B/3]          shifts[B/3:2B/3]        shifts[2B/3:B]
     |                       |                       |
  walk_fused()           walk_fused()            walk_fused()
     |                       |                       |
  local_topk              local_topk              local_topk
     |                       |                       |
     +----------MPI_Gather----------+
                    |
              global_topk (rank 0)
                    |
              write results
```

### Implementation Plan

**New file**: `src/mpi/rns_mpi.cpp`

```cpp
#include <mpi.h>
#include "rns/rns_walk.h"
#include "rns/topk.h"

struct MpiConfig {
    int world_size;
    int rank;
    int gpu_id;           // Usually = rank
    std::string log_dir;  // Per-rank logging
};

void init_mpi_rns(MpiConfig& cfg) {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg.world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &cfg.rank);
    
    // Assign GPU to rank
    cfg.gpu_id = cfg.rank;
    hipSetDevice(cfg.gpu_id);
}

void run_distributed_walk(
    const MpiConfig& cfg,
    const WalkConfig& walk_cfg,
    const Program& prog,
    const i32* all_shifts,  // Full shift array (only rank 0 has complete)
    const i32* dirs,
    const PrimeMeta* pm,
    int total_B,
    int Kkeep,
    TopKItem* global_topk)
{
    // Calculate local batch
    int local_B = total_B / cfg.world_size;
    int start_idx = cfg.rank * local_B;
    
    // Scatter shifts
    std::vector<i32> local_shifts(local_B * walk_cfg.dim);
    MPI_Scatter(all_shifts, local_B * walk_cfg.dim, MPI_INT,
                local_shifts.data(), local_B * walk_cfg.dim, MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Local walk
    WalkOutputs local_out = /* allocate */;
    walk_fused(walk_cfg, prog, local_shifts.data(), dirs, pm, local_out);
    
    // Local topk
    std::vector<TopKItem> local_topk(Kkeep);
    topk_reduce(local_out.delta2, local_out.est2, local_topk.data(), 
                {local_B, Kkeep, true});
    
    // Adjust indices to global
    for (auto& item : local_topk) {
        item.shift_idx += start_idx;
    }
    
    // Gather all local topk to rank 0
    std::vector<TopKItem> all_topk;
    if (cfg.rank == 0) {
        all_topk.resize(cfg.world_size * Kkeep);
    }
    MPI_Gather(local_topk.data(), Kkeep * sizeof(TopKItem), MPI_BYTE,
               all_topk.data(), Kkeep * sizeof(TopKItem), MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    // Final topk selection on rank 0
    if (cfg.rank == 0) {
        // Sort all_topk and take top Kkeep
        std::partial_sort(all_topk.begin(), all_topk.begin() + Kkeep, 
                         all_topk.end(), /* comparator */);
        std::copy_n(all_topk.begin(), Kkeep, global_topk);
    }
}
```

**CMake integration**:
```cmake
option(RNS_ENABLE_MPI "Enable MPI support" OFF)

if(RNS_ENABLE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(rns_rocm_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(rns_rocm_lib PUBLIC RNS_HAS_MPI)
endif()
```

**Per-rank logging**:
```cpp
std::string get_rank_log_path(const MpiConfig& cfg, const std::string& base) {
    return cfg.log_dir + "/" + base + "_rank" + std::to_string(cfg.rank) + ".jsonl";
}
```

### Testing
- Single-node multi-GPU test
- Verify results match single-GPU run
- Benchmark scaling efficiency

---

## 5. Auto-Generated Matrix Kernels

### Overview
Generate specialized GEMM kernels for any matrix size m, not just 4, 6, 8, 10.

### Approach: Template Metaprogramming + Code Generation

**Option A: C++ Templates**
```cpp
template<int M>
__global__ void k_gemm_mod_template(
    u32* C, const u32* A, const u32* B,
    const u32* primes, const u64* mus,
    int K, int batch)
{
    int k = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    
    u32 p = primes[k];
    u64 mu = mus[k];
    
    int base_A = k * batch * M * M + b * M * M;
    int base_B = k * batch * M * M + b * M * M;
    int base_C = k * batch * M * M + b * M * M;
    
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        #pragma unroll
        for (int j = 0; j < M; ++j) {
            u32 acc = 0;
            #pragma unroll
            for (int t = 0; t < M; ++t) {
                acc = fma_mod(A[base_A + i*M + t], 
                             B[base_B + t*M + j], 
                             acc, p, mu);
            }
            C[base_C + i*M + j] = acc;
        }
    }
}

// Explicit instantiations
template __global__ void k_gemm_mod_template<2>(...);
template __global__ void k_gemm_mod_template<3>(...);
// ... up to M=20
```

**Option B: Python Code Generator**

```python
def generate_gemm_kernel(m: int) -> str:
    """Generate HIP kernel source for m×m GEMM."""
    return f'''
__global__ void k_gemm_mod_m{m}(
    u32* C, const u32* A, const u32* B,
    const u32* primes, const u64* mus,
    int K, int batch)
{{
    int k = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    
    u32 p = primes[k];
    u64 mu = mus[k];
    int E = {m * m};
    int base = k * batch * E + b * E;
    
    // Unrolled loops
{generate_unrolled_loops(m)}
}}
'''

def generate_unrolled_loops(m: int) -> str:
    lines = []
    for i in range(m):
        for j in range(m):
            lines.append(f"    u32 c{i}{j} = 0;")
    lines.append("")
    
    for i in range(m):
        for j in range(m):
            for t in range(m):
                lines.append(
                    f"    c{i}{j} = fma_mod(A[base + {i*m + t}], "
                    f"B[base + {t*m + j}], c{i}{j}, p, mu);")
    lines.append("")
    
    for i in range(m):
        for j in range(m):
            lines.append(f"    C[base + {i*m + j}] = c{i}{j};")
    
    return "\n".join(lines)
```

### Build Integration

**CMake with generated sources**:
```cmake
# Generate kernels at configure time
execute_process(
    COMMAND ${Python_EXECUTABLE} 
            ${CMAKE_SOURCE_DIR}/scripts/gen_gemm_kernels.py
            --max-m 20
            --output-dir ${CMAKE_BINARY_DIR}/generated
)

file(GLOB GENERATED_KERNELS ${CMAKE_BINARY_DIR}/generated/*.hip)
list(APPEND RNS_SOURCES ${GENERATED_KERNELS})
```

### Testing
- Generate kernels for m=2..20
- Compare output to generic kernel
- Benchmark speedup vs generic

---

## 6. GPU CRT

### Overview
Perform partial CRT reconstruction on GPU for fast approximate scoring.

### Challenge
Full CRT requires BigInt arithmetic, which is slow on GPU. Instead, use floating-point approximation.

### Algorithm: Floating-Point CRT Approximation

Given residues `r[0..K-1]` for primes `p[0..K-1]`:

```cpp
// Precompute (CPU, once)
// M_i = prod(p_j for j != i)
// y_i = M_i^(-1) mod p_i
// w_i = M_i * y_i (as float, normalized)

__device__ float approx_crt(const u32* residues, const float* weights, int K) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += (float)residues[i] * weights[i];
    }
    return sum;  // Approximate magnitude
}
```

### More Accurate: Log-Space CRT

```cpp
// Precompute log(M_i * y_i) for each i
// In log space, multiplication becomes addition

__device__ float log_approx_crt(
    const u32* residues, 
    const float* log_weights, 
    int K) 
{
    // Find max term for numerical stability
    float max_log = -INFINITY;
    for (int i = 0; i < K; ++i) {
        if (residues[i] > 0) {
            float log_term = logf((float)residues[i]) + log_weights[i];
            max_log = fmaxf(max_log, log_term);
        }
    }
    
    // Log-sum-exp
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        if (residues[i] > 0) {
            float log_term = logf((float)residues[i]) + log_weights[i];
            sum += expf(log_term - max_log);
        }
    }
    
    return max_log + logf(sum);  // log of approximate value
}
```

### Implementation Plan

**New header**: `include/rns/gpu_crt.h`

```cpp
struct GpuCrtPlan {
    int K;
    float* d_weights;      // Device: CRT weights
    float* d_log_weights;  // Device: log-space weights
};

GpuCrtPlan create_gpu_crt_plan(const PrimeMeta* pm, int K);
void destroy_gpu_crt_plan(GpuCrtPlan& plan);

// GPU kernel to compute approximate norms
void gpu_approx_norms(
    const u32* residue_matrices,  // [K, B, E]
    float* norms,                 // [B]
    const GpuCrtPlan& plan,
    int B, int E);
```

### Integration with Walk Kernel

```cpp
// In walk_fused, after computing P_final:
// Instead of shadow float, use GPU CRT approximation

// Option 1: Approximate matrix norm
gpu_approx_norms(P_final, est2, crt_plan, B, E);

// Option 2: Approximate specific entries for delta
// delta ≈ |P[0,1]| / |P[0,0]|
gpu_approx_ratio(P_final, delta2, crt_plan, B, m);
```

### Testing
- Compare GPU CRT approximation to exact CPU CRT
- Verify ranking correlation is sufficient
- Benchmark kernel performance

---

## Implementation Order

1. **Montgomery backend** - Foundation for faster modular arithmetic
2. **SymPy compiler** - Enables real CMF programs
3. **K_small CRT scoring** - Fixes the overflow problem
4. **Auto-generated kernels** - Flexibility for different matrix sizes
5. **GPU CRT** - Fast approximate scoring
6. **MPI multi-GPU** - Scaling to cluster

---

## Version Milestones

### v0.3.0
- Montgomery backend
- SymPy compiler (basic)
- K_small CRT scoring

### v0.4.0
- Auto-generated kernels (m up to 20)
- GPU CRT approximation
- Improved walk kernel

### v0.5.0
- MPI multi-GPU
- Production-ready CMF pipeline
- Comprehensive benchmarks

---

## References

- Montgomery, P. L. (1985). "Modular multiplication without trial division"
- Garner, H. L. (1959). "The residue number system"
- ROCm HIP Programming Guide: https://rocm.docs.amd.com/
- MPI Standard: https://www.mpi-forum.org/

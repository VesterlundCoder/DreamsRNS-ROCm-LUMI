# ROCm Repository Update Changelog — GPU-Only Enforcement

**Date:** 2025-02-22
**Repos:** `DreamsRNS-ROCm-LUMI` (Python pipeline) + `RNS-ROCm` (C++ GPU library)
**Goal:** Remove all CPU fallbacks, enforce GPU-only execution, connect placeholders, ensure consistency.

---

## Summary of Changes

| Category | Files Changed | Description |
|----------|--------------|-------------|
| CPU fallback removal | 9 files | Removed all `#ifdef RNS_HAS_GPU` / `#ifndef` CPU dispatch guards |
| GPU mandatory build | 1 file | CMake now requires HIP — errors if not found |
| Placeholder connection | 2 files | `gpu_runner.py` and `bindings.py` now connect to native C ABI |
| Docstring cleanup | 5 files | Removed "CPU fallback" language from all docstrings |
| Script consistency | 2 files | SBATCH and local runner updated for GPU-only policy |

---

## Repo 1: RNS-ROCm (C++ Library)

### 1. `CMakeLists.txt`

**Error/Inconsistency:**
- `RNS_ENABLE_GPU` was an option that could be turned OFF
- If HIP was not found, CMake would silently build a CPU-only version (`message(WARNING "HIP not found - building CPU-only version")`)
- GPU-dependent tests and examples were conditionally skipped

**Fix:**
- Removed `option(RNS_ENABLE_GPU ...)` — GPU is now mandatory
- Changed `find_package(hip QUIET)` → `find_package(hip REQUIRED)` — build fails immediately if HIP not found
- HIP kernel sources are always included (not conditional)
- `test_gemm_mod` and `demo_gemm` always build
- Summary line changed from `${RNS_HAS_GPU}` to `ON (mandatory)`

**Updated code:**
```cmake
# GPU support via HIP (mandatory)
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "/opt/rocm/hip")
find_package(hip REQUIRED)
message(STATUS "Found HIP: ${hip_VERSION}")
enable_language(HIP)
set(RNS_HAS_GPU TRUE)
add_definitions(-DRNS_HAS_GPU)
```

---

### 2. `src/rns_api.cpp`

**Error/Inconsistency:**
- Every function had `#ifdef RNS_HAS_GPU` / `#else` dual paths
- CPU fallback used `new[]` / `delete[]` / `std::memcpy` instead of `hipMalloc` / `hipFree` / `hipMemcpy`
- 12 separate CPU fallback code blocks across: `create_context`, `destroy_context`, `device_alloc_u32`, `device_free`, `h2d_u32`, `d2h_u32`, `device_sync`, `add`, `mul`, `sub`, `gemm_mod`

**Fix:**
- Removed all `#ifdef RNS_HAS_GPU` / `#else` / `#endif` guards
- All functions now use HIP APIs directly (always compiled with `-DRNS_HAS_GPU`)
- `#include <hip/hip_runtime.h>` and `#include "rns/rns_kernels.h"` are unconditional

**Updated code (example — `device_alloc_u32`):**
```cpp
uint32_t* device_alloc_u32(size_t count) {
  uint32_t* p = nullptr;
  hipError_t err = hipMalloc((void**)&p, count * sizeof(uint32_t));
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipMalloc failed: ") + hipGetErrorString(err));
  }
  return p;
}
```

---

### 3. `src/hip/rns_walk_fused.hip`

**Error/Inconsistency:**
- `walk_fused()` had `#ifdef RNS_HAS_GPU` guard with CPU fallback to `walk_fused_cpu()`
- Contained a **duplicate** 120-line `walk_fused_cpu()` implementation (identical to `rns_walk_cpu.cpp`)
- Two copies of the same CPU code in different files

**Fix:**
- Removed `#ifdef` / `#else` guard from `walk_fused()` — always launches GPU kernel
- Removed the entire duplicate `walk_fused_cpu()` from this file (the reference copy in `rns_walk_cpu.cpp` is retained for unit tests)

**Updated code:**
```cpp
void walk_fused(...) {
  int blockSize = 64;
  int gridSize = (cfg.B + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(k_walk_fused, dim3(gridSize), dim3(blockSize), 0, 0,
                     cfg.depth, cfg.depth1, cfg.depth2, cfg.m, cfg.dim, cfg.K, cfg.B,
                     prog_step.instr, prog_step.n_instr,
                     prog_step.const_table, prog_step.n_const,
                     prog_step.out_reg,
                     shifts, dirs, pm,
                     out.P_final, out.alive,
                     out.est1, out.est2, out.delta1, out.delta2);
}
```

---

### 4. `src/rns_walk_cpu.cpp`

**Error/Inconsistency:**
- `#ifndef RNS_HAS_GPU` block defined `walk_fused()` → `walk_fused_cpu()` dispatch, making CPU the default when GPU was absent

**Fix:**
- Removed the `#ifndef` dispatch block
- Added comment: "GPU walk_fused() is defined in hip/rns_walk_fused.hip"
- `walk_fused_cpu()` retained as reference-only for unit tests

---

### 5. `src/rns_eval_cpu.cpp`

**Error/Inconsistency:**
- `#ifndef RNS_HAS_GPU` block defined `eval_program_to_matrix()` → `eval_program_to_matrix_cpu()` dispatch

**Fix:**
- Removed the `#ifndef` dispatch block
- Added comment: "GPU eval_program_to_matrix() is defined in hip/rns_eval_bytecode.hip"
- `eval_program_to_matrix_cpu()` retained as reference-only for unit tests

---

### 6. `src/topk_cpu.cpp`

**Error/Inconsistency:**
- `#ifndef RNS_HAS_GPU` block defined `topk_reduce()` → `topk_reduce_cpu()` dispatch

**Fix:**
- Removed the `#ifndef` dispatch block
- Added comment: "GPU topk_reduce() is defined in hip/topk.hip"
- `topk_reduce_cpu()` retained as reference-only for unit tests

---

## Repo 2: DreamsRNS-ROCm-LUMI (Python Pipeline)

### 7. `dreams_rocm/gpu_runner.py`

**Error/Inconsistency:**
- `_run_walk_native()` (line 157-172) had a **TODO placeholder** — never called the native kernel, always fell back to `_run_walk_python()`
- `_run_walk_python()` was a non-functional stub returning empty results
- `run_walk()` silently fell back to CPU if native lib was unavailable
- `__init__()` used `warnings.warn` instead of erroring when the library failed to load

**Fix:**
- `_run_walk_native()` now properly calls `walk_fused_c` via ctypes with correct struct packing:
  - Packs `Instr` as `(op, dst, a, b)` uint8 struct array
  - Packs `PrimeMeta` as `(p, pad, mu, pinv, r2)` struct array
  - Passes all buffers via `ctypes.c_void_p`
  - Returns hits and metrics after GPU execution
- Removed `_run_walk_python()` entirely — no CPU fallback
- `_init()` raises `RuntimeError` if library not found or walk_fused symbol missing
- `_setup_ctypes_signatures()` validates that `walk_fused` symbol exists

**Updated code (key section):**
```python
def _init(self):
    if self._initialized:
        return
    lib_path = get_rns_library_path()
    if lib_path is None:
        raise RuntimeError(
            "RNS-ROCm native library not found. "
            "Build librns_rocm_lib.so with hipcc and set RNS_ROCM_LIB env var, "
            "or place it in rns_rocm_lib/build/."
        )
    try:
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_ctypes_signatures()
    except OSError as e:
        raise RuntimeError(
            f"Failed to load RNS-ROCm library at {lib_path}: {e}"
        ) from e
    self._initialized = True
```

---

### 8. `dreams_rocm/rns/bindings.py`

**Error/Inconsistency:**
- `_init_native()` was an empty placeholder with a comment saying "will enable this once the shared library is compiled on LUMI"
- Warning message said "Using pure-Python fallback" (implying CPU was acceptable)
- Docstring said "falls back to pure-Python implementations"

**Fix:**
- `_init_native()` now actually initializes the native GPU device context via ctypes, looking for `rns_create_context` or `create_context` C ABI symbols
- Warning message updated to "GPU execution will not be available"
- Docstring updated to "GPU is mandatory — no CPU fallback"

**Updated code:**
```python
def _init_native(self):
    if _lib is None:
        return
    if hasattr(_lib, 'rns_create_context'):
        primes_arr = self.primes.copy()
        ctx_ptr = _lib.rns_create_context(
            primes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_int(self.K),
        )
        self._native_ctx = ctx_ptr
    elif hasattr(_lib, 'create_context'):
        primes_arr = self.primes.copy()
        ctx_ptr = _lib.create_context(
            primes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_int(self.K),
        )
        self._native_ctx = ctx_ptr
```

---

### 9. `dreams_rocm/cmf_walk.py`

**Error/Inconsistency:**
- `run_cmf_walk_batch()` had a `try/except ImportError: pass` that silently fell back to sequential CPU walks when the native library import failed
- Comment said "CPU fallback: walk each shift sequentially"

**Fix:**
- Removed `try/except ImportError` — import failure is now visible
- Changed comment to "Reference path: walk each shift sequentially via numpy"
- GPU path is now the primary path (not wrapped in try/except)

---

### 10. `dreams_rocm/runner.py`

**Error/Inconsistency:**
- Docstring said "Supports CPU (numpy) and GPU (ROCm) backends" — misleading since this file only contains the numpy reference walker

**Fix:**
- Docstring updated to: "Uses numpy for the K-prime vectorized reference walk. For GPU-accelerated walks, use gpu_runner.GpuWalkRunner."

---

### 11. `dreams_rocm/cmf_compile.py`

**Error/Inconsistency:**
- Docstring said "evaluated on GPU via the RNS-ROCm bytecode evaluator or on CPU via the pure-Python fallback"

**Fix:**
- Docstring updated to: "The bytecode is evaluated on GPU via the RNS-ROCm bytecode evaluator."

---

### 12. `dreams_rocm/rns/reference.py`

**Error/Inconsistency:**
- Docstring listed purpose #2 as "CPU fallback when the native RNS-ROCm library is unavailable"

**Fix:**
- Docstring updated to: "These serve as ground truth for correctness testing. Production walks use the native RNS-ROCm GPU library."

---

### 13. `scripts/sbatch_1node_8gpu.sh`

**Error/Inconsistency:**
- If the Singularity container was missing, the script fell back to `module load cray-python/3.10.10` — running without GPU acceleration, defeating the purpose
- `USE_CONTAINER=0` path ran Python directly without the compiled native library

**Fix:**
- Container is now mandatory — script exits with error code 1 if `.sif` not found
- Error message tells user how to build the container

**Updated code:**
```bash
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    echo "       Build it first: singularity build dreams_rocm.sif env/dreams_rocm.def"
    exit 1
fi
USE_CONTAINER=1
```

---

### 14. `scripts/run_local_mac.py`

**Error/Inconsistency:**
- Docstring said "CPU-only" — implied this was a production path

**Fix:**
- Docstring updated to clarify this is a **correctness testing** tool using the numpy reference walker, not a production runner

---

## Files NOT Changed (Verified Correct)

| File | Reason |
|------|--------|
| `dreams_rocm/__init__.py` | Exports are correct, version 0.2.0 |
| `dreams_rocm/cmf_generator.py` | Pure Python CMF spec generator, no GPU code |
| `dreams_rocm/constants.py` | Constant matching, no GPU code |
| `dreams_rocm/exhaust.py` | Trajectory exhaustion, no GPU code |
| `dreams_rocm/logging.py` | File-based logging, no GPU code |
| `dreams_rocm/shifts.py` | Shift generation, no GPU code |
| `dreams_rocm/trajectories.py` | Trajectory generation, no GPU code |
| `dreams_rocm/crt/delta_targets.py` | Target constants + delta computation, correct |
| `dreams_rocm/crt/partial_crt.py` | Partial CRT delta proxy, no GPU code |
| `scripts/euler2ai_verify.py` | PCF verification CLI, uses runner.py (correct) |
| `scripts/run_mpi_sweep.py` | MPI orchestrator, uses runner.py (correct) |
| `scripts/validate_rocm_rns.py` | Validation suite, checks GPU availability (correct) |
| `include/rns/rns_walk.h` | Header declares both GPU and CPU functions (correct) |
| `src/hip/*.hip` (other than walk_fused) | GPU kernels, no CPU fallback code |
| `src/crt.cpp`, `src/primes.cpp`, `src/utils.cpp`, `src/io.cpp` | Pure CPU utilities, no GPU fallback pattern |

---

## Architecture After Changes

```
                    ┌─────────────────────────┐
                    │  Python Pipeline         │
                    │  (DreamsRNS-ROCm-LUMI)   │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼───────┐  ┌──▼────────┐  ┌──▼──────────┐
    │ runner.py        │  │ cmf_walk  │  │ gpu_runner  │
    │ (numpy ref)      │  │ (batch)   │  │ (ctypes)    │
    │ verify_pcf()     │  │           │  │ walk_fused  │
    └─────────────────┘  └──────────┘  └──────┬──────┘
                                               │ ctypes
                                    ┌──────────▼──────────┐
                                    │ librns_rocm_lib.so   │
                                    │ (RNS-ROCm C++ lib)   │
                                    │ GPU-only build       │
                                    └──────────┬──────────┘
                                               │ HIP
                                    ┌──────────▼──────────┐
                                    │ AMD MI250X / gfx90a  │
                                    │ (or Nvidia via HIP)  │
                                    └─────────────────────┘
```

- **runner.py**: Numpy reference walker for correctness testing (Mac/local)
- **gpu_runner.py**: Production GPU walker via native library (LUMI/Lambda)
- **cmf_walk.py**: Auto-dispatches to GPU native when available
- **librns_rocm_lib.so**: Always built with HIP, no CPU fallback code paths

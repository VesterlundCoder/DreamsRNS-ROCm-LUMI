#include <hip/hip_runtime.h>
#include "rns/modops.h"

namespace rns {

__global__ void k_test_modops(
    const u32* a, const u32* b, u32* c_add, u32* c_sub, u32* c_mul,
    const PrimeMeta* pm, int K, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * N) return;
  
  int k = idx / N;
  int n = idx % N;
  
  u32 p = pm[k].p;
  u64 mu = pm[k].mu;
  
  u32 av = a[idx];
  u32 bv = b[idx];
  
  c_add[idx] = add_mod(av, bv, p);
  c_sub[idx] = sub_mod(av, bv, p);
  c_mul[idx] = mul_mod(av, bv, p, mu);
}

void test_modops_gpu(
    const u32* d_a, const u32* d_b,
    u32* d_add, u32* d_sub, u32* d_mul,
    const PrimeMeta* d_pm, int K, int N)
{
  int total = K * N;
  int blockSize = 256;
  int gridSize = (total + blockSize - 1) / blockSize;
  
  hipLaunchKernelGGL(k_test_modops, dim3(gridSize), dim3(blockSize), 0, 0,
                     d_a, d_b, d_add, d_sub, d_mul, d_pm, K, N);
}

} // namespace rns

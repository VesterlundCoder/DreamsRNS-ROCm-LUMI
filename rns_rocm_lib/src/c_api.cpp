/**
 * C ABI wrapper for the RNS-ROCm library.
 *
 * Provides a flat C interface suitable for ctypes/cffi from Python.
 * All functions are prefixed with rns_ and use plain C types.
 *
 * This file is compiled into librns_rocm_lib.so alongside the rest
 * of the library.
 */

#include <cstdint>
#include <cstring>
#include <vector>

#include "rns/rns_api.h"
#include "rns/rns_ops.h"
#include "rns/rns_types.h"
#include "rns/crt.h"
#include "rns/crt_types.h"
#include "rns/rns_eval.h"
#include "rns/rns_walk.h"
#include "rns/topk.h"
#include "rns/primes.h"
#include "rns/utils.h"

extern "C" {

// ============================================================================
// Prime Generation
// ============================================================================

int rns_generate_primes(uint32_t* out_primes, int K) {
    try {
        auto primes = rns::generate_primes(K);
        for (int i = 0; i < K && i < (int)primes.size(); ++i) {
            out_primes[i] = primes[i];
        }
        return (int)primes.size();
    } catch (...) {
        return -1;
    }
}

// ============================================================================
// Device Context
// ============================================================================

struct RnsDeviceContext {
    rns::DeviceContext ctx;
    std::vector<uint32_t> primes;
    std::vector<rns::PrimeMeta> pm;
};

void* rns_create_context(const uint32_t* primes, int K) {
    try {
        auto* c = new RnsDeviceContext();
        c->primes.assign(primes, primes + K);
        c->ctx = rns::create_context(c->primes);

        // Build PrimeMeta array for eval/walk kernels
        c->pm.resize(K);
        for (int i = 0; i < K; ++i) {
            c->pm[i].p = primes[i];
            c->pm[i].pad = 0;
            c->pm[i].mu = rns::compute_barrett_mu(primes[i]);
            c->pm[i].pinv = 0;
            c->pm[i].r2 = 0;
        }
        return (void*)c;
    } catch (...) {
        return nullptr;
    }
}

void rns_destroy_context(void* ctx) {
    if (ctx) {
        auto* c = (RnsDeviceContext*)ctx;
        rns::destroy_context(c->ctx);
        delete c;
    }
}

// ============================================================================
// Modular Arithmetic (elementwise, on device arrays)
// ============================================================================

int rns_add_arrays(void* ctx, uint32_t* out,
                   const uint32_t* a, const uint32_t* b, int N) {
    try {
        auto* c = (RnsDeviceContext*)ctx;
        rns::add(c->ctx, out, a, b, N);
        return 0;
    } catch (...) {
        return -1;
    }
}

int rns_mul_arrays(void* ctx, uint32_t* out,
                   const uint32_t* a, const uint32_t* b, int N) {
    try {
        auto* c = (RnsDeviceContext*)ctx;
        rns::mul(c->ctx, out, a, b, N);
        return 0;
    } catch (...) {
        return -1;
    }
}

int rns_gemm(void* ctx, uint32_t* C,
             const uint32_t* A, const uint32_t* B,
             int batch, int m) {
    try {
        auto* c = (RnsDeviceContext*)ctx;
        rns::gemm_mod(c->ctx, C, A, B, batch, m);
        return 0;
    } catch (...) {
        return -1;
    }
}

// ============================================================================
// Walk Fused (main kernel)
// ============================================================================

int rns_walk_fused(
    void* ctx,
    int depth, int depth1, int depth2,
    int m, int dim, int B,
    const uint8_t* instr_ops, const uint8_t* instr_dsts,
    const uint8_t* instr_as, const uint8_t* instr_bs,
    int n_instr,
    const uint32_t* const_table, int n_const,
    const uint16_t* out_reg,
    const int32_t* shifts,
    const int32_t* dirs,
    uint32_t* P_final,
    uint8_t* alive,
    float* est1, float* est2,
    float* delta1, float* delta2)
{
    try {
        auto* c = (RnsDeviceContext*)ctx;
        int K = c->ctx.K;

        // Build Program struct from flat arrays
        std::vector<rns::Instr> instructions(n_instr);
        for (int i = 0; i < n_instr; ++i) {
            instructions[i].op = (rns::Op)instr_ops[i];
            instructions[i].dst = instr_dsts[i];
            instructions[i].a = instr_as[i];
            instructions[i].b = instr_bs[i];
        }

        rns::Program prog;
        prog.m = m;
        prog.dim = dim;
        prog.n_instr = n_instr;
        prog.n_reg = 64;  // MAX_REGISTERS
        prog.n_const = n_const;
        prog.instr = instructions.data();
        prog.const_table = const_table;
        prog.out_reg = out_reg;

        rns::WalkConfig wcfg;
        wcfg.depth = depth;
        wcfg.depth1 = depth1;
        wcfg.depth2 = depth2;
        wcfg.m = m;
        wcfg.dim = dim;
        wcfg.K = K;
        wcfg.B = B;

        rns::WalkOutputs out;
        out.P_final = P_final;
        out.alive = alive;
        out.est1 = est1;
        out.est2 = est2;
        out.delta1 = delta1;
        out.delta2 = delta2;

        rns::walk_fused(wcfg, prog, shifts, dirs, c->pm.data(), out);
        rns::device_sync();

        return 0;
    } catch (...) {
        return -1;
    }
}

// ============================================================================
// CRT Reconstruction
// ============================================================================

int rns_crt_reconstruct(
    const uint32_t* primes, int K,
    const uint32_t* residues,
    char* out_str, int out_str_len)
{
    try {
        std::vector<uint32_t> pvec(primes, primes + K);
        auto plan = rns::CrtPlan::create(pvec);
        rns::BigInt result = rns::crt_reconstruct(residues, plan);
        std::string s = result.to_string();
        int len = std::min((int)s.size(), out_str_len - 1);
        std::memcpy(out_str, s.c_str(), len);
        out_str[len] = '\0';
        return len;
    } catch (...) {
        return -1;
    }
}

int rns_crt_reconstruct_signed(
    const uint32_t* primes, int K,
    const uint32_t* residues,
    char* out_str, int out_str_len)
{
    try {
        std::vector<uint32_t> pvec(primes, primes + K);
        auto plan = rns::CrtPlan::create(pvec);
        rns::BigInt result = rns::crt_reconstruct_signed(residues, plan);
        std::string s = result.to_string();
        int len = std::min((int)s.size(), out_str_len - 1);
        std::memcpy(out_str, s.c_str(), len);
        out_str[len] = '\0';
        return len;
    } catch (...) {
        return -1;
    }
}

// ============================================================================
// Synchronization
// ============================================================================

void rns_device_sync() {
    rns::device_sync();
}

// ============================================================================
// Version / Info
// ============================================================================

const char* rns_version() {
    return "0.2.0-lumi";
}

int rns_has_gpu() {
#ifdef RNS_HAS_GPU
    return 1;
#else
    return 0;
#endif
}

}  // extern "C"

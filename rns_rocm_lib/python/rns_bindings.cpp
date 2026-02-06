#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

#include "rns/config.h"
#include "rns/modops.h"
#include "rns/barrett.h"
#include "rns/rns_eval.h"
#include "rns/rns_walk.h"
#include "rns/topk.h"
#include "rns/crt.h"
#include "rns/primes.h"

namespace py = pybind11;
using namespace rns;

// Helper to create PrimeMeta from prime values
std::vector<PrimeMeta> create_prime_meta(const std::vector<u32>& primes) {
    std::vector<PrimeMeta> meta(primes.size());
    for (size_t i = 0; i < primes.size(); ++i) {
        meta[i].p = primes[i];
        meta[i].pad = 0;
        meta[i].mu = compute_barrett_mu(primes[i]);
        meta[i].pinv = 0;
        meta[i].r2 = 0;
    }
    return meta;
}

// Generate K primes near 2^31
std::vector<u32> py_generate_primes(int K, u64 seed = 12345) {
    std::vector<u32> primes;
    u64 state = seed;
    
    auto next_rand = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (u32)(state >> 32);
    };
    
    auto is_prime = [](u32 n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (u32 i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    };
    
    while ((int)primes.size() < K) {
        u32 candidate = (next_rand() | (1u << 30)) | 1;
        if (candidate < (1u << 30)) candidate += (1u << 30);
        
        if (is_prime(candidate)) {
            bool unique = true;
            for (u32 p : primes) {
                if (p == candidate) { unique = false; break; }
            }
            if (unique) primes.push_back(candidate);
        }
    }
    return primes;
}

// Python wrapper for bytecode evaluation
py::tuple py_eval_program(
    int m, int dim, int n_reg,
    py::array_t<uint8_t> opcodes,
    py::array_t<uint8_t> dsts,
    py::array_t<uint8_t> as,
    py::array_t<uint8_t> bs,
    py::array_t<uint16_t> out_reg,
    py::array_t<u32> const_table,  // [K, n_const]
    py::array_t<u32> x_vals,       // [K, B, dim]
    py::array_t<u32> primes)
{
    auto op_buf = opcodes.request();
    auto dst_buf = dsts.request();
    auto a_buf = as.request();
    auto b_buf = bs.request();
    auto out_buf = out_reg.request();
    auto const_buf = const_table.request();
    auto x_buf = x_vals.request();
    auto p_buf = primes.request();
    
    int n_instr = (int)op_buf.shape[0];
    int n_const = (int)const_buf.shape[1];
    int K = (int)p_buf.shape[0];
    int B = (int)x_buf.shape[1];
    int E = m * m;
    
    // Build instructions
    std::vector<Instr> instrs(n_instr);
    auto* op_ptr = static_cast<uint8_t*>(op_buf.ptr);
    auto* dst_ptr = static_cast<uint8_t*>(dst_buf.ptr);
    auto* a_ptr = static_cast<uint8_t*>(a_buf.ptr);
    auto* b_ptr = static_cast<uint8_t*>(b_buf.ptr);
    
    for (int i = 0; i < n_instr; ++i) {
        instrs[i].op = static_cast<Op>(op_ptr[i]);
        instrs[i].dst = dst_ptr[i];
        instrs[i].a = a_ptr[i];
        instrs[i].b = b_ptr[i];
    }
    
    // Build program
    Program prog;
    prog.m = m;
    prog.dim = dim;
    prog.n_instr = n_instr;
    prog.n_reg = n_reg;
    prog.n_const = n_const;
    prog.instr = instrs.data();
    prog.const_table = static_cast<u32*>(const_buf.ptr);
    prog.out_reg = static_cast<uint16_t*>(out_buf.ptr);
    
    // Create prime meta
    auto* p_ptr = static_cast<u32*>(p_buf.ptr);
    std::vector<PrimeMeta> pm = create_prime_meta(
        std::vector<u32>(p_ptr, p_ptr + K));
    
    // Allocate outputs
    py::array_t<u32> out_matrix({K, B, E});
    py::array_t<uint8_t> alive({K, B});
    
    auto out_m_buf = out_matrix.request();
    auto alive_buf = alive.request();
    
    // Call CPU implementation
    eval_program_to_matrix_cpu(
        prog,
        static_cast<u32*>(x_buf.ptr),
        static_cast<u32*>(out_m_buf.ptr),
        static_cast<uint8_t*>(alive_buf.ptr),
        pm.data(),
        K, B);
    
    return py::make_tuple(out_matrix, alive);
}

// Python wrapper for walk_fused
py::dict py_walk_fused(
    int depth, int depth1, int depth2,
    int m, int dim, int K, int B,
    // Program definition
    py::array_t<uint8_t> opcodes,
    py::array_t<uint8_t> dsts,
    py::array_t<uint8_t> as,
    py::array_t<uint8_t> bs,
    py::array_t<uint16_t> out_reg,
    py::array_t<u32> const_table,
    int n_reg,
    // Inputs
    py::array_t<i32> shifts,      // [B, dim]
    py::array_t<i32> dirs,        // [dim]
    py::array_t<u32> primes)
{
    auto op_buf = opcodes.request();
    auto dst_buf = dsts.request();
    auto a_buf = as.request();
    auto b_buf = bs.request();
    auto out_buf = out_reg.request();
    auto const_buf = const_table.request();
    auto shift_buf = shifts.request();
    auto dir_buf = dirs.request();
    auto p_buf = primes.request();
    
    int n_instr = (int)op_buf.shape[0];
    int n_const = (int)const_buf.shape[1];
    int E = m * m;
    
    // Build instructions
    std::vector<Instr> instrs(n_instr);
    auto* op_ptr = static_cast<uint8_t*>(op_buf.ptr);
    auto* dst_ptr = static_cast<uint8_t*>(dst_buf.ptr);
    auto* a_ptr = static_cast<uint8_t*>(a_buf.ptr);
    auto* b_ptr = static_cast<uint8_t*>(b_buf.ptr);
    
    for (int i = 0; i < n_instr; ++i) {
        instrs[i].op = static_cast<Op>(op_ptr[i]);
        instrs[i].dst = dst_ptr[i];
        instrs[i].a = a_ptr[i];
        instrs[i].b = b_ptr[i];
    }
    
    // Build program
    Program prog;
    prog.m = m;
    prog.dim = dim;
    prog.n_instr = n_instr;
    prog.n_reg = n_reg;
    prog.n_const = n_const;
    prog.instr = instrs.data();
    prog.const_table = static_cast<u32*>(const_buf.ptr);
    prog.out_reg = static_cast<uint16_t*>(out_buf.ptr);
    
    // Create prime meta
    auto* p_ptr = static_cast<u32*>(p_buf.ptr);
    std::vector<PrimeMeta> pm = create_prime_meta(
        std::vector<u32>(p_ptr, p_ptr + K));
    
    // Config
    WalkConfig cfg;
    cfg.depth = depth;
    cfg.depth1 = depth1;
    cfg.depth2 = depth2;
    cfg.m = m;
    cfg.dim = dim;
    cfg.K = K;
    cfg.B = B;
    
    // Allocate outputs
    py::array_t<u32> P_final({B, E});
    py::array_t<uint8_t> alive(B);
    py::array_t<float> est1(B), est2(B), delta1(B), delta2(B);
    
    WalkOutputs out;
    out.P_final = static_cast<u32*>(P_final.request().ptr);
    out.alive = static_cast<uint8_t*>(alive.request().ptr);
    out.est1 = static_cast<float*>(est1.request().ptr);
    out.est2 = static_cast<float*>(est2.request().ptr);
    out.delta1 = static_cast<float*>(delta1.request().ptr);
    out.delta2 = static_cast<float*>(delta2.request().ptr);
    
    // Call CPU implementation
    walk_fused_cpu(
        cfg, prog,
        static_cast<i32*>(shift_buf.ptr),
        static_cast<i32*>(dir_buf.ptr),
        pm.data(),
        out);
    
    py::dict result;
    result["P_final"] = P_final;
    result["alive"] = alive;
    result["est1"] = est1;
    result["est2"] = est2;
    result["delta1"] = delta1;
    result["delta2"] = delta2;
    return result;
}

// Python wrapper for topk
py::array_t<py::object> py_topk(
    py::array_t<float> scores,
    py::array_t<float> est,
    int Kkeep,
    bool ascending = true)
{
    auto score_buf = scores.request();
    auto est_buf = est.request();
    int B = (int)score_buf.shape[0];
    
    TopKConfig cfg;
    cfg.B = B;
    cfg.Kkeep = Kkeep;
    cfg.ascending = ascending;
    
    std::vector<TopKItem> items(Kkeep);
    
    topk_reduce_cpu(
        static_cast<float*>(score_buf.ptr),
        static_cast<float*>(est_buf.ptr),
        items.data(),
        cfg);
    
    // Return as structured array
    py::array_t<float> out_scores(Kkeep);
    py::array_t<int> out_indices(Kkeep);
    py::array_t<float> out_est(Kkeep);
    
    auto* s_ptr = static_cast<float*>(out_scores.request().ptr);
    auto* i_ptr = static_cast<int*>(out_indices.request().ptr);
    auto* e_ptr = static_cast<float*>(out_est.request().ptr);
    
    for (int i = 0; i < Kkeep; ++i) {
        s_ptr[i] = items[i].score;
        i_ptr[i] = items[i].shift_idx;
        e_ptr[i] = items[i].est;
    }
    
    py::dict result;
    result["scores"] = out_scores;
    result["indices"] = out_indices;
    result["est"] = out_est;
    return result;
}

// Modular arithmetic helpers
u32 py_add_mod(u32 a, u32 b, u32 p) { return add_mod(a, b, p); }
u32 py_sub_mod(u32 a, u32 b, u32 p) { return sub_mod(a, b, p); }
u32 py_mul_mod(u32 a, u32 b, u32 p) {
    u64 mu = compute_barrett_mu(p);
    return mul_mod(a, b, p, mu);
}
u32 py_inv_mod(u32 a, u32 p) {
    u64 mu = compute_barrett_mu(p);
    return inv_mod(a, p, mu);
}
u32 py_pow_mod(u32 base, u32 exp, u32 p) {
    u64 mu = compute_barrett_mu(p);
    return pow_mod(base, exp, p, mu);
}

// Vectorized modular operations
py::array_t<u32> py_mul_mod_vec(
    py::array_t<u32> a, py::array_t<u32> b, u32 p)
{
    auto a_buf = a.request();
    auto b_buf = b.request();
    size_t n = a_buf.size;
    
    py::array_t<u32> result(n);
    auto* a_ptr = static_cast<u32*>(a_buf.ptr);
    auto* b_ptr = static_cast<u32*>(b_buf.ptr);
    auto* r_ptr = static_cast<u32*>(result.request().ptr);
    
    u64 mu = compute_barrett_mu(p);
    for (size_t i = 0; i < n; ++i) {
        r_ptr[i] = mul_mod(a_ptr[i], b_ptr[i], p, mu);
    }
    return result;
}

// Op enum for Python
enum PyOp {
    PY_OP_NOP = 0,
    PY_OP_LOAD_X = 1,
    PY_OP_LOAD_C = 2,
    PY_OP_ADD = 3,
    PY_OP_SUB = 4,
    PY_OP_MUL = 5,
    PY_OP_NEG = 6,
    PY_OP_POW2 = 7,
    PY_OP_POW3 = 8,
    PY_OP_INV = 9,
    PY_OP_MULINV = 10,
    PY_OP_COPY = 11
};

PYBIND11_MODULE(rns_rocm, m) {
    m.doc() = "RNS-ROCm: Residue Number System library for exact modular arithmetic";
    
    // Op enum
    py::enum_<PyOp>(m, "Op")
        .value("NOP", PY_OP_NOP)
        .value("LOAD_X", PY_OP_LOAD_X)
        .value("LOAD_C", PY_OP_LOAD_C)
        .value("ADD", PY_OP_ADD)
        .value("SUB", PY_OP_SUB)
        .value("MUL", PY_OP_MUL)
        .value("NEG", PY_OP_NEG)
        .value("POW2", PY_OP_POW2)
        .value("POW3", PY_OP_POW3)
        .value("INV", PY_OP_INV)
        .value("MULINV", PY_OP_MULINV)
        .value("COPY", PY_OP_COPY)
        .export_values();
    
    // Prime generation
    m.def("generate_primes", &py_generate_primes,
          py::arg("K"), py::arg("seed") = 12345,
          "Generate K coprime 31-bit primes");
    
    // Scalar modular ops
    m.def("add_mod", &py_add_mod, "Modular addition");
    m.def("sub_mod", &py_sub_mod, "Modular subtraction");
    m.def("mul_mod", &py_mul_mod, "Modular multiplication (Barrett)");
    m.def("inv_mod", &py_inv_mod, "Modular inverse");
    m.def("pow_mod", &py_pow_mod, "Modular exponentiation");
    
    // Vectorized ops
    m.def("mul_mod_vec", &py_mul_mod_vec,
          "Vectorized modular multiplication");
    
    // Bytecode evaluation
    m.def("eval_program", &py_eval_program,
          py::arg("m"), py::arg("dim"), py::arg("n_reg"),
          py::arg("opcodes"), py::arg("dsts"), py::arg("as"), py::arg("bs"),
          py::arg("out_reg"), py::arg("const_table"), py::arg("x_vals"),
          py::arg("primes"),
          "Evaluate bytecode program to generate matrices");
    
    // Walk kernel
    m.def("walk_fused", &py_walk_fused,
          py::arg("depth"), py::arg("depth1"), py::arg("depth2"),
          py::arg("m"), py::arg("dim"), py::arg("K"), py::arg("B"),
          py::arg("opcodes"), py::arg("dsts"), py::arg("as"), py::arg("bs"),
          py::arg("out_reg"), py::arg("const_table"), py::arg("n_reg"),
          py::arg("shifts"), py::arg("dirs"), py::arg("primes"),
          "Run fused walk kernel");
    
    // TopK
    m.def("topk", &py_topk,
          py::arg("scores"), py::arg("est"), py::arg("Kkeep"),
          py::arg("ascending") = true,
          "Select top-K candidates");
    
    // Version info
    m.attr("__version__") = "0.2.0";
}

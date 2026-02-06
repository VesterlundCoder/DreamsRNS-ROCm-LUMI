#ifndef RNS_EVAL_H
#define RNS_EVAL_H

#include "config.h"
#include "modops.h"

namespace rns {

enum Op : uint16_t {
  OP_NOP = 0,
  OP_LOAD_X,    // dst = x[a]
  OP_LOAD_C,    // dst = const_table[a]
  OP_ADD,       // dst = a + b
  OP_SUB,       // dst = a - b
  OP_MUL,       // dst = a * b
  OP_NEG,       // dst = -a
  OP_POW2,      // dst = a^2
  OP_POW3,      // dst = a^3
  OP_INV,       // dst = 1/a (sets alive=false if a==0)
  OP_MULINV,    // dst = a * (1/b)
  OP_COPY,      // dst = a
};

struct Instr {
  Op op;
  uint16_t dst;
  uint16_t a;
  uint16_t b;
};

struct Program {
  int m;              // matrix dimension
  int dim;            // number of x-variables
  int n_instr;        // number of instructions
  int n_reg;          // number of registers used
  int n_const;        // number of constants
  const Instr* instr;
  const u32* const_table;     // [n_const] per prime -> actually [K][n_const]
  const uint16_t* out_reg;    // [E] output register indices for matrix
};

struct EvalResult {
  u32* out_matrix;    // [K][B][E] output
  uint8_t* alive;     // [K][B] alive mask (1=alive, 0=dead from INV failure)
};

// Evaluate program for all batches, producing output matrices
// x_vals: [K][B][dim] input x-values
// out_matrix: [K][B][E] output matrix (E = m*m)
// alive: [K][B] alive mask
void eval_program_to_matrix(
    const Program& prog,
    const u32* x_vals,
    u32* out_matrix,
    uint8_t* alive,
    const PrimeMeta* pm,
    int K, int B);

// CPU reference
void eval_program_to_matrix_cpu(
    const Program& prog,
    const u32* x_vals,
    u32* out_matrix,
    uint8_t* alive,
    const PrimeMeta* pm,
    int K, int B);

} // namespace rns

#endif // RNS_EVAL_H

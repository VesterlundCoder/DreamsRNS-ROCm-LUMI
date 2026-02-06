#pragma once
#include <cstdint>
#include <vector>
#include "rns_types.h"
#include "crt_types.h"

namespace rns {

// ============================================================================
// CRT Reconstruction
// ============================================================================

/**
 * Reconstruct integer from residues using Garner's algorithm.
 * 
 * Given residues r[0..K-1] where r[i] = x mod p_i,
 * reconstructs x mod (p_0 * p_1 * ... * p_{K-1}).
 * 
 * @param residues Array of K residues
 * @param plan     Precomputed CRT plan
 * @return         Reconstructed integer as BigInt
 */
BigInt crt_reconstruct(const uint32_t* residues, const CrtPlan& plan);

/**
 * Reconstruct multiple integers in batch.
 * 
 * @param residues Array of shape [K][N], flattened as k*N + i
 * @param plan     Precomputed CRT plan
 * @param N        Number of integers to reconstruct
 * @return         Vector of N BigInts
 */
std::vector<BigInt> crt_reconstruct_batch(
    const uint32_t* residues, const CrtPlan& plan, int N);

/**
 * Reconstruct a matrix from RNS representation.
 * 
 * @param residues Array of shape [K][m*m]
 * @param plan     CRT plan
 * @param m        Matrix dimension
 * @return         Vector of m*m BigInts (row-major)
 */
std::vector<BigInt> crt_reconstruct_matrix(
    const uint32_t* residues, const CrtPlan& plan, int m);

// ============================================================================
// RNS Encoding
// ============================================================================

/**
 * Encode a BigInt into RNS residues.
 * 
 * @param x      Integer to encode
 * @param plan   CRT plan
 * @param out    Output array of K residues
 */
void rns_encode(const BigInt& x, const CrtPlan& plan, uint32_t* out);

/**
 * Encode a 64-bit integer into RNS residues.
 */
void rns_encode_u64(uint64_t x, const CrtPlan& plan, uint32_t* out);

/**
 * Encode a matrix of BigInts into RNS representation.
 * 
 * @param matrix  Vector of m*m BigInts
 * @param plan    CRT plan
 * @param m       Matrix dimension
 * @param out     Output array [K][m*m]
 */
void rns_encode_matrix(
    const std::vector<BigInt>& matrix, const CrtPlan& plan, int m, uint32_t* out);

// ============================================================================
// Signed Integer Support
// ============================================================================

/**
 * Reconstruct a signed integer from RNS.
 * Assumes the result is in range [-M/2, M/2) where M is the total modulus.
 */
BigInt crt_reconstruct_signed(const uint32_t* residues, const CrtPlan& plan);

/**
 * Encode a signed BigInt into RNS.
 * For negative x, encodes (M + x) where M is the total modulus.
 */
void rns_encode_signed(const BigInt& x, const CrtPlan& plan, uint32_t* out);

}  // namespace rns

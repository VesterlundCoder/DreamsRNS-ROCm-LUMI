#pragma once
#include <cstdint>
#include <vector>
#include "rns_types.h"

namespace rns {

/**
 * Generate K distinct 31-bit primes.
 * 
 * Uses deterministic generation from a seed for reproducibility.
 * All generated primes are verified to be coprime (trivially true for distinct primes).
 * 
 * @param K     Number of primes to generate
 * @param seed  Random seed for deterministic generation
 * @return      Vector of K primes
 */
std::vector<uint32_t> generate_primes(int K, uint32_t seed = 12345);

/**
 * Generate K primes near a target value.
 * Useful for controlling prime size.
 * 
 * @param K      Number of primes
 * @param target Target prime value (will find primes near this)
 * @param seed   Random seed
 */
std::vector<uint32_t> generate_primes_near(int K, uint32_t target, uint32_t seed = 12345);

/**
 * Get a fixed set of well-known 31-bit primes.
 * These are commonly used and verified.
 */
std::vector<uint32_t> get_standard_primes_31bit(int K);

/**
 * Create Modulus32 with precomputed Barrett constant.
 */
Modulus32 make_modulus(uint32_t p);

/**
 * Create PrimeSet from a list of primes.
 * Computes all Barrett constants.
 */
PrimeSet make_prime_set(const std::vector<uint32_t>& primes);

/**
 * Verify that all primes in a set are distinct and actually prime.
 */
bool verify_primes(const std::vector<uint32_t>& primes);

}  // namespace rns

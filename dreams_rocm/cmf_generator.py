"""
Generic pFq CMF spec generator for hypergeometric families.

Supported families:
  - 2F2: rank=3, 3×3, dim=4
  - 3F2: rank=4, 4×4, dim=5
  - 3F3: rank=4, 4×4, dim=6
  - 4F3: rank=5, 5×5, dim=7
  - 5F4: rank=6, 6×6, dim=9

Each spec is a JSON-serializable dict with:
  name, rank, p, q, dim, a_params, b_params, z_str,
  matrix (row,col -> SymPy expr string), axis_names, directions, spec_hash.

The companion matrix for pFq is the standard (max(p,q)+1)×(max(p,q)+1)
construction where Pochhammer parameters define the recurrence.

Usage:
    python -m dreams_rocm.cmf_generator --family 2F2 --count 10000 -o cmfs_2F2/
    python -m dreams_rocm.cmf_generator --all --count 10000 -o sweep_data/
"""

from __future__ import annotations

import json
import math
import os
import itertools
import hashlib
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import List, Tuple, Optional, Dict, Set

import numpy as np


# ── Parameter pool ───────────────────────────────────────────────────────

PARAM_POOL_INTEGERS = list(range(-5, 6))
PARAM_POOL_HALVES   = [Fraction(k, 2) for k in range(-11, 12)]
PARAM_POOL_THIRDS   = [Fraction(k, 3) for k in range(-8, 9)]
PARAM_POOL_QUARTERS = [Fraction(k, 4) for k in range(-10, 11)]
PARAM_POOL_SIXTHS   = [Fraction(k, 6) for k in range(-12, 13)]

def _build_param_pool() -> List[Fraction]:
    pool: Set[Fraction] = set()
    for x in PARAM_POOL_INTEGERS:
        pool.add(Fraction(x))
    for x in PARAM_POOL_HALVES:
        pool.add(x)
    for x in PARAM_POOL_THIRDS:
        pool.add(x)
    for x in PARAM_POOL_QUARTERS:
        pool.add(x)
    for x in PARAM_POOL_SIXTHS:
        pool.add(x)
    # Remove 0 and negative integers (Pochhammer poles)
    pool = {x for x in pool if x != 0 and not (x < 0 and x.denominator == 1)}
    return sorted(pool, key=lambda x: (abs(x), x))

PARAM_POOL = _build_param_pool()

# Tiered pools: prefer small parameters in systematic phase
POOL_TIER1 = [x for x in PARAM_POOL if abs(x) <= Fraction(2, 1)]   # ~35 params
POOL_TIER2 = [x for x in PARAM_POOL if abs(x) <= Fraction(3, 1)]   # ~55 params
POOL_TIER3 = PARAM_POOL                                              # ~90 params


@dataclass
class CMFSpec:
    """Specification for a single CMF family."""
    name: str
    rank: int
    p: int
    q: int
    dim: int
    a_params: List[str]
    b_params: List[str]
    z_str: str
    matrix: Dict[str, str]
    axis_names: List[str]
    directions: List[int]
    spec_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Validation & hashing ─────────────────────────────────────────────────

def _spec_hash(p: int, q: int, a_params: List[Fraction], b_params: List[Fraction], z: str) -> str:
    key = f"{p}F{q}|a={sorted(str(x) for x in a_params)}|b={sorted(str(x) for x in b_params)}|z={z}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _is_valid_pFq(a_params: List[Fraction], b_params: List[Fraction]) -> bool:
    """Check validity: no b=non-positive-int, no a_i==b_j, all distinct within group."""
    for b in b_params:
        if b.denominator == 1 and b <= 0:
            return False
    if set(a_params) & set(b_params):
        return False
    if len(set(a_params)) != len(a_params):
        return False
    if len(set(b_params)) != len(b_params):
        return False
    return True


# ── Generic pFq companion matrix builder ─────────────────────────────────

def _product_str(n: str, params: List[Fraction]) -> str:
    """Build string for product of (n + param_i) terms."""
    factors = [f"({n} + {p})" for p in params]
    return " * ".join(factors)


def _sum_of_k_products_str(n: str, params: List[Fraction], k: int) -> str:
    """Build string for elementary symmetric polynomial e_k of (n+param_i)."""
    terms = []
    for combo in itertools.combinations(params, k):
        factors = [f"({n} + {p})" for p in combo]
        terms.append(" * ".join(factors))
    if not terms:
        return "0"
    return " + ".join(terms)


def build_pFq_companion_matrix(
    p: int,
    q: int,
    a_params: List[Fraction],
    b_params: List[Fraction],
) -> Dict[str, str]:
    """Build the generic (r×r) companion matrix for pFq.

    The matrix size r = max(p, q) + 1.

    Structure (last-column form):
      - Sub-diagonal is 1 (shift register)
      - Last column contains the recurrence coefficients
      - All other entries are 0

    For the (r-1)-term recurrence of pFq:
      Row 0, col r-1:     (-1)^? * prod(n + a_i)
      Row k, col r-1:     intermediate coefficient (symmetric polynomials)
      Row r-1, col r-1:   prod(n + b_j)
    """
    r = max(p, q) + 1
    n = "n"
    matrix: Dict[str, str] = {}

    # Fill with zeros and sub-diagonal ones
    for i in range(r):
        for j in range(r):
            if i == j + 1:
                matrix[f"{i},{j}"] = "1"
            else:
                matrix[f"{i},{j}"] = "0"

    # Last column: recurrence coefficients
    # Row 0: numerator Pochhammer product (sign depends on convention)
    an_prod = _product_str(n, a_params) if a_params else "1"
    bn_prod = _product_str(n, b_params) if b_params else "1"

    # The standard companion has:
    # - Top-right: ±prod(n+a_i)    (numerator rising factorial)
    # - Bottom-right: prod(n+b_j)  (denominator rising factorial)
    # - Middle rows: differences of elementary symmetric polynomials

    if r == 2:
        # Simple 2×2 (1F1 or 2F1 reduced): [[0, a_prod], [1, b_prod]]
        matrix[f"0,{r-1}"] = an_prod
        matrix[f"1,{r-1}"] = bn_prod
    else:
        # General r×r companion
        # Row 0: numerator product
        matrix[f"0,{r-1}"] = an_prod

        # Middle rows: mixed coefficients from symmetric polynomial differences
        for row in range(1, r - 1):
            # Coefficient for row `row` in the last column
            # Uses elementary symmetric polynomials of degree (r-1-row)
            k = r - 1 - row
            bn_ek = _sum_of_k_products_str(n, b_params, k) if k <= len(b_params) else "0"
            an_ek = _sum_of_k_products_str(n, a_params, k) if k <= len(a_params) else "0"
            matrix[f"{row},{r-1}"] = f"({bn_ek}) - ({an_ek})"

        # Last row: denominator product
        matrix[f"{r-1},{r-1}"] = bn_prod

    return matrix


# ── Family metadata ──────────────────────────────────────────────────────

FAMILY_CONFIG = {
    "2F2": {"p": 2, "q": 2, "rank": 3, "dim": 4},
    "3F2": {"p": 3, "q": 2, "rank": 4, "dim": 5},
    "3F3": {"p": 3, "q": 3, "rank": 4, "dim": 6},
    "4F3": {"p": 4, "q": 3, "rank": 5, "dim": 7},
    "5F4": {"p": 5, "q": 4, "rank": 6, "dim": 9},
}


def _axis_names(p: int, q: int) -> List[str]:
    names = [f"x{i}" for i in range(p)] + [f"y{j}" for j in range(q)]
    return names


# ── Generic pFq spec generator ───────────────────────────────────────────

def generate_pFq_specs(
    p: int,
    q: int,
    n_specs: int = 10000,
    seed: int = 42,
) -> List[CMFSpec]:
    """Generate n_specs unique pFq CMF specifications.

    Two-phase strategy:
      Phase 1: Systematic enumeration of small-parameter combos (tier 1 → tier 2)
      Phase 2: Random sampling from full pool to fill remaining
    """
    rng = np.random.default_rng(seed)
    n_a = p
    n_b = q
    rank = max(p, q) + 1
    dim = p + q
    family_name = f"{p}F{q}"
    axes = _axis_names(p, q)
    directions = [1] * dim

    seen_hashes: Set[str] = set()
    specs: List[CMFSpec] = []

    def _try_add(a_list: List[Fraction], b_list: List[Fraction]) -> bool:
        a_sorted = sorted(a_list)
        b_sorted = sorted(b_list)
        if not _is_valid_pFq(a_sorted, b_sorted):
            return False
        h = _spec_hash(p, q, a_sorted, b_sorted, "1")
        if h in seen_hashes:
            return False
        seen_hashes.add(h)

        matrix = build_pFq_companion_matrix(p, q, a_sorted, b_sorted)
        a_strs = [str(x) for x in a_sorted]
        b_strs = [str(x) for x in b_sorted]

        spec = CMFSpec(
            name=f"{family_name}({','.join(a_strs)};{','.join(b_strs)})",
            rank=rank, p=p, q=q, dim=dim,
            a_params=a_strs, b_params=b_strs, z_str="1",
            matrix=matrix, axis_names=axes, directions=directions,
            spec_hash=h,
        )
        specs.append(spec)
        return True

    # Phase 1: Systematic enumeration through tiered pools
    for tier_pool in [POOL_TIER1, POOL_TIER2, POOL_TIER3]:
        if len(specs) >= n_specs:
            break
        for combo_a in itertools.combinations(tier_pool, n_a):
            if len(specs) >= n_specs:
                break
            a_list = list(combo_a)
            for combo_b in itertools.combinations(tier_pool, n_b):
                if len(specs) >= n_specs:
                    break
                _try_add(a_list, list(combo_b))

    # Phase 2: Random sampling from full pool
    attempts = 0
    max_attempts = n_specs * 50
    while len(specs) < n_specs and attempts < max_attempts:
        attempts += 1
        total_params = n_a + n_b
        idxs = rng.choice(len(PARAM_POOL), size=total_params, replace=False)
        a_list = [PARAM_POOL[i] for i in idxs[:n_a]]
        b_list = [PARAM_POOL[i] for i in idxs[n_a:]]
        _try_add(a_list, b_list)

    return specs[:n_specs]


# ── Convenience wrappers (backward compat) ───────────────────────────────

def generate_2f2_specs(n_specs: int = 10000, seed: int = 42) -> List[CMFSpec]:
    return generate_pFq_specs(2, 2, n_specs, seed)

def generate_3f2_specs(n_specs: int = 10000, seed: int = 100) -> List[CMFSpec]:
    return generate_pFq_specs(3, 2, n_specs, seed)

def generate_3f3_specs(n_specs: int = 10000, seed: int = 137) -> List[CMFSpec]:
    return generate_pFq_specs(3, 3, n_specs, seed)

def generate_4f3_specs(n_specs: int = 10000, seed: int = 200) -> List[CMFSpec]:
    return generate_pFq_specs(4, 3, n_specs, seed)

def generate_5f4_specs(n_specs: int = 10000, seed: int = 300) -> List[CMFSpec]:
    return generate_pFq_specs(5, 4, n_specs, seed)


# ── Batch file writer (split into chunks of chunk_size) ──────────────────

def write_specs_chunked(
    specs: List[CMFSpec],
    family_name: str,
    output_dir: str,
    chunk_size: int = 1000,
) -> List[str]:
    """Write specs to JSONL files, split into chunks.

    Returns list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    n_chunks = math.ceil(len(specs) / chunk_size)

    for i in range(n_chunks):
        chunk = specs[i * chunk_size : (i + 1) * chunk_size]
        fname = f"{family_name}_part{i:02d}.jsonl"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'w') as f:
            for spec in chunk:
                f.write(json.dumps(spec.to_dict(), default=str) + "\n")
        paths.append(fpath)

    return paths


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate pFq CMF specs for LUMI exhaust sweep")
    parser.add_argument("--family", type=str, default=None,
                        choices=list(FAMILY_CONFIG.keys()),
                        help="Single family to generate (or use --all)")
    parser.add_argument("--all", action="store_true",
                        help="Generate all 5 families")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of CMFs per family")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="CMFs per output file")
    parser.add_argument("-o", "--output-dir", type=str, default="sweep_data",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (offset per family)")
    args = parser.parse_args()

    if not args.all and args.family is None:
        parser.error("Specify --family or --all")

    families = list(FAMILY_CONFIG.keys()) if args.all else [args.family]
    seed_offsets = {"2F2": 0, "3F2": 100, "3F3": 200, "4F3": 300, "5F4": 400}

    grand_total = 0
    for fam in families:
        cfg = FAMILY_CONFIG[fam]
        seed = args.seed + seed_offsets.get(fam, 0)
        print(f"\n{'='*60}")
        print(f"Generating {args.count} {fam} specs (p={cfg['p']}, q={cfg['q']}, "
              f"rank={cfg['rank']}, dim={cfg['dim']}) ...")

        specs = generate_pFq_specs(cfg["p"], cfg["q"], args.count, seed)
        fam_dir = os.path.join(args.output_dir, fam)
        paths = write_specs_chunked(specs, fam, fam_dir, args.chunk_size)

        print(f"  Generated: {len(specs)} unique specs")
        print(f"  Files:     {len(paths)} × {args.chunk_size} → {fam_dir}/")
        for p in paths:
            print(f"    {os.path.basename(p)}")
        grand_total += len(specs)

    print(f"\n{'='*60}")
    print(f"TOTAL: {grand_total} CMFs across {len(families)} families → {args.output_dir}/")


if __name__ == "__main__":
    main()

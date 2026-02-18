"""
CMF spec generator for pFq hypergeometric families.

Generates parametric CMF families for:
  - 2F2 (rank 3, 3×3 matrices, dim=4 axes: x0,x1,y0,y1)
  - 3F3 (rank 4, 4×4 matrices, dim=6 axes: x0,x1,x2,y0,y1,y2)

Each spec is a JSON-serializable dict containing:
  - name, rank, p, q, dim
  - a_params, b_params (numerator/denominator Pochhammer parameters)
  - matrix: dict of (row,col) -> SymPy expression string
  - axis_names, directions

The companion matrix for pFq is the standard (p+1)×(p+1) construction
where the Pochhammer parameters define the recurrence.

Usage:
    python -m dreams_rocm.cmf_generator --n2f2 1000 --n3f3 1000 -o cmfs_exhaust.jsonl
"""

from __future__ import annotations

import json
import math
import itertools
import hashlib
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import List, Tuple, Optional, Dict

import numpy as np


# ── Parameter sampling ───────────────────────────────────────────────────

# Small rational parameter pool for Pochhammer (a_i), (b_j)
# These are typical in known PCF/CMF identities and cover good search space
PARAM_POOL_INTEGERS = list(range(-5, 6))  # -5..5
PARAM_POOL_HALVES = [Fraction(k, 2) for k in range(-9, 10)]  # -9/2..9/2
PARAM_POOL_THIRDS = [Fraction(k, 3) for k in range(-6, 7)]   # -2..2 by 1/3
PARAM_POOL_QUARTERS = [Fraction(k, 4) for k in range(-8, 9)]  # -2..2 by 1/4

# Merged pool sorted by absolute value (prefer small params)
def _build_param_pool() -> List[Fraction]:
    pool = set()
    for x in PARAM_POOL_INTEGERS:
        pool.add(Fraction(x))
    for x in PARAM_POOL_HALVES:
        pool.add(x)
    for x in PARAM_POOL_THIRDS:
        pool.add(x)
    for x in PARAM_POOL_QUARTERS:
        pool.add(x)
    # Remove 0 and negative integers (Pochhammer diverges)
    pool = {x for x in pool if x != 0 and not (x < 0 and x.denominator == 1)}
    return sorted(pool, key=lambda x: (abs(x), x))

PARAM_POOL = _build_param_pool()


@dataclass
class CMFSpec:
    """Specification for a single CMF family."""
    name: str
    rank: int        # matrix size = rank
    p: int           # numerator parameter count
    q: int           # denominator parameter count
    dim: int         # p + q (number of walk axes)
    a_params: List[str]  # numerator Pochhammer params as strings
    b_params: List[str]  # denominator Pochhammer params as strings
    z_str: str       # argument z (usually "1" or "-1" or polynomial)
    matrix: Dict[str, str]  # "(row,col)" -> sympy expr string
    axis_names: List[str]
    directions: List[int]  # default direction (all 1s)
    spec_hash: str   # deterministic hash for dedup

    def to_dict(self) -> dict:
        return asdict(self)


def _spec_hash(p: int, q: int, a_params: List[Fraction], b_params: List[Fraction], z: str) -> str:
    """Deterministic hash for dedup."""
    key = f"{p}F{q}|a={sorted(str(x) for x in a_params)}|b={sorted(str(x) for x in b_params)}|z={z}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _is_valid_pFq(a_params: List[Fraction], b_params: List[Fraction]) -> bool:
    """Check if parameters give a valid (non-degenerate) pFq."""
    # b_params must not be non-positive integers (poles of Gamma)
    for b in b_params:
        if b.denominator == 1 and b <= 0:
            return False
    # a_i must not equal any b_j (trivial cancellation)
    a_set = set(a_params)
    b_set = set(b_params)
    if a_set & b_set:
        return False
    # All params must be distinct within their group
    if len(set(a_params)) != len(a_params):
        return False
    if len(set(b_params)) != len(b_params):
        return False
    return True


# ── 2F2 companion matrix (3×3, rank=3, dim=4) ───────────────────────────

def _build_2f2_matrix(a1: Fraction, a2: Fraction, b1: Fraction, b2: Fraction) -> Dict[str, str]:
    """Build the 3×3 companion matrix for 2F2(a1,a2; b1,b2; z).

    The recurrence for 2F2 gives a 3-term relation which maps to a 3×3
    companion matrix acting on [T_{n+2}, T_{n+1}, T_n]^T.

    Axes: x0 = n (step), x1 = k (unused but available), y0, y1 (shift params).

    The matrix entries are polynomials in n (loaded via LOAD_X with axis=0).
    We use the standard contiguous relation for generalized hypergeometric:

      (n+b1)(n+b2) T_{n+1} = [(2n+1)(stuff) + ...] T_n - (n+a1)(n+a2) z T_{n-1}

    For the companion matrix M(n), we write the recurrence as:
      [T_{n+1}]     [0  (n+a1)(n+a2)*z / ((n+b1)(n+b2))   0 ] [T_n    ]
      [T_n    ]  =  [1  0                                   0 ] [T_{n-1}]
      [1      ]     [0  0                                   1 ] [1      ]

    But for CMF walks, we use the full polynomial form without division.
    """
    # For a generic pFq companion matrix, the standard approach uses
    # the Pochhammer-shifted form. We build explicit polynomial entries.
    n = "n"  # will be mapped to axis x0

    # Numerator rising factorials: product of (n + a_i)
    num_factors = [f"({n} + {a1})", f"({n} + {a2})"]
    # Denominator rising factorials: product of (n + b_j)
    den_factors = [f"({n} + {b1})", f"({n} + {b2})"]

    # 3×3 companion matrix for the 3-term recurrence
    # M(n) = [[0, b(n)], [1, a(n)]] generalized to rank 3
    #
    # Standard form:
    # Row 0: [0, 0, -(n+a1)*(n+a2)]
    # Row 1: [1, 0, (n+b1)*(n+b2) + (n+a1)*(n+a2)]
    # Row 2: [0, 1, ... middle coefficient ...]
    #
    # Actually for pFq the companion matrix is simpler:
    # Use the standard (p+1)×(p+1) form where p=q=2
    matrix = {}

    # The 3×3 companion for 2F2:
    # Column 0 (shift): identity-like
    # We use the Euler/Gauss-style continued fraction companion:
    #
    # M(n) acts on [p_{n}, p_{n-1}, p_{n-2}]
    # This is the generalized [[0, b(n)], [1, a(n)]] for rank 3

    # Numerator product
    an_prod = f"({n} + {a1}) * ({n} + {a2})"
    # Denominator product
    bn_prod = f"({n} + {b1}) * ({n} + {b2})"

    # Companion matrix (column-major convention matching our walk):
    # [[   0,         0,    -(n+a1)(n+a2)  ],
    #  [   1,         0,     c_middle       ],
    #  [   0,         1,     (n+b1)(n+b2)   ]]
    #
    # where c_middle encodes the 3-term recurrence coefficient

    # Middle coefficient for 2F2: involves sum of products
    # c = (2n+1)(a1+a2+b1+b2)/2 - ... (simplified for standard 2F2)
    # For generality, use: c_mid = (n+b1)(n+b2) + (n+a1)(n+a2) - n*(n+1)
    # This is a standard form that works for convergent extraction

    c_mid = f"({bn_prod}) + ({an_prod}) - {n}*({n} + 1)"

    matrix["0,0"] = "0"
    matrix["0,1"] = "0"
    matrix["0,2"] = f"-({an_prod})"
    matrix["1,0"] = "1"
    matrix["1,1"] = "0"
    matrix["1,2"] = c_mid
    matrix["2,0"] = "0"
    matrix["2,1"] = "1"
    matrix["2,2"] = bn_prod

    return matrix


def generate_2f2_specs(
    n_specs: int = 1000,
    seed: int = 42,
) -> List[CMFSpec]:
    """Generate n_specs unique 2F2 CMF specifications.

    Systematically enumerates parameter combinations from PARAM_POOL,
    filters for validity, deduplicates, and returns up to n_specs.
    """
    rng = np.random.default_rng(seed)
    pool = PARAM_POOL
    seen_hashes: set = set()
    specs: List[CMFSpec] = []

    # Strategy: enumerate small params first, then sample from larger pool
    # Phase 1: systematic enumeration of "nice" params
    nice_params = [x for x in pool if abs(x) <= Fraction(3, 1)]

    for a1, a2 in itertools.combinations(nice_params, 2):
        if len(specs) >= n_specs:
            break
        for b1, b2 in itertools.combinations(nice_params, 2):
            if len(specs) >= n_specs:
                break

            a_list = sorted([a1, a2])
            b_list = sorted([b1, b2])

            if not _is_valid_pFq(a_list, b_list):
                continue

            h = _spec_hash(2, 2, a_list, b_list, "1")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            a_strs = [str(x) for x in a_list]
            b_strs = [str(x) for x in b_list]

            matrix = _build_2f2_matrix(a1, a2, b1, b2)

            spec = CMFSpec(
                name=f"2F2({','.join(a_strs)};{','.join(b_strs)})",
                rank=3,
                p=2, q=2,
                dim=4,
                a_params=a_strs,
                b_params=b_strs,
                z_str="1",
                matrix=matrix,
                axis_names=["x0", "x1", "y0", "y1"],
                directions=[1, 1, 1, 1],
                spec_hash=h,
            )
            specs.append(spec)

    # Phase 2: random sampling if still short
    if len(specs) < n_specs:
        for _ in range(n_specs * 10):
            if len(specs) >= n_specs:
                break
            idxs = rng.choice(len(pool), size=4, replace=False)
            a1, a2 = sorted([pool[idxs[0]], pool[idxs[1]]])
            b1, b2 = sorted([pool[idxs[2]], pool[idxs[3]]])

            if not _is_valid_pFq([a1, a2], [b1, b2]):
                continue
            h = _spec_hash(2, 2, [a1, a2], [b1, b2], "1")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            a_strs = [str(x) for x in [a1, a2]]
            b_strs = [str(x) for x in [b1, b2]]
            matrix = _build_2f2_matrix(a1, a2, b1, b2)

            spec = CMFSpec(
                name=f"2F2({','.join(a_strs)};{','.join(b_strs)})",
                rank=3, p=2, q=2, dim=4,
                a_params=a_strs, b_params=b_strs, z_str="1",
                matrix=matrix,
                axis_names=["x0", "x1", "y0", "y1"],
                directions=[1, 1, 1, 1],
                spec_hash=h,
            )
            specs.append(spec)

    return specs[:n_specs]


# ── 3F3 companion matrix (4×4, rank=4, dim=6) ───────────────────────────

def _build_3f3_matrix(
    a1: Fraction, a2: Fraction, a3: Fraction,
    b1: Fraction, b2: Fraction, b3: Fraction,
) -> Dict[str, str]:
    """Build the 4×4 companion matrix for 3F3(a1,a2,a3; b1,b2,b3; z).

    4×4 companion for the 4-term recurrence.
    """
    n = "n"

    an_prod = f"({n} + {a1}) * ({n} + {a2}) * ({n} + {a3})"
    bn_prod = f"({n} + {b1}) * ({n} + {b2}) * ({n} + {b3})"

    # Middle coefficients for 4-term recurrence
    # c2 involves cross terms
    an_sum2 = f"({n} + {a1})*({n} + {a2}) + ({n} + {a1})*({n} + {a3}) + ({n} + {a2})*({n} + {a3})"
    bn_sum2 = f"({n} + {b1})*({n} + {b2}) + ({n} + {b1})*({n} + {b3}) + ({n} + {b2})*({n} + {b3})"

    c2 = f"({bn_sum2}) - ({an_sum2})"
    c3 = f"({bn_prod}) + ({an_prod}) - {n}*({n}+1)*({n}+2)"

    matrix = {}
    # 4×4 companion:
    # [[0, 0, 0, (n+a1)(n+a2)(n+a3)   ],
    #  [1, 0, 0, c2                     ],
    #  [0, 1, 0, c3                     ],
    #  [0, 0, 1, (n+b1)(n+b2)(n+b3)    ]]
    matrix["0,0"] = "0"
    matrix["0,1"] = "0"
    matrix["0,2"] = "0"
    matrix["0,3"] = an_prod
    matrix["1,0"] = "1"
    matrix["1,1"] = "0"
    matrix["1,2"] = "0"
    matrix["1,3"] = c2
    matrix["2,0"] = "0"
    matrix["2,1"] = "1"
    matrix["2,2"] = "0"
    matrix["2,3"] = c3
    matrix["3,0"] = "0"
    matrix["3,1"] = "0"
    matrix["3,2"] = "1"
    matrix["3,3"] = bn_prod

    return matrix


def generate_3f3_specs(
    n_specs: int = 1000,
    seed: int = 137,
) -> List[CMFSpec]:
    """Generate n_specs unique 3F3 CMF specifications."""
    rng = np.random.default_rng(seed)
    pool = PARAM_POOL
    seen_hashes: set = set()
    specs: List[CMFSpec] = []

    # Smaller nice pool for 3F3 (6 params means combinatorial explosion)
    nice_params = [x for x in pool if abs(x) <= Fraction(2, 1)]

    # Phase 1: systematic
    for combo_a in itertools.combinations(nice_params, 3):
        if len(specs) >= n_specs:
            break
        a_list = sorted(combo_a)
        for combo_b in itertools.combinations(nice_params, 3):
            if len(specs) >= n_specs:
                break
            b_list = sorted(combo_b)

            if not _is_valid_pFq(list(a_list), list(b_list)):
                continue
            h = _spec_hash(3, 3, list(a_list), list(b_list), "1")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            a_strs = [str(x) for x in a_list]
            b_strs = [str(x) for x in b_list]
            matrix = _build_3f3_matrix(*a_list, *b_list)

            spec = CMFSpec(
                name=f"3F3({','.join(a_strs)};{','.join(b_strs)})",
                rank=4, p=3, q=3, dim=6,
                a_params=a_strs, b_params=b_strs, z_str="1",
                matrix=matrix,
                axis_names=["x0", "x1", "x2", "y0", "y1", "y2"],
                directions=[1, 1, 1, 1, 1, 1],
                spec_hash=h,
            )
            specs.append(spec)

    # Phase 2: random sampling
    if len(specs) < n_specs:
        for _ in range(n_specs * 20):
            if len(specs) >= n_specs:
                break
            idxs = rng.choice(len(pool), size=6, replace=False)
            a_list = sorted([pool[idxs[0]], pool[idxs[1]], pool[idxs[2]]])
            b_list = sorted([pool[idxs[3]], pool[idxs[4]], pool[idxs[5]]])

            if not _is_valid_pFq(list(a_list), list(b_list)):
                continue
            h = _spec_hash(3, 3, list(a_list), list(b_list), "1")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            a_strs = [str(x) for x in a_list]
            b_strs = [str(x) for x in b_list]
            matrix = _build_3f3_matrix(*a_list, *b_list)

            spec = CMFSpec(
                name=f"3F3({','.join(a_strs)};{','.join(b_strs)})",
                rank=4, p=3, q=3, dim=6,
                a_params=a_strs, b_params=b_strs, z_str="1",
                matrix=matrix,
                axis_names=["x0", "x1", "x2", "y0", "y1", "y2"],
                directions=[1, 1, 1, 1, 1, 1],
                spec_hash=h,
            )
            specs.append(spec)

    return specs[:n_specs]


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate CMF specs for exhaust sweep")
    parser.add_argument("--n2f2", type=int, default=1000, help="Number of 2F2 specs")
    parser.add_argument("--n3f3", type=int, default=1000, help="Number of 3F3 specs")
    parser.add_argument("-o", "--output", type=str, default="cmfs_exhaust.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--seed2", type=int, default=42, help="Seed for 2F2")
    parser.add_argument("--seed3", type=int, default=137, help="Seed for 3F3")
    args = parser.parse_args()

    specs_2f2 = generate_2f2_specs(args.n2f2, seed=args.seed2)
    specs_3f3 = generate_3f3_specs(args.n3f3, seed=args.seed3)

    all_specs = specs_2f2 + specs_3f3

    with open(args.output, 'w') as f:
        for spec in all_specs:
            f.write(json.dumps(spec.to_dict(), default=str) + "\n")

    print(f"Generated {len(specs_2f2)} 2F2 specs + {len(specs_3f3)} 3F3 specs")
    print(f"  Total: {len(all_specs)} CMFs → {args.output}")

    # Summary
    from .exhaust import exhaust_summary
    for d in [4, 6]:
        s = exhaust_summary(d)
        n_cmfs = len(specs_2f2) if d == 4 else len(specs_3f3)
        print(f"\n  dim={d}: {n_cmfs} CMFs × {s['n_trajectories']} traj × stages:")
        print(f"    Stage 1: {s['n_shifts_stage1']:>5} shifts → {n_cmfs * s['runs_stage1']:>12,} total runs")
        print(f"    Stage 2: {s['n_shifts_stage2']:>5} shifts → {n_cmfs * s['runs_stage2']:>12,} total runs")
        print(f"    Stage 3: {s['n_shifts_full']:>5} shifts → {n_cmfs * s['runs_full']:>12,} total runs")


if __name__ == "__main__":
    main()

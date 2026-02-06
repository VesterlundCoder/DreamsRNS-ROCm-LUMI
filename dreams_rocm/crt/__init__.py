"""
CRT (Chinese Remainder Theorem) module for Dreams pipeline.

Two-stage approach:
1. partial_crt: Fast GPU-assisted partial reconstruction for delta proxy
2. full_crt_cpu: High-precision CPU reconstruction for positive verification
"""

from .partial_crt import partial_crt_delta_proxy, partial_crt_reconstruct
from .full_crt_cpu import full_crt_verify, full_crt_reconstruct
from .delta_targets import ZETA_TARGETS, get_target_value, get_target_value_hp

__all__ = [
    "partial_crt_delta_proxy", "partial_crt_reconstruct",
    "full_crt_verify", "full_crt_reconstruct",
    "ZETA_TARGETS", "get_target_value", "get_target_value_hp",
]

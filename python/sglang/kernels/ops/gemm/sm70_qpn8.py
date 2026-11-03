"""QPN8: SM70 (V100) decode GEMM for block-FP8 (E4M3) weights, M <= 8.

Streams packed FP8 codes from HBM and decodes them into the FP16 operands of
``mma.sync.m8n8k4`` (adapted from https://github.com/dnv2003/v100-skinny, MIT).
Weights use block-wise ``[N/128][K/128]`` FP32 scales. The extension is
JIT-built from ``kernels/jit/csrc/gemm/sm70_fp8_qpn8_decode.cu`` on first use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "jit"
    / "csrc"
    / "gemm"
    / "sm70_fp8_qpn8_decode.cu"
)
# Fragment order of the 16 K values one lane decodes (matches the kernel).
_KORDER8 = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
MAX_M = 8


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_sm70_qpn8",
        sources=[str(_SOURCE)],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def prepack_codes(w8: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 E4M3 codes -> [N/32][K/16][lane 32][16 B] kernel order."""
    n, k = w8.shape
    device = w8.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=device)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(_KORDER8, device=device)
    kidx = torch.arange(groups, device=device).view(groups, 1) * 16 + korder.view(1, 16)
    ncol = torch.arange(tiles, device=device).view(tiles, 1) * 32 + col.view(1, 32)
    out = w8[
        ncol.view(tiles, 1, 32, 1).expand(tiles, groups, 32, 16),
        kidx.view(1, groups, 1, 16).expand(tiles, groups, 32, 16),
    ]
    return out.contiguous().view(-1)


def group_scales(scale_inv: torch.Tensor, k: int) -> torch.Tensor:
    """[N/128][K/128] FP32 -> FP16 [K/128][N/32], with the decoder's x256 folded in."""
    n_blocks = scale_inv.shape[0]
    scales = scale_inv.to(torch.float32).t().reshape(k // 128, n_blocks) * 256.0
    return scales.repeat_interleave(4, dim=1).to(torch.float16).contiguous()


def qpn8_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    k: int,
    split_k: int,
    nacc_arg: int,
) -> torch.Tensor:
    """``x`` [M <= 8, K] FP16 -> [M, N] FP16."""
    return _extension().qpn8_linear(x, codes, scales, n, k, split_k, nacc_arg)

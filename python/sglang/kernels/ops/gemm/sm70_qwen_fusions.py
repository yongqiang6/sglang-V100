"""Qwen3.8 Flash Next projection fusions for V100 (SM70) decode and MTP verify.

From the sglang-V100 fork, for the FP16 TP4 shapes (hidden 2560):

- ``gate_up``: the shared expert's [320, 2560] gate_up projection with
  SiLU-and-mul, one row.
- ``qkv_ba``: the GDN layer's [4096, 2560] qkvz and [24, 2560] ba projections in
  one launch, one row.
- ``qkvzba``: the same projections at two to seven rows, written straight into
  the split qkv / z / b / a layout of a ratio-three GDN layer (12 value heads
  of 128 over 4 key heads per rank).
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.gemm.sm70_dense_gemv import (
    MAX_VERIFY_ROWS,
    is_aligned_fp16,
    is_sm70,
)
from sglang.srt.environ import envs

HIDDEN_SIZE = 2560
SHARED_EXPERT_GATE_UP_SHAPE = (320, HIDDEN_SIZE)
QKVZ_SHAPE = (4096, HIDDEN_SIZE)
BA_SHAPE = (24, HIDDEN_SIZE)
# [Q 512, K 512, V 1536] of a qkvz row, then Z 1536; B and A are 12 each.
_QKV_WIDTH = 2560
_V_HEADS = 12
_HEAD_V_DIM = 128


def _enabled_on(x: torch.Tensor) -> bool:
    return (
        envs.SGLANG_SM70_QWEN_FUSIONS.get()
        and x.is_cuda
        and x.ndim == 2
        and x.shape[1] == HIDDEN_SIZE
    )


def gate_up_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _enabled_on(x)
        and x.shape[0] == 1
        and tuple(weight.shape) == SHARED_EXPERT_GATE_UP_SHAPE
        and weight.device == x.device
        and is_aligned_fp16(x, weight)
        and is_sm70(x.device)
    )


def _projections_supported(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> bool:
    return (
        _enabled_on(x)
        and tuple(qkvz_weight.shape) == QKVZ_SHAPE
        and tuple(ba_weight.shape) == BA_SHAPE
        and qkvz_weight.device == x.device
        and ba_weight.device == x.device
        and is_aligned_fp16(x, qkvz_weight, ba_weight)
        and is_sm70(x.device)
    )


def qkv_ba_supported(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> bool:
    return x.shape[0] == 1 and _projections_supported(x, qkvz_weight, ba_weight)


def qkvzba_supported(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> bool:
    return 2 <= x.shape[0] <= MAX_VERIFY_ROWS and _projections_supported(
        x, qkvz_weight, ba_weight
    )


@cache_once
def _gate_up_module():
    return load_jit(
        "sm70_qwen_gate_up",
        cuda_files=["gemm/sm70_qwen_fusions.cuh"],
        cuda_wrappers=[("gate_up", "sglang::sm70_qwen_fusions::gate_up")],
        # Keep the eager path's separate multiply and add roundings.
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _qkv_ba_module():
    return load_jit(
        "sm70_qwen_qkv_ba",
        cuda_files=["gemm/sm70_dense_gemv.cuh"],
        cuda_wrappers=[("qkv_ba", "sglang::sm70_dense_gemv::qkv_ba")],
    )


@cache_once
def _qkvzba_module(rows: int):
    return load_jit(
        "sm70_qwen_gdn",
        str(rows),
        cuda_files=["gemm/sm70_qwen_gdn.cuh"],
        cuda_wrappers=[("run", f"sglang::sm70_qwen_gdn::run<{rows},64>")],
    )


def gate_up(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up of ``x @ weight.T``, [1, 160]."""
    output = torch.empty(
        (1, SHARED_EXPERT_GATE_UP_SHAPE[0] // 2), dtype=x.dtype, device=x.device
    )
    _gate_up_module().gate_up(x, weight, output)
    return output


def qkv_ba(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(x @ qkvz_weight.T, x @ ba_weight.T) for one row, from one buffer."""
    output = torch.empty(
        (1, QKVZ_SHAPE[0] + BA_SHAPE[0]), dtype=x.dtype, device=x.device
    )
    _qkv_ba_module().qkv_ba(x, qkvz_weight, ba_weight, output)
    return output[:, : QKVZ_SHAPE[0]], output[:, QKVZ_SHAPE[0] :]


def qkvzba(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The split (mixed_qkv, z, b, a) GDN inputs for two or four rows."""
    rows = x.shape[0]
    qkv = torch.empty((rows, _QKV_WIDTH), dtype=x.dtype, device=x.device)
    z = torch.empty((rows, _V_HEADS * _HEAD_V_DIM), dtype=x.dtype, device=x.device)
    b = torch.empty((rows, _V_HEADS), dtype=x.dtype, device=x.device)
    a = torch.empty_like(b)
    _qkvzba_module(rows).run(x, qkvz_weight, ba_weight, qkv, z, b, a)
    return qkv, z.view(rows, _V_HEADS, _HEAD_V_DIM), b, a

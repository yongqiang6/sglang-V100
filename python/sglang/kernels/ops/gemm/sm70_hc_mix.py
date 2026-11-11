"""Hyper-connection mix and inject gates for V100 (SM70) decode and MTP verify.

From the sglang-V100 fork, for Qwen3.8 Flash Next's FP16 shape (4 streams of
2560, low rank 320) at one to seven rows. The persistent Triton mix is
pathologically slow on SM70; these kernels split it into a down projection with
SiLU and an up projection that fuses sigmoid, multiply and the stream mean,
keeping the eager FP16 rounding points. From two rows up the down projection
also computes the combine's inject-gate partial dots.
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

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
_ROW_SIZE = HC_COUNT * HIDDEN_SIZE
# The inject-gate partials: [rows, split, HC_COUNT] FP32, hc_combine's layout.
_GATE_SPLITS = 8


def mix_supported(
    x: torch.Tensor, weight_down: torch.Tensor, weight_up: torch.Tensor
) -> bool:
    return (
        envs.SGLANG_SM70_QWEN_FUSIONS.get()
        and x.is_cuda
        and x.ndim == 2
        and 1 <= x.shape[0] <= MAX_VERIFY_ROWS
        and x.shape[1] == _ROW_SIZE
        and tuple(weight_down.shape) == (LOWRANK, _ROW_SIZE)
        and tuple(weight_up.shape) == (_ROW_SIZE, LOWRANK)
        and weight_down.device == x.device
        and weight_up.device == x.device
        and is_aligned_fp16(x, weight_down, weight_up)
        and is_sm70(x.device)
    )


def gate_supported(x: torch.Tensor, inject_weight: torch.Tensor) -> bool:
    """Whether the down projection can also produce the inject-gate partials;
    the caller has already checked ``mix_supported``."""
    return (
        2 <= x.shape[0] <= MAX_VERIFY_ROWS
        and tuple(inject_weight.shape) == (HC_COUNT, _ROW_SIZE)
        and inject_weight.device == x.device
        and is_aligned_fp16(inject_weight)
    )


@cache_once
def _single_row_module():
    return load_jit(
        "sm70_hc_mix",
        cuda_files=["elementwise/sm70_hc_mix.cuh"],
        cuda_wrappers=[
            ("down", "sglang::sm70_hc::down"),
            ("up", "sglang::sm70_hc::up"),
        ],
        # Keep the eager path's separate multiply and add roundings.
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _batch_module(rows: int):
    return load_jit(
        "sm70_hc_batch",
        str(rows),
        cuda_files=["elementwise/sm70_hc_batch.cuh"],
        cuda_wrappers=[
            ("down", f"sglang::sm70_hc_batch::down_run<{rows}>"),
            ("up", f"sglang::sm70_hc_batch::up_run<{rows}>"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _gate_module(rows: int):
    # FMA contraction stays on to match hc_combine's own gate reduction.
    return load_jit(
        "sm70_hc_gate",
        str(rows),
        cuda_files=["elementwise/sm70_hc_gate.cuh"],
        cuda_wrappers=[
            ("down", f"sglang::sm70_hc_gate::down<{rows}>"),
            ("apply", "sglang::sm70_hc_gate::apply"),
        ],
    )


def _module(rows: int):
    return _single_row_module() if rows == 1 else _batch_module(rows)


def hc_down(x: torch.Tensor, weight_down: torch.Tensor) -> torch.Tensor:
    """silu(fp16(x @ weight_down.T) / HC_COUNT), [rows, LOWRANK]."""
    output = torch.empty((x.shape[0], LOWRANK), device=x.device, dtype=x.dtype)
    _module(x.shape[0]).down(x, weight_down, output)
    return output


def hc_up(
    activated: torch.Tensor, x: torch.Tensor, weight_up: torch.Tensor
) -> torch.Tensor:
    """Mean over streams of sigmoid(fp16(activated @ weight_up.T)) * x."""
    output = torch.empty((x.shape[0], HIDDEN_SIZE), device=x.device, dtype=x.dtype)
    _module(x.shape[0]).up(activated, x, weight_up, output)
    return output


def hc_down_with_gate(
    x: torch.Tensor, weight_down: torch.Tensor, inject_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_down`` plus the combine's inject-gate partials over ``x``."""
    rows = x.shape[0]
    activated = torch.empty((rows, LOWRANK), device=x.device, dtype=x.dtype)
    partials = torch.empty(
        (rows, _GATE_SPLITS, HC_COUNT), device=x.device, dtype=torch.float32
    )
    _gate_module(rows).down(x, weight_down, inject_weight, activated, partials)
    return activated, partials


def hc_apply_gate(
    block_output: torch.Tensor, residual: torch.Tensor, partials: torch.Tensor
) -> torch.Tensor:
    """residual + 2 * sigmoid(gate / HC_COUNT) * block_output, per stream."""
    output = torch.empty_like(residual)
    _gate_module(block_output.shape[0]).apply(block_output, residual, partials, output)
    return output

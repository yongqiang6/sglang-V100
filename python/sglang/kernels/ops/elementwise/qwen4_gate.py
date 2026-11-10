from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.numerics import round_bf16_to_fp32

_QWEN4_HC_COUNT = 4
_QWEN4_HIDDEN_SIZE = 2560


@triton.jit
def _qwen4_gate_value_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    num_tokens,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_group = tl.program_id(0)
    token = token_group // HC_COUNT
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = (token < num_tokens) & (hidden < HIDDEN_SIZE)

    # `gate` arrives already rounded by the eager multiply/reduce/divide; every
    # remaining eager rounding boundary of its dtype is reproduced below.
    gate = tl.load(gate_ptr + token_group).to(tl.float32)
    magnitude = tl.maximum(tl.abs(gate), 1.0e-6)
    if gate_ptr.dtype.element_ty == tl.float16:
        # FP16 (V100): eager clamp_min also rounds, and an approximate
        # reciprocal or exp can cross an FP16 midpoint for small gates.
        magnitude = magnitude.to(tl.float16).to(tl.float32)
        root = tl.sqrt(magnitude).to(tl.float16).to(tl.float32)
        sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
        transformed = (root * sign).to(tl.float16).to(tl.float32)
        activated = (
            tl.div_rn(1.0, 1.0 + libdevice.exp(-transformed))
            .to(tl.float16)
            .to(tl.float32)
        )
    else:
        root = round_bf16_to_fp32(tl.sqrt(magnitude))
        sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
        transformed = round_bf16_to_fp32(root * sign)
        activated = round_bf16_to_fp32(tl.sigmoid(transformed))

    value = tl.load(value_ptr + token * HIDDEN_SIZE + hidden, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offset = token_group * HIDDEN_SIZE + hidden
    tl.store(output_ptr + output_offset, activated * value, mask=mask)


def _sm70_fp16_gate_enabled(gate: torch.Tensor) -> bool:
    from sglang.kernels.ops.gemm.sm70_dense_gemv import is_sm70
    from sglang.srt.environ import envs

    return envs.SGLANG_SM70_QWEN_FUSIONS.get() and is_sm70(gate.device)


def can_fuse_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether inputs match Qwen4's fixed gate/value contract: BF16, or
    FP16 on V100."""

    return (
        gate.is_cuda
        and (
            gate.dtype == torch.bfloat16
            or (gate.dtype == torch.float16 and _sm70_fp16_gate_enabled(gate))
        )
        and gate.dim() == 3
        and gate.shape[1:] == (_QWEN4_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.is_cuda
        and value.dtype == gate.dtype
        and value.shape == (gate.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.is_contiguous()
    )


def fused_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply Qwen4's post-reduction gate and value broadcast in one kernel."""

    if not can_fuse_qwen4_gate_value(gate, value):
        raise ValueError("unsupported input for fused Qwen4 PLE gate/value")
    output = torch.empty(
        (gate.shape[0], _QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        _qwen4_gate_value_kernel[(gate.shape[0] * _QWEN4_HC_COUNT,)](
            gate,
            value,
            output,
            gate.shape[0],
            HC_COUNT=_QWEN4_HC_COUNT,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
    return output


@triton.jit
def _qwen4_gate_reduce_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = hidden < HIDDEN_SIZE
    key = tl.load(key_ptr + group * HIDDEN_SIZE + hidden, mask, other=0).to(tl.float32)
    query = tl.load(query_ptr + group * HIDDEN_SIZE + hidden, mask, other=0).to(
        tl.float32
    )
    product = round_bf16_to_fp32(key * query)
    gate = round_bf16_to_fp32(tl.sum(product, 0))
    gate = round_bf16_to_fp32(gate * (HIDDEN_SIZE**-0.5))
    magnitude = round_bf16_to_fp32(tl.maximum(tl.abs(gate), 1.0e-6))
    root = round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0, 1.0, tl.where(gate < 0, -1.0, 0.0))
    activated = round_bf16_to_fp32(tl.sigmoid(round_bf16_to_fp32(root * sign)))
    value = tl.load(
        value_ptr + (group // HC_COUNT) * HIDDEN_SIZE + hidden, mask, other=0
    ).to(tl.float32)
    tl.store(output_ptr + group * HIDDEN_SIZE + hidden, activated * value, mask)


def can_fuse_qwen4_gate_reduce(key, query, value):
    return (
        key.is_cuda
        and key.dtype == torch.bfloat16
        and key.ndim == 3
        and key.shape[1:] == (_QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE)
        and key.is_contiguous()
        and query.shape == key.shape
        and query.dtype == key.dtype
        and query.device == key.device
        and query.is_contiguous()
        and value.shape == (key.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.dtype == key.dtype
        and value.device == key.device
        and value.is_contiguous()
    )


def fused_qwen4_gate_reduce(key, query, value):
    if not can_fuse_qwen4_gate_reduce(key, query, value):
        raise ValueError("unsupported input for Qwen4 PLE gate reduction")
    output = torch.empty_like(key)
    if key.shape[0]:
        _qwen4_gate_reduce_kernel[(key.shape[0] * _QWEN4_HC_COUNT,)](
            key,
            query,
            value,
            output,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            HC_COUNT=_QWEN4_HC_COUNT,
            BLOCK_SIZE=triton.next_power_of_2(_QWEN4_HIDDEN_SIZE),
            enable_fp_fusion=False,
        )
    return output

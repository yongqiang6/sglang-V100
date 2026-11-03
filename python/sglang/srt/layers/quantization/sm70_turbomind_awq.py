"""V100 (SM70) AWQ dense linear layers on TurboMind's W4A16 GEMM.

The packed TurboMind copy replaces the AWQ tensors. Selected Qwen3.6-27B-AWQ TP4
shapes dequantize into a shared workspace and run a dense FP16 GEMM for
4096-token prefill chunks, the measured faster path there.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.ops.gemm import sm70_turbomind
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GROUP_SIZES = (32, 64, 128)
# [K, N] shapes (gate_up, down, in_proj_qkvz, out/o_proj at TP4) and the exact
# chunk size where dequantize + dense GEMM beats TurboMind W4A16 (group 128).
_PREFILL_DENSE_SHAPES = {(5120, 8704), (4352, 5120), (5120, 4096), (1536, 5120)}
_PREFILL_DENSE_M = 4096
_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(k * n for k, n in _PREFILL_DENSE_SHAPES)
# Owns one 85 MiB workspace per device; layers keep a reference to it.
_PREFILL_DENSE_WORKSPACES: dict[int, torch.Tensor] = {}


def sm70_turbomind_awq_supported(group_size: int) -> bool:
    return (
        envs.SGLANG_SM70_AWQ_TURBOMIND.get()
        and group_size in _GROUP_SIZES
        and sm70_turbomind.ops() is not None
    )


def _prefill_dense_workspace(weight: torch.Tensor) -> Optional[torch.Tensor]:
    device = weight.device.index
    workspace = _PREFILL_DENSE_WORKSPACES.get(device)
    if workspace is None:
        try:
            workspace = torch.empty(
                _PREFILL_DENSE_WORKSPACE_ELEMENTS,
                dtype=torch.float16,
                device=weight.device,
            )
        except torch.OutOfMemoryError:
            logger.warning(
                "SM70 AWQ: no memory for the exact dense prefill workspace; "
                "using TurboMind W4A16 for all prefills."
            )
            return None
        _PREFILL_DENSE_WORKSPACES[device] = workspace
    return workspace


def prepare_sm70_turbomind_awq_linear(
    layer: torch.nn.Module, group_size: int, pack_factor: int
) -> None:
    input_size = layer.qweight.shape[0]
    output_size = layer.qweight.shape[1] * pack_factor
    tm_weight, tm_scales, meta = sm70_turbomind.ops().awq_prepare(
        layer.qweight, layer.scales, layer.qzeros, group_size, False
    )
    layer.sm70_awq_weight = tm_weight
    layer.sm70_awq_scales = tm_scales
    layer.sm70_awq_k_ld = int(meta[0].item())
    layer.sm70_awq_q_ld = int(meta[1].item())
    layer.sm70_awq_group_size = group_size
    layer.sm70_awq_shape = (input_size, output_size)
    layer.sm70_awq_prefill_workspace = None
    if (
        envs.SGLANG_SM70_AWQ_PREFILL_EXACT_DENSE.get()
        and group_size == 128
        and (input_size, output_size) in _PREFILL_DENSE_SHAPES
    ):
        layer.sm70_awq_prefill_workspace = _prefill_dense_workspace(tm_weight)
    # The TurboMind copy serves every batch size, so drop the AWQ tensors.
    empty = torch.empty(0, dtype=torch.int32, device=tm_weight.device)
    layer.qweight = Parameter(empty, requires_grad=False)
    layer.qzeros = Parameter(empty.clone(), requires_grad=False)
    layer.scales = Parameter(
        torch.empty(0, dtype=tm_scales.dtype, device=tm_weight.device),
        requires_grad=False,
    )
    logger.info_once("SM70: AWQ dense layers use TurboMind W4A16.")


def apply_sm70_turbomind_awq_linear(
    layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    k, n = layer.sm70_awq_shape
    x_2d = x.reshape(-1, x.shape[-1])
    ops = sm70_turbomind.ops()
    if (
        x_2d.shape[0] == _PREFILL_DENSE_M
        and layer.sm70_awq_prefill_workspace is not None
    ):
        dense_weight = layer.sm70_awq_prefill_workspace[: k * n].view(k, n)
        ops.awq_dequantize_out(
            dense_weight,
            layer.sm70_awq_weight,
            layer.sm70_awq_scales,
            layer.sm70_awq_group_size,
        )
        out = torch.mm(x_2d, dense_weight)
    else:
        out = torch.empty((x_2d.shape[0], n), dtype=x.dtype, device=x.device)
        ops.awq_gemm_out(
            out,
            x_2d,
            layer.sm70_awq_weight,
            layer.sm70_awq_scales,
            layer.sm70_awq_group_size,
            layer.sm70_awq_k_ld,
            layer.sm70_awq_q_ld,
            False,
        )
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], n)

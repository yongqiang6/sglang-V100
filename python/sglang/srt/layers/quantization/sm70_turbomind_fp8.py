"""V100 (SM70) block-FP8 W8A16 linear layers on TurboMind, with QPN8 decode.

Layers with [128, 128] block-FP8 weights run TurboMind's W8A16 GEMM. Selected
Qwen3.8-27B-FP8 TP4 shapes additionally use QPN8 for decode (M <= 8) and an
exact dense FP16 GEMM for long prefills. Without the TurboMind library, FP8
layers keep weight-only FP8 Marlin.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.ops.gemm import sm70_qpn8, sm70_turbomind
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_BLOCK_SIZE = [128, 128]
_BACKENDS = ("auto", "turbomind", "marlin")

# [N, K] checkpoint shapes (Qwen3.8-27B-FP8, TP4) measured faster with QPN8 at
# M <= 8: down_proj, gate_up_proj, in_proj_qkvz, out_proj/o_proj.
_QPN8_SHAPES = {(5120, 4352), (8704, 5120), (4096, 5120), (5120, 1536)}
# The word-parallel decoder (NACC argument 3) was validated on these only.
_QPN8_FASTDEC_SHAPES = {(5120, 4352), (8704, 5120), (4096, 5120)}
_QPN8_SPLIT_K = 16
_QPN8_NACC = 1
_QPN8_FASTDEC_NACC = 3

# [K, N] TurboMind shapes (same model) where an exact dense FP16 GEMM beats
# W8A16 for M >= 3920, admitted by 1Cat's real-weight bitwise gates.
_PREFILL_DENSE_SHAPES = {(5120, 8704), (4352, 5120), (1536, 5120)}
_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(k * n for k, n in _PREFILL_DENSE_SHAPES)
# The layers keep only a CUDA address; this cache owns the 85 MiB workspace, so it
# stays out of torch.compile and CUDA-graph inputs.
_PREFILL_DENSE_WORKSPACES: dict[int, torch.Tensor] = {}


def sm70_turbomind_fp8_supported(weight_block_size: Optional[list[int]]) -> bool:
    backend = envs.SGLANG_SM70_FP8_BACKEND.get().lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"SGLANG_SM70_FP8_BACKEND must be one of {_BACKENDS}, got {backend!r}."
        )
    if backend == "marlin":
        return False
    block = None if weight_block_size is None else list(weight_block_size)
    supported = block == _BLOCK_SIZE and sm70_turbomind.ops() is not None
    if backend == "turbomind" and not supported:
        raise RuntimeError(
            "SGLANG_SM70_FP8_BACKEND=turbomind needs an SM70 GPU, the TurboMind "
            "library (scripts/v100/setup_v100_turbomind.sh) and "
            "weight_block_size=[128, 128]."
        )
    return supported


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
                "SM70 FP8: no memory for the exact dense prefill workspace; "
                "using TurboMind W8A16 for long prefills."
            )
            return None
        _PREFILL_DENSE_WORKSPACES[device] = workspace
    return workspace


def prepare_sm70_turbomind_fp8_linear(layer: torch.nn.Module) -> None:
    if layer.orig_dtype != torch.float16:
        raise RuntimeError(
            f"SM70 TurboMind FP8 requires FP16 activations, got {layer.orig_dtype}."
        )
    weight = layer.weight.data
    scales = layer.weight_scale_inv.data.to(torch.float32).contiguous()
    n, k = weight.shape

    # QPN8 packs the checkpoint layout, so capture it before the TurboMind repack.
    layer.sm70_qpn8_codes = None
    if (
        envs.SGLANG_SM70_FP8_DECODE_QPN8.get()
        and (n, k) in _QPN8_SHAPES
        and n % 128 == 0
        and k % 128 == 0
    ):
        layer.sm70_qpn8_codes = sm70_qpn8.prepack_codes(weight.view(torch.uint8))
        layer.sm70_qpn8_scales = sm70_qpn8.group_scales(scales, k)
        layer.sm70_qpn8_nacc = (
            _QPN8_FASTDEC_NACC
            if (n, k) in _QPN8_FASTDEC_SHAPES
            and envs.SGLANG_SM70_FP8_QPN8_FASTDEC.get()
            else _QPN8_NACC
        )

    ops = sm70_turbomind.ops()
    tm_weight, tm_scales, meta = ops.fp8_prepare(weight, scales, 128, False)
    layer.weight = Parameter(tm_weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(tm_scales, requires_grad=False)
    layer.sm70_fp8_k_ld = int(meta[0].item())
    layer.sm70_fp8_q_ld = int(meta[1].item())

    layer.sm70_prefill_dense_workspace_ptr = 0
    if (
        envs.SGLANG_SM70_FP8_PREFILL_EXACT_DENSE.get()
        and tuple(tm_weight.shape) in _PREFILL_DENSE_SHAPES
    ):
        workspace = _prefill_dense_workspace(tm_weight)
        if workspace is not None:
            layer.sm70_prefill_dense_workspace_ptr = workspace.data_ptr()
    logger.info_once("SM70: block-FP8 linear layers use TurboMind W8A16.")


def apply_sm70_turbomind_fp8_linear(
    layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    n = layer.output_size_per_partition
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    m = x_2d.shape[0]
    if layer.sm70_qpn8_codes is not None and 0 < m <= sm70_qpn8.MAX_M:
        out = sm70_qpn8.qpn8_linear(
            x_2d,
            layer.sm70_qpn8_codes,
            layer.sm70_qpn8_scales,
            n,
            x_2d.shape[-1],
            _QPN8_SPLIT_K,
            layer.sm70_qpn8_nacc,
        )
    else:
        ops = sm70_turbomind.ops()
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        if layer.sm70_prefill_dense_workspace_ptr:
            ops.fp8_prefill_dispatch(
                out,
                layer.sm70_prefill_dense_workspace_ptr,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
                envs.SGLANG_SM70_FP8_PREFILL_MIN_TOKENS.get(),
            )
        else:
            ops.fp8_gemm(
                out,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
            )
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], n)

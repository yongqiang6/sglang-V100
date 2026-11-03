"""FP16 -> E5M2 KV-cache bytes with saturation, for GPUs without FP8 casts (SM70).

``Tensor.to(torch.float8_e5m2)`` rounds FP16 magnitudes above 57344 (the
largest finite E5M2) to infinity, which then poisons attention. E5M2 is the top
byte of FP16, so the cast is a round-to-nearest-even on the low byte, clamped to
the largest finite code; NaN stays NaN (0x7f).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _fp16_to_e5m2_satfinite_kernel(
    x_ptr,
    out_ptr,
    row_dim,
    x_row_stride,
    scale,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < row_dim
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    if HAS_SCALE:
        x = (x.to(tl.float32) / scale).to(tl.float16)
    bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    sign = (bits >> 8) & 0x80
    magnitude = bits & 0x7FFF
    rounded = magnitude >> 8
    remainder = magnitude & 0xFF
    round_up = (remainder > 0x80) | ((remainder == 0x80) & ((rounded & 1) != 0))
    rounded = tl.minimum(rounded + round_up.to(tl.int32), 0x7B)
    code = sign | rounded
    is_inf_or_nan = (magnitude >> 10) == 0x1F
    is_nan = is_inf_or_nan & ((magnitude & 0x3FF) != 0)
    code = tl.where(is_inf_or_nan, sign | 0x7B, code)
    code = tl.where(is_nan, 0x7F, code)
    tl.store(out_ptr + row * row_dim + cols, code.to(tl.uint8), mask=mask)


def fp16_to_e5m2_satfinite(
    x: torch.Tensor, scale: Optional[Union[float, torch.Tensor]] = None
) -> torch.Tensor:
    """``x / scale`` as E5M2 bytes (uint8, contiguous, same shape as ``x``).

    ``x`` is FP16; dims after the first form one row, so a per-token K/V view
    with a padded token stride (QKV split) needs no copy.
    """
    if x.dtype != torch.float16:
        raise ValueError(f"expected float16 input, got {x.dtype}")
    if isinstance(scale, torch.Tensor):
        x = x / scale
        scale = None
    rows = x.flatten(1) if x.ndim > 1 else x.view(1, -1)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    if out.numel() == 0:
        return out
    row_dim = rows.shape[1]
    grid = (rows.shape[0], triton.cdiv(row_dim, _BLOCK))
    _fp16_to_e5m2_satfinite_kernel[grid](
        rows,
        out,
        row_dim,
        rows.stride(0),
        1.0 if scale is None else float(scale),
        HAS_SCALE=scale is not None and float(scale) != 1.0,
        BLOCK=_BLOCK,
    )
    return out

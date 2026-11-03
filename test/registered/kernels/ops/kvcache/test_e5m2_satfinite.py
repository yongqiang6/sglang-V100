"""The SM70 E5M2 KV cast must equal torch's round-to-nearest-even cast wherever
that cast stays finite, and saturate where torch would overflow to infinity."""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.e5m2_satfinite import fp16_to_e5m2_satfinite
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Halfway between the largest finite E5M2 (57344) and 2^16: torch rounds to inf above.
_E5M2_OVERFLOW_MIDPOINT = 61440.0
_E5M2_MAX_CODE = 0x7B


def _all_fp16() -> torch.Tensor:
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
    return bits.to(torch.int16).view(torch.float16).reshape(64, 1024)


def test_matches_torch_cast_and_saturates_overflow():
    x = _all_fp16()
    out = fp16_to_e5m2_satfinite(x).reshape(-1)
    flat = x.reshape(-1)
    reference = flat.to(torch.float8_e5m2).view(torch.uint8)

    finite = torch.isfinite(flat)
    in_range = finite & (flat.abs().float() < _E5M2_OVERFLOW_MIDPOINT)
    torch.testing.assert_close(out[in_range], reference[in_range], rtol=0, atol=0)

    saturated = ~in_range & ~torch.isnan(flat)
    assert bool(((out[saturated] & 0x7F) == _E5M2_MAX_CODE).all())
    assert bool(((out[saturated] & 0x80) == (reference[saturated] & 0x80)).all())
    assert bool((out[torch.isnan(flat)] == 0x7F).all())


def test_strided_token_view_with_scale():
    qkv = torch.randn(7, 3 * 4 * 64, dtype=torch.float16, device="cuda") * 1000
    k = qkv[:, : 4 * 64].view(7, 4, 64)
    out = fp16_to_e5m2_satfinite(k, scale=2.0)
    reference = (k.float() / 2.0).half().to(torch.float8_e5m2).view(torch.uint8)
    assert out.shape == k.shape and out.is_contiguous()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

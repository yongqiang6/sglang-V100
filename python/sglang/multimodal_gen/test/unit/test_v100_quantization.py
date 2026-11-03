# SPDX-License-Identifier: Apache-2.0
"""Contracts of the online V100 diffusion quantizations (W4A16 AWQ layout, W8A16)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization import get_quantization_config
from sglang.multimodal_gen.runtime.layers.quantization.v100_w4a16_awq import (
    V100W4A16AWQConfig,
    _pack_awq_columns,
    _unpack_awq_columns,
)
from sglang.multimodal_gen.runtime.layers.quantization.v100_w8a16 import (
    V100W8A16Config,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _resolve_target_param_dtype,
)


def test_pack_order_matches_awq_checkpoints():
    """TurboMind's awq_prepare reads the AutoAWQ nibble interleave; logical columns
    0..7 must pack to 0x75316420 or every weight is silently permuted."""
    logical = torch.arange(8, dtype=torch.uint8).view(1, 8)
    packed = _pack_awq_columns(logical)
    assert packed.dtype == torch.int32
    assert packed.item() == 0x75316420
    values = torch.randint(0, 16, (3, 64), dtype=torch.uint8)
    assert torch.equal(_unpack_awq_columns(_pack_awq_columns(values)), values)


def test_online_quantization_stays_within_half_a_step():
    """Off SM70 the layer dequantizes its AWQ tensors; RTN error per weight must
    stay within half a quantization step of its group."""
    torch.manual_seed(0)
    group_size, k, n = 32, 128, 64
    layer = ReplicatedLinear(
        k,
        n,
        bias=False,
        params_dtype=torch.float16,
        quant_config=V100W4A16AWQConfig(group_size=group_size),
    )
    weight = torch.randn(n, k, dtype=torch.float16)
    layer.weight.data.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.int32
    assert layer.weight.shape == (k, n // 8)
    assert layer.weight_scale.shape == (k // group_size, n)
    assert layer.weight_zero.shape == (k // group_size, n // 8)

    dequant = layer.quant_method._dequantize_checkpoint_weight(layer)
    step = layer.weight_scale.float().t().repeat_interleave(group_size, dim=1)
    error = (dequant.float() - weight.float()).abs()
    # Slack covers the FP16 rounding of the stored scale and the dequantized weight.
    assert torch.all(error <= step * 0.51 + 4e-3)

    x = torch.randn(4, k, dtype=torch.float16)
    out = layer.quant_method.apply(layer, x)
    torch.testing.assert_close(
        out.float(), x.float() @ dequant.float().t(), atol=2e-2, rtol=2e-2
    )


def test_w8a16_stays_within_half_a_step():
    torch.manual_seed(0)
    k, n = 96, 40
    layer = ReplicatedLinear(
        k, n, bias=False, params_dtype=torch.float16, quant_config=V100W8A16Config()
    )
    weight = torch.randn(n, k, dtype=torch.float16)
    layer.weight.data.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (n,)
    dequant = layer.weight.float() * layer.weight_scale.float().unsqueeze(1)
    step = layer.weight_scale.float().unsqueeze(1)
    # Slack covers the FP16 rounding of the stored scale.
    assert torch.all((dequant - weight.float()).abs() <= step * 0.51 + 1e-3)


@pytest.mark.parametrize(
    "name, config_cls",
    [("v100_w4a16_awq", V100W4A16AWQConfig), ("v100_w8a16", V100W8A16Config)],
)
def test_fp16_source_formats_load_the_dit_in_fp16(name, config_cls):
    """The V100 formats quantize FP16 tensors; without the loader hook the
    checkpoint keeps its BF16 dtype and create_weights rejects it."""
    quant_cls = get_quantization_config(name)
    assert quant_cls is config_cls
    server_args = SimpleNamespace()
    assert (
        _resolve_target_param_dtype(
            quant_config=quant_cls(), nunchaku_config=None, server_args=server_args
        )
        == torch.float16
    )

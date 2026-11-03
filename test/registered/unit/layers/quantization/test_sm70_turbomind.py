"""Which layers the SM70 TurboMind paths take. Each path implements a narrow
contract; a layer outside it must keep its default kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import sm70_turbomind
from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.sm70_fp16_moe import sm70_fp16_moe_supported
from sglang.srt.layers.quantization.sm70_turbomind_awq import (
    sm70_turbomind_awq_supported,
)
from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
    sm70_turbomind_fp8_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _library_loaded():
    return patch.object(sm70_turbomind, "ops", lambda: SimpleNamespace())


class TestSm70TurboMindSelection(CustomTestCase):
    def test_fp16_moe_only_takes_layers_it_implements(self):
        """The TurboMind FP16 MoE applies gated SiLU and the router weights only;
        EP, routed scaling, other activations and clamps must stay on Triton."""
        base = dict(num_experts=8, num_local_experts=8, top_k=2)
        unsupported = {
            "expert parallel": dict(num_local_experts=4),
            "routed scaling": dict(routed_scaling_factor=2.5),
            "gelu": dict(activation="gelu"),
            "ungated": dict(is_gated=False),
            "swiglu limit": dict(swiglu_limit=7.0),
            "gemm1 alpha": dict(gemm1_alpha=1.702),
            "router weight on input": dict(apply_router_weight_on_input=True),
            "no combine": dict(no_combine=True),
        }
        with _library_loaded():
            self.assertTrue(
                sm70_fp16_moe_supported(
                    MoeRunnerConfig(**base), torch.float16, with_bias=False
                )
            )
            self.assertFalse(
                sm70_fp16_moe_supported(
                    MoeRunnerConfig(**base), torch.bfloat16, with_bias=False
                )
            )
            self.assertFalse(
                sm70_fp16_moe_supported(
                    MoeRunnerConfig(**base), torch.float16, with_bias=True
                )
            )
            for name, fields in unsupported.items():
                with self.subTest(name):
                    config = MoeRunnerConfig(**{**base, **fields})
                    self.assertFalse(
                        sm70_fp16_moe_supported(config, torch.float16, False)
                    )
            with envs.SGLANG_SM70_FP16_MOE.override(False):
                self.assertFalse(
                    sm70_fp16_moe_supported(
                        MoeRunnerConfig(**base), torch.float16, False
                    )
                )
        with patch.object(sm70_turbomind, "ops", lambda: None):
            self.assertFalse(
                sm70_fp16_moe_supported(MoeRunnerConfig(**base), torch.float16, False)
            )

    def test_fp8_takes_only_128_block_weights(self):
        """TurboMind W8A16 is built for [128, 128] block scales; other layouts keep
        FP8 Marlin, and forcing TurboMind without its prerequisites must fail."""
        with _library_loaded():
            self.assertTrue(sm70_turbomind_fp8_supported([128, 128]))
            self.assertFalse(sm70_turbomind_fp8_supported([64, 64]))
            self.assertFalse(sm70_turbomind_fp8_supported(None))
            with envs.SGLANG_SM70_FP8_BACKEND.override("marlin"):
                self.assertFalse(sm70_turbomind_fp8_supported([128, 128]))
        with patch.object(sm70_turbomind, "ops", lambda: None):
            self.assertFalse(sm70_turbomind_fp8_supported([128, 128]))
            with envs.SGLANG_SM70_FP8_BACKEND.override("turbomind"):
                with self.assertRaises(RuntimeError):
                    sm70_turbomind_fp8_supported([128, 128])

    def test_awq_takes_supported_group_sizes_only(self):
        with _library_loaded():
            for group_size in (32, 64, 128):
                self.assertTrue(sm70_turbomind_awq_supported(group_size))
            for group_size in (-1, 256):
                self.assertFalse(sm70_turbomind_awq_supported(group_size))
            with envs.SGLANG_SM70_AWQ_TURBOMIND.override(False):
                self.assertFalse(sm70_turbomind_awq_supported(128))


if __name__ == "__main__":
    unittest.main()

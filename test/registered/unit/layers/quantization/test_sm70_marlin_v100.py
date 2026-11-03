"""SM70 (V100) Marlin runs on the external marlin_v100 kernels, whose metadata
contract differs from SM80+ Marlin. These cases pin that contract on CPU with
the library replaced by stubs."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.gemm import sm70_marlin_v100
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.quantization import marlin_utils
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.utils import get_scalar_types, pack_cols
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_, scalar_types = get_scalar_types()

# AWQ stores each 8-column group of 4-bit zero points in this interleaved order.
_AWQ_INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]


def _sm70_marlin_v100_active(stack: ExitStack) -> None:
    stack.enter_context(patch.object(marlin_utils, "_is_cuda", True))
    stack.enter_context(patch.object(sm70_marlin_v100, "is_active", lambda: True))


class TestSm70MarlinV100(CustomTestCase):
    def test_awq_zero_points_become_logical_float_zp_times_scale(self):
        """marlin_v100 reads AWQ zero points as fp16 ``zp * scale`` in logical
        column order: the AWQ interleave must be undone before scaling."""
        torch.manual_seed(0)
        groups, size_n = 2, 16
        zp = torch.randint(0, 16, (groups, size_n), dtype=torch.int32, device="cpu")
        scales = torch.rand(groups, size_n, dtype=torch.float16, device="cpu") + 0.5
        interleaved = zp.reshape(-1, 8)[:, _AWQ_INTERLEAVE].reshape(groups, size_n)
        packed = pack_cols(interleaved, 4, groups, size_n)

        with ExitStack() as stack:
            _sm70_marlin_v100_active(stack)
            out = marlin_utils.awq_to_marlin_zero_points(
                packed, groups, size_n, 4, scales=scales
            )

        expected = (zp.float() * scales.float()).half()
        self.assertEqual(out.dtype, torch.float16)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_moe_gemm_zeros_provided_output_for_inactive_routes(self):
        """marlin_v100 skips blocks with a negative expert id and leaves their
        rows untouched, so a caller-provided output must be zeroed first."""
        c = torch.full((4, 64), 7.0, dtype=torch.float16, device="cpu")
        library_gemm = Mock(side_effect=lambda a, c, *args: c)
        tensor = torch.empty(0, device="cpu")
        with ExitStack() as stack:
            _sm70_marlin_v100_active(stack)
            stack.enter_context(
                patch.object(sm70_marlin_v100, "moe_wna16_marlin_gemm", library_gemm)
            )
            moe_wna16_marlin_gemm(
                torch.empty(2, 32, dtype=torch.float16, device="cpu"),
                c,
                tensor,
                None,
                tensor,
                None,
                None,
                None,
                None,
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                moe_block_size=8,
                top_k=2,
                mul_topk_weights=True,
                is_ep=False,
                b_q_type=SimpleNamespace(id=0),
                size_m=2,
                size_n=64,
                size_k=32,
                zero_inactive_output=True,
            )
        library_gemm.assert_called_once()
        self.assertTrue(bool((c == 0).all()))

    def test_sm70_support_excludes_fp4_and_act_order_promotion(self):
        """FP4 Marlin needs an SM70 scale encoding that is not ported, and
        marlin_v100 rejects act-order g_idx, so neither may be selected."""
        self.assertEqual(
            marlin_utils.query_marlin_supported_quant_types(device_capability=70), []
        )
        gptq = {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True}
        with ExitStack() as stack:
            _sm70_marlin_v100_active(stack)
            stack.enter_context(
                patch.object(marlin_utils, "get_device_capability", lambda: (7, 0))
            )
            supported = marlin_utils.query_marlin_supported_quant_types(
                device_capability=70
            )
            promote = GPTQMarlinConfig.is_gptq_marlin_compatible(
                {**gptq, "desc_act": False}
            )
            promote_act_order = GPTQMarlinConfig.is_gptq_marlin_compatible(
                {**gptq, "desc_act": True}
            )
        self.assertIn(scalar_types.uint4b8, supported)
        self.assertIn(scalar_types.uint4, supported)
        self.assertNotIn(scalar_types.float4_e2m1f, supported)
        self.assertTrue(promote)
        self.assertFalse(promote_act_order)


if __name__ == "__main__":
    unittest.main()

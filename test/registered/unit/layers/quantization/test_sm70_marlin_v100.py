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
from sglang.srt.layers.quantization import (
    fp4_utils,
    fp8_utils,
    marlin_utils,
    marlin_utils_fp4,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.utils import get_scalar_types, pack_cols
from sglang.srt.utils.common import temp_set_env
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

    def test_sm70_marlin_support_and_act_order_promotion(self):
        """SM70 Marlin support exists only with marlin_v100 loaded, and act-order
        GPTQ, whose g_idx marlin_v100 rejects, must not be promoted to it."""
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
        self.assertTrue(promote)
        self.assertFalse(promote_act_order)

    def test_sm70_nvfp4_global_scale_is_fp32_and_mxfp4_is_rejected(self):
        """marlin_v100 requires an FP32 NVFP4 global scale (one per expert) with
        the FP16 exponent bias folded in; its MXFP4 path is not wired."""
        global_scale = torch.tensor([0.5, 0.25], dtype=torch.float32, device="cpu")
        sm80 = marlin_utils_fp4._nvfp4_marlin_global_scale(global_scale, torch.half)
        self.assertEqual(sm80.dtype, torch.half)
        self.assertFalse(fp4_utils.nvfp4_uses_marlin((7, 0)))
        self.assertTrue(fp4_utils.nvfp4_uses_marlin((8, 9)))
        self.assertEqual(ModelOptFp4Config.get_min_capability(), 80)
        with ExitStack() as stack:
            _sm70_marlin_v100_active(stack)
            self.assertTrue(fp4_utils.nvfp4_uses_marlin((7, 0)))
            self.assertEqual(ModelOptFp4Config.get_min_capability(), 70)
            sm70 = marlin_utils_fp4._nvfp4_marlin_global_scale(global_scale, torch.half)
            with self.assertRaises(NotImplementedError):
                marlin_utils_fp4._reject_sm70_mxfp4_marlin()
        self.assertEqual(sm70.dtype, torch.float32)
        torch.testing.assert_close(sm70, global_scale * 2**7, rtol=0, atol=0)
        torch.testing.assert_close(sm80.float(), sm70, rtol=0, atol=0)

    def test_sm70_nvfp4_moe_repacks_into_loaded_storage(self):
        """Qwen3.8 Flash Next ran out of memory on 32 GiB V100s during NVFP4 MoE
        post-processing, which held a second copy of the experts. On SM70 the
        Marlin experts and scales must reuse the loaded storage, match the
        out-of-place repack, and skip the FlashInfer-only swizzled scales."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )
        from sglang.srt.runtime_context import get_flags

        experts, hidden, intermediate = 4, 128, 64

        def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
            # Value-dependent, so a slot read after being overwritten shows up.
            return b_q_weight.reshape(size_k // 16, size_n * 2) * 3 + 1

        with ExitStack() as stack:
            _sm70_marlin_v100_active(stack)
            stack.enter_context(
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.MARLIN,
                    a2a_backend=MoeA2ABackend.NONE,
                )
            )
            stack.enter_context(
                patch.object(
                    marlin_utils_fp4, "gptq_marlin_repack", fake_repack, create=True
                )
            )
            stack.enter_context(
                patch.object(
                    marlin_utils_fp4,
                    "marlin_make_workspace",
                    lambda device, max_blocks_per_sm: torch.zeros(1, dtype=torch.int),
                )
            )
            quant_config = Mock(group_size=16, use_per_token_activation=False)
            quant_config.get_name.return_value = "modelopt_fp4"
            method = ModelOptNvFp4FusedMoEMethod(quant_config)
            layer = FusedMoE.__new__(FusedMoE)
            torch.nn.Module.__init__(layer)
            layer.quant_method = method
            layer.moe_runner_config = MoeRunnerConfig(is_gated=True)
            layer.num_local_experts = layer.num_experts = experts
            method.create_weights(
                layer=layer,
                num_experts=experts,
                hidden_size=hidden,
                intermediate_size_per_partition=intermediate,
                params_dtype=torch.float16,
                weight_loader=layer.weight_loader,
            )
            self.assertIsNone(layer.w13_blockscale_swizzled)
            self.assertIsNone(layer.w2_blockscale_swizzled)

            generator = torch.Generator().manual_seed(0)
            for name in ("w13_weight", "w2_weight"):
                data = getattr(layer, name).data
                data.copy_(torch.randint(0, 256, data.shape, generator=generator))
            for name in ("w13_weight_scale", "w2_weight_scale"):
                # Positive E4M3 codes; NVFP4 Marlin scales must be non-negative.
                data = getattr(layer, name).data.view(torch.uint8)
                data.copy_(torch.randint(0x08, 0x70, data.shape, generator=generator))
            layer.w13_weight_scale_2.data.fill_(0.5)
            layer.w2_weight_scale_2.data.fill_(0.25)

            perm = torch.empty(0, dtype=torch.int)
            shapes = {
                "w13": (2 * intermediate, hidden),
                "w2": (hidden, intermediate),
            }
            expected, loaded_ptrs = {}, {}
            for prefix, (size_n, size_k) in shapes.items():
                weight = getattr(layer, f"{prefix}_weight").data
                scale = getattr(layer, f"{prefix}_weight_scale").data
                loaded_ptrs[prefix] = (weight.data_ptr(), scale.data_ptr())
                expected[prefix] = (
                    marlin_utils_fp4._repack_moe_fp4_weight_for_marlin(
                        weight.clone(),
                        num_experts=experts,
                        size_n=size_n,
                        size_k=size_k,
                        perm=perm,
                    ),
                    marlin_utils_fp4._permute_moe_fp4_scales_for_marlin(
                        scale.to(torch.float16),
                        num_experts=experts,
                        size_n=size_n,
                        size_k=size_k,
                        group_size=16,
                        process_scales=marlin_utils_fp4.nvfp4_marlin_process_scales,
                    ),
                )

            method.process_weights_after_loading(layer)

        for prefix, (expected_weight, expected_scale) in expected.items():
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            self.assertEqual((weight.data_ptr(), scale.data_ptr()), loaded_ptrs[prefix])
            torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
            torch.testing.assert_close(
                scale.view(torch.uint8), expected_scale.view(torch.uint8)
            )

    def test_sm70_nvfp4_scales_match_the_decode_kernel_reader(self):
        """The SM70 NVFP4 MoE decode kernel reads the scale for logical column n
        at stored byte {0,2,1,3}[n % 4] of its group of four, as the FP16 bits
        ``byte << 7`` == scale * 128. Every non-negative finite E4M3 code must
        survive SGLang's Marlin scale encoding that way."""
        codes = torch.arange(0x00, 0x7F, dtype=torch.uint8)
        codes = torch.cat([codes, codes.new_zeros(-codes.numel() % 4)])
        logical = codes.view(torch.float8_e4m3fn).reshape(1, -1)
        stored = marlin_utils_fp4.nvfp4_marlin_process_scales(logical.half())
        stored = stored.view(torch.uint8).reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1)
        decoded = (stored.to(torch.int16) << 7).view(torch.float16).float()
        torch.testing.assert_close(
            decoded, logical.float().reshape(-1) * 128, rtol=0, atol=0
        )

    def test_sm70_nvfp4_moe_decode_gates(self):
        """The decode kernel and its top-10 router hard-code one shape and plain
        gated-SiLU / softmax semantics; every other variant must stay on
        Marlin and the Triton router instead of being computed wrongly."""
        from sglang.kernels.ops.moe import sm70_nvfp4_moe_decode as sm70
        from sglang.srt.layers.moe import topk as topk_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            _use_sm70_nvfp4_decode,
        )

        def meta(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="meta")

        def decode_case(tokens=1, weight_dtype=torch.float32, quant=(), config=()):
            quant_info = dict(
                w13_qweight=meta(sm70.W13_SHAPE, torch.int32),
                w2_qweight=meta(sm70.W2_SHAPE, torch.int32),
                w13_scales=meta((512, 160, 320), torch.float8_e4m3fn),
                w2_scales=meta((512, 10, 2560), torch.float8_e4m3fn),
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                weight_bits=4,
                w13_global_scale=meta((512,), torch.float32),
                w2_global_scale=meta((512,), torch.float32),
            )
            return _use_sm70_nvfp4_decode(
                torch.zeros(tokens, 2560, dtype=torch.float16),
                torch.zeros(tokens, 10, dtype=torch.int32),
                torch.zeros(tokens, 10, dtype=weight_dtype),
                MarlinMoeQuantInfo(**{**quant_info, **dict(quant)}),
                MoeRunnerConfig(
                    **{"num_experts": 512, "num_local_experts": 512, **dict(config)}
                ),
            )

        def topk_case(tokens=1, topk=10, renormalize=True, bias=None, shared=0):
            logits = torch.zeros(tokens, 512, dtype=torch.float16)
            return topk_module._use_sm70_topk10_softmax(
                logits, topk, renormalize, bias, shared
            )

        with ExitStack() as stack:
            stack.enter_context(patch.object(topk_module, "_is_cuda", True))
            for name in (
                "sm70_nvfp4_moe_decode_available",
                "sm70_nvfp4_moe_ops_loaded",
            ):
                stack.enter_context(patch.object(sm70, name, lambda: True))

            self.assertTrue(decode_case())
            self.assertTrue(decode_case(tokens=4))
            self.assertTrue(decode_case(tokens=7))
            for label, rejected in {
                "eight tokens": decode_case(tokens=8),
                "bf16 router weights": decode_case(weight_dtype=torch.bfloat16),
                "routed scaling": decode_case(config={"routed_scaling_factor": 2.5}),
                "swiglu clamp": decode_case(config={"swiglu_limit": 7.0}),
                "gemm1 alpha": decode_case(config={"gemm1_alpha": 1.702}),
                "no combine": decode_case(config={"no_combine": True}),
                "expert parallel": decode_case(config={"num_local_experts": 128}),
                "expert map": decode_case(quant={"expert_map": torch.zeros(512)}),
                "w13 bias": decode_case(quant={"w13_bias": torch.zeros(1)}),
                "zero points": decode_case(quant={"w13_qzeros": torch.zeros(1)}),
                "other hidden size": decode_case(
                    quant={"w13_qweight": meta((512, 128, 640), torch.int32)}
                ),
            }.items():
                self.assertFalse(rejected, label)

            self.assertTrue(topk_case())
            self.assertTrue(topk_case(tokens=7))
            for label, rejected in {
                "eight tokens": topk_case(tokens=8),
                "top-8": topk_case(topk=8),
                "no renormalize": topk_case(renormalize=False),
                "correction bias": topk_case(bias=torch.zeros(512)),
                "fused shared expert": topk_case(shared=1),
            }.items():
                self.assertFalse(rejected, label)

    def test_sm70_marlin_moe_prefill_tuning_is_scoped_to_its_shapes(self):
        """marlin_v100 applies the geometry environment overrides to every MoE
        launch, so the prefill tuning must be cleared before any other shape
        runs, and overrides the user set must never be replaced."""
        import os

        names = sm70_marlin_v100._MOE_TUNING_ENV
        fp4_scales = torch.empty(0, dtype=torch.float8_e4m3fn)

        def launch(block, top_k, m, n, k):
            sm70_marlin_v100._apply_moe_tuning(
                sm70_marlin_v100.qwen38_nvfp4_prefill_moe_tuning(
                    fp4_scales, block, top_k, m, n, k
                )
            )
            return tuple(os.environ.get(name) for name in names)

        with ExitStack() as stack:
            stack.enter_context(temp_set_env(**{name: None for name in names}))
            stack.enter_context(
                patch.object(sm70_marlin_v100, "_active_moe_tuning", None)
            )
            stack.enter_context(
                patch.object(sm70_marlin_v100, "_USER_MOE_TUNING", False)
            )
            gate_up = ("32x64x64x4x32x32x32", "1", "vector_words")
            self.assertEqual(launch(32, 10, 1024, 320, 2560), gate_up)
            # Decode gate/up keeps marlin_v100's compiled-in selection.
            self.assertEqual(launch(8, 10, 1, 320, 2560), (None, None, None))
            self.assertEqual(launch(32, 10, 1024, 320, 2560), gate_up)
            self.assertEqual(
                launch(64, 1, 81920, 2560, 160),
                ("64x256x32x4x64x64x32", "1", "vector_words"),
            )
            self.assertEqual(launch(32, 10, 4096, 320, 2560), (None, None, None))
            self.assertEqual(launch(64, 1, 20480, 2560, 160), (None, None, None))
            self.assertIsNone(
                sm70_marlin_v100.qwen38_nvfp4_prefill_moe_tuning(
                    torch.empty(0, dtype=torch.float16), 32, 10, 1024, 320, 2560
                ),
                "FP16-scaled (non-NVFP4) weights",
            )

            stack.enter_context(temp_set_env(**{names[0]: "user-choice"}))
            stack.enter_context(
                patch.object(sm70_marlin_v100, "_USER_MOE_TUNING", True)
            )
            self.assertEqual(launch(32, 10, 1024, 320, 2560)[0], "user-choice")
            self.assertEqual(launch(8, 10, 1, 320, 2560)[0], "user-choice")

    def test_fp8_marlin_on_sm70_requires_marlin_v100(self):
        """FP8 checkpoints load on SM70 only as weight-only Marlin, which needs
        marlin_v100; without it the SM80 floor must stay."""
        config = Fp8Config(is_checkpoint_fp8_serialized=True)
        with patch.object(fp8_utils, "get_device_capability", lambda: (7, 0)):
            self.assertFalse(fp8_utils.can_auto_enable_marlin_fp8())
            self.assertEqual(config.get_min_capability(), 80)
            with ExitStack() as stack:
                _sm70_marlin_v100_active(stack)
                self.assertTrue(fp8_utils.can_auto_enable_marlin_fp8())
                self.assertEqual(config.get_min_capability(), 70)


if __name__ == "__main__":
    unittest.main()

"""Dispatch of the V100 (SM70) decode kernels for Qwen3.8 Flash Next and the
opt-in greedy TP top-1 exchange. Each fast path hard-codes one layout or skips
work other consumers need, so the gates must refuse every input it would
compute wrongly."""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.elementwise import qwen4_gate
from sglang.kernels.ops.gemm import sm70_dense_gemv, sm70_hc_mix, sm70_qwen_fusions
from sglang.srt.environ import envs
from sglang.srt.layers import greedy_tp_top1
from sglang.srt.layers.hyperconnection import GatedResidual
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CudaTensor:
    """The metadata the gates read from a CUDA tensor, without a GPU."""

    def __init__(self, *shape, dtype=torch.float16, contiguous=True, offset=0):
        self.shape = torch.Size(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.is_cuda = True
        self.device = torch.device("cuda", 0)
        self._contiguous = contiguous
        self._ptr = 1 << 20 | offset

    def dim(self):
        return self.ndim

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr


def _on_sm70(sm70: bool = True):
    return patch.object(sm70_dense_gemv, "_is_sm70_index", lambda index: sm70)


class TestSm70DecodeFusionGates(CustomTestCase):
    def test_dense_linear_takes_only_measured_bias_free_aligned_shapes(self):
        x1, x4 = _CudaTensor(1, 2560), _CudaTensor(4, 2560)
        with _on_sm70():
            self.assertTrue(sm70_dense_gemv.supported(x1, _CudaTensor(4096, 2560)))
            self.assertTrue(sm70_dense_gemv.supported(x1, _CudaTensor(62080, 2560)))
            self.assertTrue(sm70_dense_gemv.supported(x4, _CudaTensor(320, 2560)))
            # Longer MTP chains verify up to seven rows on the four-row launch.
            for rows in (3, 5, 6, 7):
                self.assertTrue(
                    sm70_dense_gemv.supported(
                        _CudaTensor(rows, 2560), _CudaTensor(320, 2560)
                    ),
                    rows,
                )
            rejected = {
                "bias": sm70_dense_gemv.supported(
                    x1, _CudaTensor(4096, 2560), bias=_CudaTensor(4096)
                ),
                "bf16": sm70_dense_gemv.supported(
                    _CudaTensor(1, 2560, dtype=torch.bfloat16),
                    _CudaTensor(4096, 2560, dtype=torch.bfloat16),
                ),
                "unaligned weight": sm70_dense_gemv.supported(
                    x1, _CudaTensor(4096, 2560, offset=2)
                ),
                "strided input": sm70_dense_gemv.supported(
                    _CudaTensor(4, 2560, contiguous=False), _CudaTensor(320, 2560)
                ),
                "eight rows": sm70_dense_gemv.supported(
                    _CudaTensor(8, 2560), _CudaTensor(320, 2560)
                ),
                # Only four-row shapes extend to the other row counts.
                "five rows of a two-row shape": sm70_dense_gemv.supported(
                    _CudaTensor(5, 2560), _CudaTensor(1, 2560)
                ),
                "unmeasured shape": sm70_dense_gemv.supported(
                    x1, _CudaTensor(1000, 2560)
                ),
                "unmeasured row count": sm70_dense_gemv.supported(
                    x4, _CudaTensor(1, 2560)
                ),
            }
            with envs.SGLANG_SM70_DENSE_GEMV.override(False):
                rejected["disabled"] = sm70_dense_gemv.supported(
                    x1, _CudaTensor(4096, 2560)
                )
        with _on_sm70(False):
            rejected["not SM70"] = sm70_dense_gemv.supported(
                x1, _CudaTensor(4096, 2560)
            )
        for label, accepted in rejected.items():
            self.assertFalse(accepted, label)

    def test_fusions_take_only_their_rows_and_layouts(self):
        down, up = _CudaTensor(320, 10240), _CudaTensor(10240, 320)
        qkvz, ba = _CudaTensor(4096, 2560), _CudaTensor(24, 2560)
        with _on_sm70():
            self.assertTrue(sm70_hc_mix.mix_supported(_CudaTensor(1, 10240), down, up))
            self.assertTrue(sm70_hc_mix.mix_supported(_CudaTensor(4, 10240), down, up))
            self.assertTrue(sm70_hc_mix.mix_supported(_CudaTensor(7, 10240), down, up))
            self.assertTrue(
                sm70_hc_mix.gate_supported(_CudaTensor(7, 10240), _CudaTensor(4, 10240))
            )
            self.assertTrue(
                sm70_qwen_fusions.qkvzba_supported(_CudaTensor(7, 2560), qkvz, ba)
            )
            self.assertTrue(
                sm70_hc_mix.gate_supported(_CudaTensor(2, 10240), _CudaTensor(4, 10240))
            )
            self.assertTrue(
                sm70_qwen_fusions.qkv_ba_supported(_CudaTensor(1, 2560), qkvz, ba)
            )
            self.assertTrue(
                sm70_qwen_fusions.qkvzba_supported(_CudaTensor(4, 2560), qkvz, ba)
            )
            self.assertTrue(
                sm70_qwen_fusions.gate_up_supported(
                    _CudaTensor(1, 2560), _CudaTensor(320, 2560)
                )
            )
            rejected = {
                "hc 65 rows": sm70_hc_mix.mix_supported(
                    _CudaTensor(65, 10240), down, up
                ),
                "hc gate 65 rows": sm70_hc_mix.gate_supported(
                    _CudaTensor(65, 10240), _CudaTensor(4, 10240)
                ),
                # GDN chunking is opt-in: cuBLAS reads the weights once.
                "qkvzba eight rows": sm70_qwen_fusions.qkvzba_supported(
                    _CudaTensor(8, 2560), qkvz, ba
                ),
                # One row's combine computes its own gates.
                "hc gate one row": sm70_hc_mix.gate_supported(
                    _CudaTensor(1, 10240), _CudaTensor(4, 10240)
                ),
                "qkv_ba two rows": sm70_qwen_fusions.qkv_ba_supported(
                    _CudaTensor(2, 2560), qkvz, ba
                ),
                "qkvzba one row": sm70_qwen_fusions.qkvzba_supported(
                    _CudaTensor(1, 2560), qkvz, ba
                ),
                "other ba heads": sm70_qwen_fusions.qkv_ba_supported(
                    _CudaTensor(1, 2560), qkvz, _CudaTensor(32, 2560)
                ),
                "gate_up two rows": sm70_qwen_fusions.gate_up_supported(
                    _CudaTensor(2, 2560), _CudaTensor(320, 2560)
                ),
            }
            with envs.SGLANG_SM70_QWEN_FUSIONS.override(False):
                rejected["disabled"] = sm70_hc_mix.mix_supported(
                    _CudaTensor(1, 10240), down, up
                )
            # Several requests' verify rows run as chunked launches.
            x16 = _CudaTensor(16, 10240)
            self.assertTrue(sm70_hc_mix.mix_supported(x16, down, up))
            # 6 requests x 6 draft tokens.
            self.assertTrue(sm70_hc_mix.mix_supported(_CudaTensor(36, 10240), down, up))
            self.assertTrue(sm70_hc_mix.gate_supported(x16, _CudaTensor(4, 10240)))
            with envs.SGLANG_SM70_HC_MIX_MAX_ROWS.override(7):
                rejected["hc capped"] = sm70_hc_mix.mix_supported(x16, down, up)
                # The cap never drops below one launch's rows.
                self.assertTrue(
                    sm70_hc_mix.mix_supported(_CudaTensor(7, 10240), down, up)
                )
            with envs.SGLANG_SM70_HC_MIX_MAX_ROWS.override(128):
                # The apply kernel's own bound still holds.
                rejected["hc beyond the kernel bound"] = sm70_hc_mix.mix_supported(
                    _CudaTensor(65, 10240), down, up
                )
            with envs.SGLANG_SM70_GDN_PROJ_MAX_ROWS.override(16):
                self.assertTrue(
                    sm70_qwen_fusions.qkvzba_supported(_CudaTensor(16, 2560), qkvz, ba)
                )
                rejected["qkvzba 17 rows"] = sm70_qwen_fusions.qkvzba_supported(
                    _CudaTensor(17, 2560), qkvz, ba
                )
        for label, accepted in rejected.items():
            self.assertFalse(accepted, label)

    def test_chunked_launches_cover_every_row_once(self):
        """Rows beyond one launch are split across launches that write into
        slices of shared outputs; a wrong offset would leave rows unwritten or
        written from another request's input."""
        for launch_rows in (7, 8, 16):
            for rows in range(1, 65):
                chunks = sm70_dense_gemv.verify_row_chunks(rows, launch_rows)
                self.assertEqual(sum(chunks), rows)
                self.assertLessEqual(max(chunks), launch_rows)
                # Each launch re-reads the weights, so use as few as fit.
                self.assertEqual(len(chunks), -(-rows // launch_rows))
                if rows >= 2:
                    # The inject-gate and GDN launches have no one-row form.
                    self.assertGreaterEqual(min(chunks), 2, rows)

        launches = []

        class RowWise:
            """Stands in for a kernel module: each output row depends only on
            its own input row."""

            def __init__(self, count):
                self.count = count

            def _check(self, x):
                self.assert_rows(x.shape[0], self.count)
                launches.append(self.count)

            def down(self, x, weight, *outputs):
                self._check(x)
                outputs[-2 if len(outputs) == 3 else -1].copy_(x[:, :320] + 1)
                if len(outputs) == 3:
                    outputs[-1].copy_(x[:, :32].reshape(-1, 8, 4).float())

            def up(self, activated, x, weight, output):
                self._check(x)
                output.copy_(activated.repeat(1, 8) + x[:, :2560])

            def run(self, x, w, t, qkv, z, b, a):
                self._check(x)
                qkv.copy_(x)
                z.copy_(x[:, :1536] * 2)
                b.copy_(x[:, :12] + 3)
                a.copy_(x[:, :12] + 4)

        RowWise.assert_rows = lambda _, got, want: self.assertEqual(got, want)
        rows = 13
        x = torch.arange(rows * 10240, dtype=torch.float32).reshape(rows, 10240)
        with (
            patch.object(sm70_hc_mix, "_module", RowWise),
            patch.object(sm70_hc_mix, "_gate_module", RowWise),
            patch.object(sm70_qwen_fusions, "_qkvzba_module", RowWise),
        ):
            activated = sm70_hc_mix.hc_down(x, None)
            self.assertTrue(torch.equal(activated, x[:, :320] + 1))
            self.assertTrue(
                torch.equal(
                    sm70_hc_mix.hc_up(activated, x, None),
                    activated.repeat(1, 8) + x[:, :2560],
                )
            )
            gated, partials = sm70_hc_mix.hc_down_with_gate(x, None, None)
            self.assertTrue(torch.equal(gated, activated))
            self.assertTrue(torch.equal(partials, x[:, :32].reshape(-1, 8, 4)))
            hidden = x[:, :2560].contiguous()
            qkv, z, b, a = sm70_qwen_fusions.qkvzba(hidden, None, None)
            self.assertTrue(torch.equal(qkv, hidden))
            self.assertTrue(torch.equal(z.flatten(1), hidden[:, :1536] * 2))
            self.assertTrue(torch.equal(b, hidden[:, :12] + 3))
            self.assertTrue(torch.equal(a, hidden[:, :12] + 4))
        # One launch holds seven rows by default.
        self.assertEqual(launches, [7, 6] * 4)

        def hc_launches(launch_rows, rows=13):
            del launches[:]
            with (
                patch.object(sm70_hc_mix, "_module", RowWise),
                envs.SGLANG_SM70_HC_MIX_LAUNCH_ROWS.override(launch_rows),
            ):
                sm70_hc_mix.hc_down(torch.zeros(rows, 10240), None)
            return list(launches)

        self.assertEqual(hc_launches(16), [13])
        # The setting is held to what the kernels are templated for.
        self.assertEqual(hc_launches(64, rows=20), [10, 10])
        self.assertEqual(hc_launches(4), [7, 6])

    def test_batched_hc_mix_runs_as_gemms(self):
        """Batches beyond one request's verify rows take two GEMMs; the result
        must be the mix the row kernels define, with weight_up's rows taken
        stream-major, and combine() must then compute its own gates."""
        torch.manual_seed(0)
        rows = 9
        x = torch.randn(rows, 10240).half()
        down = (torch.randn(320, 10240) / 100).half()
        up = (torch.randn(10240, 320) / 4).half()
        activated = torch.nn.functional.silu(x.double() @ down.double().T / 4)
        gate = torch.sigmoid(activated @ up.double().T)
        expected = (gate.view(rows, 4, 2560) * x.double().view(rows, 4, 2560)).mean(1)
        mixed = sm70_hc_mix.hc_mix_gemm(x, down, up)
        self.assertEqual(mixed.dtype, torch.float16)
        self.assertEqual(tuple(mixed.shape), (rows, 2560))
        self.assertLess((mixed.double() - expected).abs().max().item(), 5e-3)
        # The other pairing of gate and stream is far outside rounding.
        swapped = (
            gate.view(rows, 2560, 4).transpose(1, 2) * x.double().view(rows, 4, 2560)
        ).mean(1)
        self.assertGreater((swapped - expected).abs().max().item(), 0.2)

        self.assertTrue(sm70_hc_mix.use_gemm(8))
        self.assertFalse(sm70_hc_mix.use_gemm(7))
        with envs.SGLANG_SM70_HC_MIX_GEMM_MIN_ROWS.override(65):
            self.assertFalse(sm70_hc_mix.use_gemm(64))

        layer = SimpleNamespace(
            _sm70_inject_gate=True,
            input_mix_weight_down=SimpleNamespace(weight=down),
            input_mix_weight_up=SimpleNamespace(weight=up),
            block_inject_weight=SimpleNamespace(weight=_CudaTensor(4, 10240)),
        )
        with patch.object(sm70_hc_mix, "hc_down_with_gate") as row_kernels:
            result, partials = GatedResidual._sm70_mix_forward(layer, x)
        row_kernels.assert_not_called()
        self.assertIsNone(partials)
        self.assertTrue(torch.equal(result, mixed))

    def test_row_bounds_match_the_compiled_kernels(self):
        """Each kernel asserts its own row bound when it is built on a V100, so
        a Python gate that admits more rows fails there and nowhere else."""
        from sglang.kernels.jit.utils import KERNEL_PATH
        from sglang.kernels.ops.attention.tilelang_fa_v100 import _decode_cuda
        from sglang.kernels.ops.moe import sm70_nvfp4_moe_decode

        def bounds(relative_path, pattern):
            source = (KERNEL_PATH / "csrc" / relative_path).read_text()
            found = {int(bound) for bound in re.findall(pattern, source)}
            self.assertTrue(found, relative_path)
            return found

        rows = {sm70_dense_gemv.MAX_VERIFY_ROWS}
        for path in ("gemm/sm70_small_gemm.cuh", "gemm/sm70_qwen_gdn.cuh"):
            self.assertEqual(bounds(path, r"M <= (\d+)\);"), rows, path)
        for path in ("elementwise/sm70_hc_batch.cuh", "elementwise/sm70_hc_gate.cuh"):
            self.assertEqual(
                bounds(path, r"M <= (\d+)\);"), {sm70_hc_mix.MAX_LAUNCH_ROWS}, path
            )
        self.assertEqual(
            bounds("elementwise/sm70_hc_gate.cuh", r"rows\.unwrap\(\) <= (\d+)"),
            {sm70_hc_mix.MAX_ROWS},
        )
        self.assertEqual(
            bounds("moe/sm70_nvfp4_moe_decode.cu", r"kMaxTokens = (\d+);"),
            {sm70_nvfp4_moe_decode.MAX_TOKENS},
        )
        self.assertEqual(
            bounds("attention/sm70_qsa_combine.cuh", r"batch\.unwrap\(\) <= (\d+)"),
            {_decode_cuda._NATIVE_COMBINE_MAX_BATCH},
        )

    def test_gdn_verify_small_value_tile_covers_every_mtp_chain_length(self):
        """The 48-CTA default launch is the slow one on V100, so each supported
        MTP chain length must select the 192-CTA tile, and nothing else may."""
        from sglang.kernels.ops.attention.linear import (
            fused_sigmoid_gating_recurrent as recurrent,
        )

        def launch(steps, n=1, verify=True):
            return recurrent._select_recurrent_launch_config(
                n, 4, 12, 128, 128, False, verify, cache_steps=steps
            )

        with patch.object(recurrent, "_is_sm70", True):
            for steps in range(2, sm70_dense_gemv.MAX_VERIFY_ROWS + 1):
                self.assertEqual(launch(steps), (8, 1), steps)
            self.assertEqual(launch(8), (32, 1))
            self.assertEqual(launch(4, n=2), (32, 1))
            self.assertEqual(launch(4, verify=False), (32, 1))
        with patch.object(recurrent, "_is_sm70", False):
            self.assertEqual(launch(4), (32, 1))

    def test_fp16_ple_gate_value_fuses_only_on_sm70(self):
        """The FP16 rounding branch of the PLE gate/value kernel matches eager
        FP16 only as validated on V100; BF16 keeps its own contract."""
        gate, value = _CudaTensor(1, 4, 1), _CudaTensor(1, 2560)
        with _on_sm70():
            self.assertTrue(qwen4_gate.can_fuse_qwen4_gate_value(gate, value))
            with envs.SGLANG_SM70_QWEN_FUSIONS.override(False):
                self.assertFalse(qwen4_gate.can_fuse_qwen4_gate_value(gate, value))
        with _on_sm70(False):
            self.assertFalse(qwen4_gate.can_fuse_qwen4_gate_value(gate, value))
            bf16 = dict(dtype=torch.bfloat16)
            self.assertTrue(
                qwen4_gate.can_fuse_qwen4_gate_value(
                    _CudaTensor(1, 4, 1, **bf16), _CudaTensor(1, 2560, **bf16)
                )
            )

    def test_sm70_inject_gate_partials_are_applied_without_a_gate_stream(self):
        """Partials from the SM70 mix must reach the SM70 apply kernel; the
        other partials path waits on a gate stream SM70 never creates."""
        layer = SimpleNamespace(_gate_stream=None, hidden_size=2560)
        layer.combine = lambda output, residuals: GatedResidual.combine(
            layer, output, residuals
        )
        residuals = (torch.zeros(2, 10240), torch.zeros(2, 10240), torch.zeros(2, 8, 4))
        fp16_norm = SimpleNamespace(
            group_size=2560, weight=torch.zeros(1, dtype=torch.float16)
        )
        with patch.object(
            sm70_hc_mix, "hc_apply_gate", lambda output, residual, partials: "sm70"
        ):
            self.assertEqual(
                GatedResidual.combine(layer, torch.zeros(2, 2560), residuals), "sm70"
            )
            self.assertEqual(
                GatedResidual.combine_and_normalize(
                    layer, torch.zeros(2, 2560), residuals, fp16_norm
                ),
                ("sm70", None),
            )

    def test_greedy_tp_top1_matches_the_gathered_argmax(self):
        """Exchanging one candidate per rank must give the argmax of the full
        gathered logits: padding rows ignored, the lower id on a tie between
        ranks, and ids past FP16 precision exact through the FP32 exchange."""
        shard_rows, padded_rows = 5, 6
        # Rank 1 starts past 2048, where FP16 no longer holds odd integers.
        shard_starts = (0, 37985)
        hidden = torch.zeros(3, 1)
        full = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0],
                [0.0, 0.1, 0.2, 0.3, 0.7, 0.7, 0.1, 0.2, 0.3, 0.4],
                [-3.0, -2.0, -4.0, -5.0, -6.0, -1.0, -7.0, -8.0, -9.0, -1.5],
            ],
            dtype=torch.float16,
        )
        candidates = []
        for rank, start in enumerate(shard_starts):
            shard = SimpleNamespace(
                org_vocab_start_index=start, num_org_elements=shard_rows
            )
            lm_head = SimpleNamespace(shard_indices=shard)
            local = torch.full((3, padded_rows), 100.0, dtype=torch.float16)
            local[:, :shard_rows] = full[:, rank * shard_rows : (rank + 1) * shard_rows]
            candidates.append(
                greedy_tp_top1.local_top1_candidates(
                    hidden, lm_head, lambda states, head, local=local: local
                )
            )
        ids = greedy_tp_top1.pick_global_top1(
            torch.cat(candidates, -1), len(shard_starts)
        )
        expected = [
            shard_starts[col // shard_rows] + col % shard_rows
            for col in full.float().argmax(-1).tolist()
        ]
        self.assertEqual(expected, [1, 4, 37985])
        self.assertEqual(ids.tolist(), expected)

    def test_greedy_tp_top1_only_for_batches_nothing_else_reads(self):
        def batch(**overrides):
            sampling = dict(
                is_all_greedy=True,
                grammars=None,
                has_custom_logit_processor=False,
                logit_bias=None,
                vocab_mask=None,
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                penalizer_orchestrator=SimpleNamespace(is_required=False),
                has_watermark_candidates=False,
            )
            fields = dict(
                is_prefill_only=False,
                forward_mode=ForwardMode.DECODE,
                spec_algorithm=None,
                global_num_tokens_cpu=None,
                return_logprob=False,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
            )
            for key, value in overrides.items():
                (sampling if key in sampling else fields)[key] = value
            return SimpleNamespace(sampling_info=SimpleNamespace(**sampling), **fields)

        self.assertFalse(greedy_tp_top1.is_strict_greedy_batch(batch()), "opt-in")
        with envs.SGLANG_V100_GREEDY_TP_TOP1.override(True):
            self.assertTrue(greedy_tp_top1.is_strict_greedy_batch(batch()))
            self.assertTrue(
                greedy_tp_top1.is_strict_greedy_batch(
                    batch(forward_mode=ForwardMode.EXTEND)
                )
            )
            rejected = {
                "sampling": dict(is_all_greedy=False),
                "logprobs": dict(return_logprob=True),
                "top logprobs": dict(top_logprobs_nums=[0, 2]),
                "token logprobs": dict(token_ids_logprobs=[None, [3]]),
                "verify": dict(forward_mode=ForwardMode.TARGET_VERIFY),
                "speculative": dict(
                    spec_algorithm=SimpleNamespace(is_none=lambda: False)
                ),
                "MLP-sync padding": dict(global_num_tokens_cpu=[1, 1]),
                "prefill only": dict(is_prefill_only=True),
                "grammar": dict(grammars=[None, object()]),
                "penalties": dict(
                    penalizer_orchestrator=SimpleNamespace(is_required=True)
                ),
                "logit bias": dict(logit_bias=torch.zeros(1)),
                "vocab mask": dict(vocab_mask=torch.zeros(1)),
                "custom processor": dict(has_custom_logit_processor=True),
                "watermark": dict(has_watermark_candidates=True),
            }
            for label, overrides in rejected.items():
                self.assertFalse(
                    greedy_tp_top1.is_strict_greedy_batch(batch(**overrides)), label
                )

    def test_gdn_verify_split_only_for_target_verify_ratio_three(self):
        """The fused split writes the ratio-three [Q, K, V | Z | B | A] layout;
        decode keeps the raw projections for the fused projection/conv path."""
        weights = (_CudaTensor(4096, 2560), _CudaTensor(24, 2560))

        def layer(value_dim=6144):
            return SimpleNamespace(
                key_dim=2048,
                value_dim=value_dim,
                num_v_heads=value_dim // 128,
                head_k_dim=128,
                head_v_dim=128,
                attn_tp_size=4,
                _sm70_input_proj_weights=lambda hidden_states: weights,
            )

        def split(mode, value_dim=6144):
            batch = SimpleNamespace(forward_mode=mode)
            return Qwen3_5GatedDeltaNet._sm70_verify_split(
                layer(value_dim), _CudaTensor(4, 2560), batch
            )

        with (
            _on_sm70(),
            patch.object(sm70_qwen_fusions, "qkvzba", lambda x, qkvz, ba: "split"),
        ):
            self.assertEqual(split(ForwardMode.TARGET_VERIFY), "split")
            self.assertIsNone(split(ForwardMode.DECODE))
            self.assertIsNone(split(ForwardMode.EXTEND))
            self.assertIsNone(split(ForwardMode.TARGET_VERIFY, value_dim=4096))


if __name__ == "__main__":
    unittest.main()

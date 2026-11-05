"""Routing of QSA (Qwen3.8 Flash Next) on V100 (SM70). The SM70 routes read
different inputs than the upstream ones, so each guards the case it must not take."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import qsa_indexer, sm70
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.mem_cache import qsa_kv_pool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_CPU = torch.device("cpu")


def _batch(seq_lens, extend_lens, mode=ForwardMode.EXTEND, original_mode=None):
    return SimpleNamespace(
        forward_mode=mode,
        _original_forward_mode=original_mode,
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens_cpu=torch.tensor(extend_lens),
    )


class TestQSASM70Routing(CustomTestCase):
    def test_dense_prefill_only_without_a_cached_prefix(self):
        """Dense prefill attends over the current chunk's K/V only; with a cached
        prefix, past the length limit or outside plain prefill it must not run,
        or the indexer would skip the selection the sparse path needs."""
        with patch.object(sm70, "is_sm70", lambda device: True):
            self.assertTrue(sm70.use_dense_prefill(_batch([5, 8], [5, 8]), _CPU, 256))
            excluded = {
                "cached prefix": (_batch([9, 8], [5, 8]), 256),
                "over the limit": (_batch([8193], [8193]), 256),
                "mixed batch": (_batch([5], [5], ForwardMode.MIXED), 256),
                "draft extend": (
                    _batch([5], [5], original_mode=ForwardMode.DRAFT_EXTEND_V2),
                    256,
                ),
                "head dim 128": (_batch([5], [5]), 128),
            }
            for name, (batch, head_dim) in excluded.items():
                with self.subTest(name):
                    self.assertFalse(sm70.use_dense_prefill(batch, _CPU, head_dim))
            with envs.SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS.override(0):
                self.assertFalse(sm70.use_dense_prefill(_batch([5], [5]), _CPU, 256))
        self.assertFalse(sm70.use_dense_prefill(_batch([5], [5]), _CPU, 256))

    def test_sm70_indexer_cache_is_fp16(self):
        """Volta's MMA reads FP16 only: the indexer cache must not be BF16 or FP8."""
        with patch.object(qsa_kv_pool, "_sm70", lambda: True):
            self.assertEqual(
                qsa_kv_pool.resolve_qsa_indexer_dtype("auto"), torch.float16
            )
            for name in ("bfloat16", "fp8_e4m3"):
                with self.subTest(name), self.assertRaises(ValueError):
                    qsa_kv_pool.resolve_qsa_indexer_dtype(name)
        self.assertEqual(qsa_kv_pool.resolve_qsa_indexer_dtype("auto"), torch.bfloat16)

    def test_decode_query_padding_matches_the_scorer(self):
        """The SM70 CUDA scorer reads exactly 4 query heads, the Volta TileLang
        MMA needs multiples of 16 and other GPUs multiples of 8."""
        exact = SimpleNamespace(index_n_heads=4, index_head_dim=128)
        other = SimpleNamespace(index_n_heads=4, index_head_dim=64)
        with patch.object(qsa_indexer, "is_sm70", lambda device: True):
            self.assertEqual(QSAIndexer._decode_query_heads(exact, _CPU), 4)
            self.assertEqual(QSAIndexer._decode_query_heads(other, _CPU), 16)
        self.assertEqual(QSAIndexer._decode_query_heads(exact, _CPU), 8)


if __name__ == "__main__":
    unittest.main()

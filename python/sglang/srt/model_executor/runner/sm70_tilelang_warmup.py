"""Compile the SM70 TileLang prefill kernels before serving.

The TileLang paged prefill kernel is specialized per batch size, and a cold
compile takes seconds, so the first prefill at each batch size would otherwise
stall a live request. Run one EXTEND dummy forward per power-of-two batch size up
to max_running_requests; other batch sizes still compile on first use (and are
then served from TileLang's on-disk cache).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

# One page per request: the kernel key depends on batch size, not token count.
_WARMUP_TOKENS_PER_REQ = 16


def _uses_tilelang_v100(attn_backend) -> bool:
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.tilelang_v100_backend import (
        TileLangV100AttnBackend,
    )

    # Hybrid linear-attention models wrap the full-attention backend.
    if isinstance(attn_backend, HybridLinearAttnBackend):
        attn_backend = attn_backend.full_attn_backend
    return isinstance(attn_backend, TileLangV100AttnBackend)


def maybe_warmup_sm70_tilelang_prefill(runner: BaseRunner) -> None:
    mr = runner.model_runner
    if (
        mr.is_draft_worker
        or not mr.is_generation
        or mr.spec_algorithm.is_speculative()
        or not _uses_tilelang_v100(mr.attn_backend)
    ):
        return
    max_bs = max(1, min(mr.max_running_requests, mr.req_to_token_pool.size))
    batch_sizes = []
    bs = 1
    while bs <= max_bs:
        batch_sizes.append(bs)
        bs *= 2
    log_info_on_rank0(
        logger,
        f"SM70 TileLang prefill warmup for batch sizes {batch_sizes}.",
    )
    buffers = runner._alloc_dummy_decode_buffers(
        batch_sizes[-1],
        num_tokens_per_req=_WARMUP_TOKENS_PER_REQ,
        allocate_logits_buffer=False,
    )
    for batch_size in batch_sizes:
        runner._dummy_run(
            batch_size=batch_size,
            buffers=buffers,
            forward_mode_override=ForwardMode.EXTEND,
            extend_num_tokens_per_req=_WARMUP_TOKENS_PER_REQ,
        )

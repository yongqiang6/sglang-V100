"""QSA attention routes for V100 (SM70), as validated by the sglang-V100 fork.

Upstream QSA attention needs FlashAttention for decode (FA2 is SM80+, FA4 is
Hopper/Blackwell), and SM70 has no FP8 dot product. On SM70:

- A prefill without a cached prefix, up to
  ``SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS`` tokens, runs dense causal
  attention with the TileLang D256 kernel. The fork measured the sparse Triton
  gather at 29.5 ms per layer against 3.55 ms dense at the Qwen3.8 TP4 4K shape.
  The indexer still compresses keys but skips the selection.
- When the context limit fits that threshold, decode reads the paged E5M2 cache
  densely through the grouped TileLang decoder.
- Otherwise, prefill with a single cached prefix and decode/MTP verify, at the
  exact Qwen3.8 Flash Next TP4 shape over an E5M2 cache, read the selected
  keys straight from the cache with the SM70 CUDA kernels.
- Any other decode runs the Triton packed-decode kernel in place of
  FlashAttention, as on ROCm.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.mqa import is_sm70
from sglang.srt.model_executor.forward_mode import ForwardMode

logger = logging.getLogger(__name__)

# [heads, head_dim] of the query and of the KV cache rows the CUDA kernels read.
_EXACT_Q_SHAPE = (6, 256)
_EXACT_KV_SHAPE = (1, 256)
# The dense TileLang prefill and decode kernels are head-dim-256 only.
_DENSE_HEAD_DIM = 256
_DENSE_DECODE_PAGE_SIZE = 16


def _dense_limit() -> int:
    return envs.SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS.get()


def use_dense_prefill(forward_batch, device: torch.device, head_dim: int) -> bool:
    """Whether this ordinary, prefix-free prefill runs dense attention on SM70.

    The model and the backend both ask, so the indexer skips exactly the
    selections the backend will not read. It reads CPU lengths only, so it
    never synchronizes the GPU.
    """
    if (
        not is_sm70(device)
        or head_dim != _DENSE_HEAD_DIM
        or forward_batch.forward_mode != ForwardMode.EXTEND
        or forward_batch._original_forward_mode not in (None, ForwardMode.EXTEND)
    ):
        return False
    limit = _dense_limit()
    lengths = forward_batch.seq_lens_cpu
    extend_lengths = forward_batch.extend_seq_lens_cpu
    return (
        limit > 0
        and lengths is not None
        and extend_lengths is not None
        and len(lengths) > 0
        and len(lengths) == len(extend_lengths)
        and all(int(n) == int(e) for n, e in zip(lengths, extend_lengths))
        and max(int(n) for n in lengths) <= limit
    )


def dense_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    extend_lens: list[int],
    softmax_scale: float,
) -> torch.Tensor:
    """Dense causal prefill over request-major packed Q/K/V without a prefix."""
    from sglang.kernels.ops.attention.tilelang_fa_v100._kernels_dense_d256 import (
        get_dense_prefix_d256_kernel,
    )

    if sum(extend_lens) != q.shape[0]:
        raise ValueError(
            "QSA dense-prefill packed rows do not match extend lengths: "
            f"rows={q.shape[0]}, extend={sum(extend_lens)}"
        )
    kernel = get_dense_prefix_d256_kernel(q.shape[1], k.shape[1])
    outputs = []
    row_start = 0
    # The kernel takes one sequence per launch; split at the packed boundaries.
    for extend_len in extend_lens:
        row_end = row_start + extend_len
        if extend_len:
            outputs.append(
                kernel(
                    q[row_start:row_end].contiguous(),
                    k[row_start:row_end].contiguous(),
                    v[row_start:row_end].contiguous(),
                    0,
                    softmax_scale,
                )
            )
        row_start = row_end
    if not outputs:
        return q.new_empty(q.shape)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def _exact_cache_shape(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    return (
        q.is_cuda
        and is_sm70(q.device)
        and q.dtype == torch.float16
        and q.ndim == 3
        and q.shape[0] > 0
        and tuple(q.shape[1:]) == _EXACT_Q_SHAPE
        and topk_indices.ndim == 2
        and topk_indices.shape[0] == q.shape[0]
        and topk_indices.shape[1] > 0
        and k_buffer.dtype == torch.float8_e5m2
        and v_buffer.dtype == k_buffer.dtype
        and k_buffer.ndim == 3
        and v_buffer.shape == k_buffer.shape
        and tuple(k_buffer.shape[1:]) == _EXACT_KV_SHAPE
    )


def can_use_cache_prefill(q, k_buffer, v_buffer, forward_batch, topk_indices) -> bool:
    return (
        forward_batch.forward_mode.is_extend()
        and forward_batch.req_pool_indices.numel() == 1
        and _exact_cache_shape(q, k_buffer, v_buffer, topk_indices)
    )


def cache_prefill(
    q, k_buffer, v_buffer, req_to_token, forward_batch, topk_indices, softmax_scale
) -> torch.Tensor:
    from sglang.kernels.ops.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_prefill,
    )

    return sm70_cuda_qsa_prefill(
        q.contiguous(),
        k_buffer,
        v_buffer,
        req_to_token,
        forward_batch.req_pool_indices,
        topk_indices,
        forward_batch.seq_lens,
        softmax_scale,
    )


def can_use_cache_decode(
    q, k_buffer, v_buffer, topk_indices, num_rows: int, decode_like: bool
) -> bool:
    return (
        decode_like
        and q.shape[0] == num_rows
        and _exact_cache_shape(q, k_buffer, v_buffer, topk_indices)
    )


def cache_decode(
    q, k_buffer, v_buffer, req_to_token, req_indices, topk_indices, seq_lens, scale
) -> torch.Tensor:
    from sglang.kernels.ops.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_decode,
    )

    return sm70_cuda_qsa_decode(
        q, k_buffer, v_buffer, req_to_token, req_indices, topk_indices, seq_lens, scale
    )


def can_use_dense_decode(
    q, k_buffer, v_buffer, *, is_decode: bool, max_context_len: int, page_size: int
) -> bool:
    """Whole-context decode for E5M2 caches that fit the dense threshold.

    Reading the paged cache once is cheaper than materializing ~2K random
    selections in FP16 there, and keeps full-attention semantics.
    """
    limit = _dense_limit()
    return (
        is_decode
        and limit > 0
        and 0 < max_context_len <= limit
        and page_size == _DENSE_DECODE_PAGE_SIZE
        and q.is_cuda
        and is_sm70(q.device)
        and q.dtype == torch.float16
        and q.shape[-1] == _DENSE_HEAD_DIM
        and k_buffer.dtype == torch.float8_e5m2
        and v_buffer.dtype == k_buffer.dtype
        and k_buffer.ndim in (3, 4)
        and v_buffer.ndim == k_buffer.ndim
        and k_buffer.shape[-2:] == v_buffer.shape[-2:]
        and k_buffer.shape[-1] == _DENSE_HEAD_DIM
        and q.shape[1] == _EXACT_Q_SHAPE[0] * k_buffer.shape[-2]
    )


class DenseDecode:
    """Builds the dense decoder's page table from the request token map."""

    def __init__(self, max_context_len: int, page_size: int, device) -> None:
        # The first token slot of every page a request can reach.
        self.page_starts = torch.arange(
            0, max_context_len, page_size, dtype=torch.long, device=device
        )
        self.page_size = page_size

    def __call__(
        self, q, k_buffer, v_buffer, req_to_token, req_indices, seq_lens, layer
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.tilelang_fa_v100 import (
            grouped_decode_forward,
        )

        page_slots = req_to_token[
            req_indices.long().unsqueeze(1), self.page_starts.unsqueeze(0)
        ]
        page_table = torch.div(page_slots, self.page_size, rounding_mode="floor").to(
            torch.int32
        )
        if k_buffer.ndim == 3:
            k_buffer = k_buffer.view(
                -1, self.page_size, k_buffer.shape[-2], k_buffer.shape[-1]
            )
            v_buffer = v_buffer.view(
                -1, self.page_size, v_buffer.shape[-2], v_buffer.shape[-1]
            )
        output = grouped_decode_forward(
            q,
            k_buffer,
            v_buffer,
            page_table,
            seq_lens,
            softmax_scale=layer.scaling,
            k_scale=layer.k_scale_float if layer.k_scale is not None else 1.0,
            v_scale=layer.v_scale_float if layer.v_scale is not None else 1.0,
        )
        return output.reshape(q.shape[0], -1)

"""SM70 (V100) attention: the Triton backend with TileLang prefill and decode.

Volta has no FA2/FA3; the TileLang kernels in
``sglang.kernels.ops.attention.tilelang_fa_v100`` read the paged KV cache with
coalesced page loads. This backend keeps every piece of the Triton backend
(metadata, CUDA graphs, speculative paths, SWA, DCP) and swaps in TileLang only
for eager EXTEND/MIXED prefill and the batch-1 grouped G6/D256 decode. Any layer
or batch outside those shapes runs the inherited Triton path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# The TileLang kernels assume 16-token pages (SM70 page_size default).
V100_PAGE_SIZE = 16
_TILELANG_HEAD_DIMS = (64, 128, 256, 512)
_FP8_KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


class TileLangExtendMetadata(msgspec.Struct, kw_only=True):
    """Per-forward page table for the TileLang paged prefill kernel."""

    # [batch, ceil(context_len / page_size)] int32 physical page ids; the fixed
    # width keeps the TileLang compile key stable across requests.
    page_table: torch.Tensor
    # [batch] int32 total KV length (prefix + extend) per request.
    seq_lens: torch.Tensor
    # [batch + 1] int32 cumulative extend lengths.
    query_start_loc: torch.Tensor
    # [batch] int32 cached prefix length per request.
    prefix_kv_lens: torch.Tensor
    max_seq_len: int


def tilelang_v100_available() -> bool:
    """Whether the SM70 TileLang kernels can be imported in this process."""
    try:
        import tilelang  # noqa: F401

        from sglang.kernels.ops.attention.tilelang_fa_v100 import (  # noqa: F401
            paged_forward,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("SM70 TileLang attention unavailable, using Triton: %s", e)
        return False
    return True


class TileLangV100AttnBackend(TritonAttnBackend):
    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)
        self._max_pages = (self.max_context_len + V100_PAGE_SIZE - 1) // V100_PAGE_SIZE
        self._tilelang_md: Optional[TileLangExtendMetadata] = None
        kv_dtype = model_runner.kv_cache_dtype
        self._fp8_kv = kv_dtype in _FP8_KV_DTYPES
        self._fp8_prefill_scratch_enabled = (
            self._fp8_kv and envs.SGLANG_V100_FP8_PREFILL_SCRATCH.get()
        )
        self._grouped_decode_enabled = envs.SGLANG_V100_GROUPED_DECODE.get()
        self._grouped_decode_page_table = torch.zeros(
            1, self._max_pages, dtype=torch.int32, device=self.device
        )
        self._fp8_scratch: Optional[tuple[torch.Tensor, ...]] = None
        self._fp8_scratch_key: Optional[tuple[int, int, int]] = None
        if self.page_size != V100_PAGE_SIZE:
            logger.warning(
                "SM70 TileLang attention expects page_size=%d, got %d; "
                "prefill and decode stay on Triton.",
                V100_PAGE_SIZE,
                self.page_size,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._tilelang_md = None
        if self._batch_uses_tilelang_extend(forward_batch):
            self._tilelang_md = self._build_extend_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # Graph-replayed forwards never take the eager TileLang extend path.
        self._tilelang_md = None
        super().init_forward_metadata_out_graph(forward_batch, in_capture)

    def _batch_uses_tilelang_extend(self, forward_batch: ForwardBatch) -> bool:
        mode = forward_batch.forward_mode
        return (
            self.page_size == V100_PAGE_SIZE
            and mode.is_extend_or_draft_extend_or_mixed()
            and self.dcp_size == 1
            and not self.use_mla
            and self.dllm_attention is None
            and (not self._fp8_kv or self._fp8_prefill_scratch_enabled)
            and forward_batch.out_cache_loc is not None
        )

    def _build_extend_metadata(
        self, forward_batch: ForwardBatch
    ) -> Optional[TileLangExtendMetadata]:
        bs = forward_batch.batch_size
        if bs == 0:
            return None
        prefix_lens = forward_batch.extend_prefix_lens[:bs].to(torch.int32)
        extend_lens = forward_batch.extend_seq_lens[:bs].to(torch.int32)
        # Warmup batches may carry a decode-oriented seq_len fill value.
        seq_lens = torch.maximum(
            forward_batch.seq_lens[:bs].to(torch.int32), prefix_lens + extend_lens
        )
        if forward_batch.seq_lens_cpu is not None:
            seq_lens_cpu = torch.maximum(
                forward_batch.seq_lens_cpu[:bs].to(torch.int64),
                torch.tensor(
                    [
                        p + e
                        for p, e in zip(
                            forward_batch.extend_prefix_lens_cpu,
                            forward_batch.extend_seq_lens_cpu,
                        )
                    ],
                    dtype=torch.int64,
                ),
            )
            seq_lens_sum = int(seq_lens_cpu.sum())
            max_seq_len = int(seq_lens_cpu.max())
        else:
            seq_lens_sum = int(seq_lens.sum().item())
            max_seq_len = int(seq_lens.max().item())
        if seq_lens_sum == 0:
            return None

        # Physical slot of every token (prefix and this forward's writes): the
        # plan reads up to forward_batch.seq_lens, which includes the extend.
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        kv_indices = torch.empty(seq_lens_sum, dtype=torch.int64, device=self.device)
        physical = self.runner_ids.pack_read_stream(
            forward_batch.iteration_ids,
            req_pool_indices=forward_batch.req_pool_indices[:bs],
            seq_lens=seq_lens,
            indptr=kv_indptr,
            out=kv_indices,
            num_token_blocks=None,
            kv_lens_sum=seq_lens_sum,
        )
        if not physical:
            return None

        # Each 16-token logical page lives in one physical page, so the slot of
        # its first token names the page.
        max_pages = (max_seq_len + V100_PAGE_SIZE - 1) // V100_PAGE_SIZE
        page_starts = (
            torch.arange(max_pages, dtype=torch.int32, device=self.device)
            * V100_PAGE_SIZE
        )
        valid = page_starts[None, :] < seq_lens[:, None]
        positions = torch.where(valid, kv_indptr[:-1, None] + page_starts[None, :], 0)
        page_table = torch.zeros(
            bs, self._max_pages, dtype=torch.int32, device=self.device
        )
        page_table[:, :max_pages] = torch.where(
            valid, kv_indices[positions.long()] // V100_PAGE_SIZE, 0
        ).to(torch.int32)

        query_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        query_start_loc[1:] = torch.cumsum(extend_lens, dim=0)
        return TileLangExtendMetadata(
            page_table=page_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            prefix_kv_lens=prefix_lens,
            max_seq_len=max_seq_len,
        )

    # ------------------------------------------------------------------
    # Extend
    # ------------------------------------------------------------------
    def _layer_uses_tilelang(self, layer: RadixAttention) -> bool:
        return (
            layer.qk_head_dim == layer.v_head_dim
            and layer.qk_head_dim in _TILELANG_HEAD_DIMS
            and not layer.is_cross_attention
            and layer.attn_type == AttentionType.DECODER
            and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
            and not layer.logit_cap
            and layer.xai_temperature_len <= 0
            and (
                not self.use_sliding_window_kv_pool
                or self.kv_layers.route(layer.layer_id).sub_pool
                != IdSpaceKind.SLIDING_WINDOW
            )
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        md = self._tilelang_md
        if (
            md is None
            or k is None
            or v is None
            or sinks is not None
            or score_mod is not None
            or aux_tensors is not None
            or getattr(forward_batch, "_attn_output", None) is not None
            or torch.cuda.is_current_stream_capturing()
            or not self._layer_uses_tilelang(layer)
        ):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                sinks=sinks,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
            )
        return self._forward_extend_tilelang(
            q, k, v, layer, forward_batch, save_kv_cache, md
        )

    def _forward_extend_tilelang(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        md: TileLangExtendMetadata,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.tilelang_fa_v100 import paged_forward

        # The kernel reads the extend tokens back from the cache.
        if save_kv_cache:
            self._set_kv_buffer(
                forward_batch,
                layer,
                KVWriteLoc.for_batch(forward_batch, swa_loc=None),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        num_tokens = q.shape[0]
        head_dim = layer.qk_head_dim
        # TileLang specializes on strides; a fused-QKV view of q is not packed.
        q3 = q.reshape(num_tokens, layer.tp_q_head_num, head_dim).contiguous()
        k_cache, v_cache = self.kv_layers.read(layer.layer_id).kv()
        # [num_pages, page_size, num_kv_heads, head_dim] view of the flat pool.
        k_cache = k_cache.view(-1, V100_PAGE_SIZE, k_cache.shape[-2], k_cache.shape[-1])
        v_cache = v_cache.view(-1, V100_PAGE_SIZE, v_cache.shape[-2], v_cache.shape[-1])

        page_table = md.page_table
        logical_dense_kv = False
        if k_cache.dtype in _FP8_KV_DTYPES:
            k_cache, v_cache, page_table = self._dequant_fp8_pages(
                k_cache, v_cache, layer, md
            )
            logical_dense_kv = md.page_table.shape[0] == 1

        k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
        out = torch.empty(
            num_tokens, layer.tp_q_head_num, head_dim, dtype=q.dtype, device=q.device
        )
        paged_forward(
            q3,
            k_cache,
            v_cache,
            page_table,
            md.seq_lens,
            md.query_start_loc,
            md.prefix_kv_lens,
            out=out,
            block_size=V100_PAGE_SIZE,
            softmax_scale=layer.scaling,
            causal=True,
            num_kv_heads=layer.tp_k_head_num,
            k_scale=k_scale,
            v_scale=v_scale,
            max_seq_len_hint=md.max_seq_len,
            logical_dense_kv=logical_dense_kv,
        )
        return out.reshape(num_tokens, layer.tp_q_head_num * head_dim)

    def _dequant_fp8_pages(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        md: TileLangExtendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dequantize each active page once into reusable FP16 scratch; the
        TileLang paged kernel cannot take an FP8 operand on SM70."""
        from sglang.kernels.ops.attention.tilelang_fa_v100 import gather_fp8_paged_kv

        batch = md.page_table.shape[0]
        total_pages = batch * self._max_pages
        scratch_k, scratch_v, logical_pages = self._fp8_scratch_for(
            total_pages, layer.tp_k_head_num, layer.qk_head_dim
        )
        gather_fp8_paged_kv(
            k_cache, v_cache, md.page_table, md.seq_lens, scratch_k, scratch_v
        )
        return scratch_k, scratch_v, logical_pages.view(batch, self._max_pages)

    def _fp8_scratch_for(
        self, total_pages: int, num_kv_heads: int, head_dim: int
    ) -> tuple[torch.Tensor, ...]:
        key = self._fp8_scratch_key
        if key is None or key[1:] != (num_kv_heads, head_dim) or key[0] < total_pages:
            # Grow to a power of two so a mix of batch sizes allocates once.
            capacity = 1 << (max(1, total_pages) - 1).bit_length()
            scratch_k = torch.empty(
                capacity,
                V100_PAGE_SIZE,
                num_kv_heads,
                head_dim,
                dtype=torch.float16,
                device=self.device,
            )
            self._fp8_scratch = (
                scratch_k,
                torch.empty_like(scratch_k),
                torch.arange(capacity, dtype=torch.int32, device=self.device),
            )
            self._fp8_scratch_key = (capacity, num_kv_heads, head_dim)
        return tuple(t[:total_pages] for t in self._fp8_scratch)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def _can_use_grouped_decode(
        self, q: torch.Tensor, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        if not (
            self._grouped_decode_enabled
            and self.page_size == V100_PAGE_SIZE
            and self.dcp_size == 1
            and not self.use_mla
            and forward_batch.batch_size == 1
            and q.dtype == torch.float16
            and layer.tp_k_head_num in (1, 2, 4)
            and layer.tp_q_head_num == 6 * layer.tp_k_head_num
            and layer.qk_head_dim == 256
            and layer.v_head_dim == 256
            and self._layer_uses_tilelang(layer)
        ):
            return False
        k_cache, v_cache = self.kv_layers.read(layer.layer_id).kv()
        return (
            k_cache.dtype == v_cache.dtype
            and k_cache.dtype in (torch.float16, *_FP8_KV_DTYPES)
            and k_cache.stride(-1) == 1
            and v_cache.stride(-1) == 1
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        if (
            sinks is None
            and score_mod is None
            and aux_tensors is None
            and self._can_use_grouped_decode(q, layer, forward_batch)
        ):
            return self._forward_grouped_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )
        return super().forward_decode(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            sinks=sinks,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )

    def _forward_grouped_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        """Grouped G6/D256 split-KV decode over Triton's KV ordering metadata."""
        from sglang.kernels.ops.attention.tilelang_fa_v100 import (
            grouped_decode_forward,
        )

        if save_kv_cache:
            self._set_kv_buffer(
                forward_batch,
                layer,
                KVWriteLoc.for_batch(forward_batch, swa_loc=None),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        # Batch 1: every 16th entry of Triton's token list starts a page. Eager
        # metadata is runtime-sized; graph metadata is capacity-sized, fixing
        # the launch grid while seq_lens bounds the active partitions.
        kv_indices = self.forward_metadata.kv_indices
        token_capacity = min(kv_indices.numel(), self.max_context_len)
        page_starts = kv_indices[:token_capacity:V100_PAGE_SIZE]
        page_table = self._grouped_decode_page_table
        torch.div(
            page_starts,
            V100_PAGE_SIZE,
            rounding_mode="floor",
            out=page_table[0, : page_starts.numel()],
        )

        k_cache, v_cache = self.kv_layers.read(layer.layer_id).kv()
        k_cache = k_cache.view(-1, V100_PAGE_SIZE, k_cache.shape[-2], k_cache.shape[-1])
        v_cache = v_cache.view(-1, V100_PAGE_SIZE, v_cache.shape[-2], v_cache.shape[-1])
        result = grouped_decode_forward(
            q.reshape(1, layer.tp_q_head_num, layer.qk_head_dim),
            k_cache,
            v_cache,
            page_table,
            forward_batch.seq_lens[:1],
            softmax_scale=layer.scaling,
            k_scale=layer.k_scale_float if layer.k_scale is not None else 1.0,
            v_scale=layer.v_scale_float if layer.v_scale is not None else 1.0,
        )
        return result.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

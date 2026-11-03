"""Dense packed variable-length FlashAttention for SM70 (V100) in TileLang.

Adapted from TileLang's variable-length MHA forward example. Contract: FP16
``[tokens, heads, 128]`` Q/K/V, int32 cumulative sequence lengths, non-causal
self-attention, FP32 softmax accumulation. MiniMax-H3's packed attention uses it
through the diffusion ``tilelang_fa_v100`` backend.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
HEAD_DIM = 128


@tilelang.jit(out_idx=[6], pass_configs=_PASS_CONFIGS)
def _dense_varlen_kernel_func(
    batch: int,
    heads: int,
    dim: int,
    *,
    block_m: int = 64,
    block_n: int = 32,
    num_stages: int = 0,
    threads: int = 256,
):
    total_tokens = T.dynamic("total_tokens")

    @T.prim_func
    def main(
        query: T.Tensor([total_tokens, heads, dim], T.float16),
        key: T.Tensor([total_tokens, heads, dim], T.float16),
        value: T.Tensor([total_tokens, heads, dim], T.float16),
        cu_seqlens: T.Tensor([batch + 1], T.int32),
        max_seqlen: T.int32,
        softmax_scale: T.float32,
        output: T.Tensor([total_tokens, heads, dim], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(max_seqlen, block_m), heads, batch, threads=threads
        ) as (query_block, head, sequence):
            query_shared = T.alloc_shared([block_m, dim], T.float16)
            key_shared = T.alloc_shared([block_n, dim], T.float16)
            value_shared = T.alloc_shared([block_n, dim], T.float16)
            probability_shared = T.alloc_shared([block_m, block_n], T.float16)
            scores = T.alloc_fragment([block_m, block_n], T.float32)
            scores_fp16 = T.alloc_fragment([block_m, block_n], T.float16)
            output_acc = T.alloc_fragment([block_m, dim], T.float32)
            row_max = T.alloc_fragment([block_m], T.float32)
            previous_row_max = T.alloc_fragment([block_m], T.float32)
            row_sum = T.alloc_fragment([block_m], T.float32)
            score_sum = T.alloc_fragment([block_m], T.float32)
            rescale = T.alloc_fragment([block_m], T.float32)

            sequence_start = cu_seqlens[sequence]
            sequence_stop = cu_seqlens[sequence + 1]
            sequence_length = sequence_stop - sequence_start
            query_start = sequence_start + query_block * block_m
            softmax_scale_log2 = softmax_scale * 1.44269504

            T.clear(query_shared)
            for row, column in T.Parallel(block_m, dim):
                if query_start + row < sequence_stop:
                    query_shared[row, column] = query[query_start + row, head, column]

            T.fill(output_acc, 0)
            T.fill(row_max, -T.infinity(T.float32))
            T.fill(row_sum, 0)

            for key_block in T.Pipelined(
                T.ceildiv(sequence_length, block_n), num_stages=num_stages
            ):
                T.clear(key_shared)
                for row, column in T.Parallel(block_n, dim):
                    key_offset = key_block * block_n + row
                    if key_offset < sequence_length:
                        key_shared[row, column] = key[
                            sequence_start + key_offset, head, column
                        ]

                for row, column in T.Parallel(block_m, block_n):
                    scores[row, column] = T.if_then_else(
                        (query_block * block_m + row < sequence_length)
                        & (key_block * block_n + column < sequence_length),
                        0,
                        -T.infinity(T.float32),
                    )
                T.gemm(
                    query_shared,
                    key_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(row_max, previous_row_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for row in T.Parallel(block_m):
                    row_max[row] = T.if_then_else(
                        row_max[row] == -T.infinity(T.float32),
                        0,
                        T.max(row_max[row], previous_row_max[row]),
                    )
                    rescale[row] = T.exp2(
                        previous_row_max[row] * softmax_scale_log2
                        - row_max[row] * softmax_scale_log2
                    )
                    row_sum[row] *= rescale[row]

                for row, column in T.Parallel(block_m, dim):
                    output_acc[row, column] *= rescale[row]
                for row, column in T.Parallel(block_m, block_n):
                    scores[row, column] = T.exp2(
                        scores[row, column] * softmax_scale_log2
                        - row_max[row] * softmax_scale_log2
                    )
                T.reduce_sum(scores, score_sum, dim=1)
                for row in T.Parallel(block_m):
                    row_sum[row] += score_sum[row]

                T.clear(value_shared)
                for row, column in T.Parallel(block_n, dim):
                    value_offset = key_block * block_n + row
                    if value_offset < sequence_length:
                        value_shared[row, column] = value[
                            sequence_start + value_offset, head, column
                        ]
                for row, column in T.Parallel(block_m, block_n):
                    probability_shared[row, column] = T.cast(
                        scores[row, column], T.float16
                    )
                T.copy(probability_shared, scores_fp16)
                T.gemm(
                    scores_fp16,
                    value_shared,
                    output_acc,
                    policy=T.GemmWarpPolicy.Square,
                )

            for row, column in T.Parallel(block_m, dim):
                if query_start + row < sequence_stop:
                    output[query_start + row, head, column] = T.cast(
                        output_acc[row, column] / row_sum[row], T.float16
                    )

    return main


@lru_cache(maxsize=None)
def _get_kernel(batch: int, heads: int, dim: int):
    return _dense_varlen_kernel_func(batch=batch, heads=heads, dim=dim)


def supports(query: torch.Tensor, causal: bool) -> bool:
    return (
        not causal
        and query.is_cuda
        and query.dtype == torch.float16
        and query.ndim == 3
        and query.shape[-1] == HEAD_DIM
        and torch.cuda.get_device_capability(query.device) == (7, 0)
    )


def tilelang_flash_attn_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Non-causal packed attention; Q/K/V are [tokens, heads, 128] FP16."""
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("TileLang V100 attention needs matching Q/K/V shapes.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must describe at least one sequence.")
    cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
    kernel = _get_kernel(cu_seqlens.numel() - 1, query.shape[1], query.shape[2])
    return kernel(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        cu_seqlens,
        int(max_seqlen),
        float(softmax_scale),
    )

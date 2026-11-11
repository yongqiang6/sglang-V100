"""Exact greedy decoding under tensor parallelism without gathering the vocabulary.

Opt-in for V100 (``SGLANG_V100_GREEDY_TP_TOP1``), from the sglang-V100 fork.
Each TP rank takes the argmax of its own vocabulary shard, and the ranks
exchange one (value, global id) pair per row instead of all-gathering the
[rows, vocab] logits. The result is the argmax of the gathered logits with the
lowest id on ties: shards are in vocabulary order, and ``torch.max`` and
``argmax`` both return the first maximum.

Only batches where nothing else reads the logits qualify, and only eager
steps: CUDA graphs pass a logits buffer, which disqualifies the step. A
sampling observer also reads the logits, so do not combine this with one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# FP32 holds every integer up to 2**24 exactly, so one FP32 tensor carries
# both the logit and the token id through the exchange.
_MAX_EXACT_TOKEN_ID = 1 << 24


def is_strict_greedy_batch(forward_batch: ForwardBatch) -> bool:
    """Whether the opt-in is set and only the argmax token of each row is read."""
    if not envs.SGLANG_V100_GREEDY_TP_TOP1.get() or forward_batch.is_prefill_only:
        return False
    mode = forward_batch.forward_mode
    if mode.is_target_verify() or not (mode.is_decode() or mode.is_extend()):
        return False
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and not spec_algorithm.is_none():
        return False
    # MLP-sync padding slices the logits after the forward.
    if forward_batch.global_num_tokens_cpu is not None:
        return False
    if forward_batch.return_logprob:
        return False
    if forward_batch.top_logprobs_nums and any(forward_batch.top_logprobs_nums):
        return False
    if forward_batch.token_ids_logprobs and any(
        ids is not None for ids in forward_batch.token_ids_logprobs
    ):
        return False
    sampling = forward_batch.sampling_info
    # Beam rows always sample at temperature 1.0 (beam_search/fork.py), so this
    # also keeps out beam search, which reads the logits after sampling.
    if sampling is None or not sampling.is_all_greedy:
        return False
    penalizer = sampling.penalizer_orchestrator
    return not (
        (sampling.grammars and any(g is not None for g in sampling.grammars))
        or sampling.has_custom_logit_processor
        or sampling.logit_bias is not None
        or sampling.vocab_mask is not None
        or sampling.acc_additive_penalties is not None
        or sampling.acc_scaling_penalties is not None
        or (penalizer is not None and penalizer.is_required)
        or sampling.has_watermark_candidates
    )


def lm_head_supported(hidden_states: torch.Tensor, lm_head) -> bool:
    """An FP16 V100 vocabulary shard without added tokens, ids exact in FP32."""
    from sglang.kernels.ops.gemm.sm70_dense_gemv import is_sm70
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if not (
        isinstance(lm_head, VocabParallelEmbedding)
        and hidden_states.is_cuda
        # A captured or compiled graph would replay this batch's decision for
        # later batches, whatever their sampling parameters.
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
        and hidden_states.dtype == torch.float16
        and lm_head.weight.dtype == torch.float16
        and lm_head.weight.device == hidden_states.device
        and is_sm70(hidden_states.device)
    ):
        return False
    shard = lm_head.shard_indices
    return (
        lm_head.num_added_embeddings == 0
        and shard.num_org_elements > 0
        and lm_head.weight.shape[0] >= shard.num_org_elements
        and shard.org_vocab_end_index <= _MAX_EXACT_TOKEN_ID
    )


def local_top1_candidates(
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    compute_lm_head: Callable[[torch.Tensor, VocabParallelEmbedding], torch.Tensor],
) -> torch.Tensor:
    """This rank's best (logit, global token id) per row, FP32 [rows, 2]."""
    shard = lm_head.shard_indices
    # Rows past num_org_elements are vocabulary padding.
    local_logits = compute_lm_head(hidden_states, lm_head)[:, : shard.num_org_elements]
    values, indices = torch.max(local_logits, dim=-1)
    global_ids = indices + shard.org_vocab_start_index
    return torch.stack((values.float(), global_ids.float()), dim=-1)


def pick_global_top1(gathered: torch.Tensor, tp_size: int) -> torch.Tensor:
    """The global argmax ids from the ranks' candidates, [rows, 2 * tp_size]."""
    candidates = gathered.reshape(gathered.shape[0], tp_size, 2)
    best_rank = candidates[..., 0].argmax(dim=-1, keepdim=True)
    return torch.gather(candidates[..., 1], 1, best_rank).squeeze(1).long()


def greedy_tp_top1(
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    compute_lm_head: Callable[[torch.Tensor, VocabParallelEmbedding], torch.Tensor],
    tp_size: int,
) -> torch.Tensor:
    """Global greedy token ids for ``hidden_states``; int64 [rows]."""
    candidates = local_top1_candidates(hidden_states, lm_head, compute_lm_head)
    return pick_global_top1(
        tensor_model_parallel_all_gather(candidates, dim=-1), tp_size
    )

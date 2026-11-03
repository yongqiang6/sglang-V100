# SPDX-License-Identifier: Apache-2.0
"""TileLang dense attention for V100 (SM70), opt in with
``--attention-backend tilelang_fa_v100``.

Packed varlen calls with FP16 ``[tokens, heads, 128]`` non-causal Q/K/V (the
MiniMax-H3 contract) run the TileLang kernel; every other call takes the SDPA
path this backend inherits.
"""

import torch

from sglang.kernels.ops.attention import tilelang_dense_varlen_sm70
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class TileLangFlashAttentionV100Backend(SDPABackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TILELANG_FA_V100

    @staticmethod
    def get_impl_cls() -> type["TileLangFlashAttentionV100Impl"]:
        return TileLangFlashAttentionV100Impl


class TileLangFlashAttentionV100Impl(SDPAImpl):
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if not tilelang_dense_varlen_sm70.supports(query, self.causal):
            return super().forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return tilelang_dense_varlen_sm70.tilelang_flash_attn_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            softmax_scale=self.softmax_scale,
        )

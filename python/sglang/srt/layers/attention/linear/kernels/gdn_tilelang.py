"""SM70 (V100) GDN backend on the TileLang kernels in
``sglang.kernels.ops.attention.linear.gdn_*tilelang_sm70``.

Selected with ``--linear-attn-prefill-backend tilelang`` and/or
``--linear-attn-decode-backend tilelang``. Target verify stays on Triton, and
decode calls carrying ReplaySSM ring caches fall back to Triton.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

_REPLAYSSM_KWARGS = ("replayssm_d", "replayssm_k", "replayssm_g")


class TileLangGDNKernel(LinearAttnKernelBase):
    """SM70 persistent GDN kernels: packed decode and chunked prefill."""

    supports_packed_decode = True
    supports_target_verify = False

    def __init__(self):
        self._triton = TritonGDNKernel()
        # Persistent so a captured decode graph keeps reading the same buffer.
        self._decode_query_start_loc: dict[tuple, torch.Tensor] = {}

    def _decode_cu_seqlens(self, batch: int, device: torch.device) -> torch.Tensor:
        key = (batch, device)
        cu_seqlens = self._decode_query_start_loc.get(key)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device)
            self._decode_query_start_loc[key] = cu_seqlens
        return cu_seqlens

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if any(kwargs.get(name) is not None for name in _REPLAYSSM_KWARGS):
            return self._triton.packed_decode(
                mixed_qkv,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                **kwargs,
            )
        from sglang.kernels.ops.attention.linear.gdn_tilelang_sm70 import (
            _run_packed_gdn,
        )

        out = _run_packed_gdn(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=self._decode_cu_seqlens(
                mixed_qkv.shape[0], mixed_qkv.device
            ),
        )
        # [B, HV, V] -> [1, B, HV, V], the layout of the other decode kernels.
        return out.unsqueeze(0)

    def decode(self, *args, **kwargs) -> torch.Tensor:
        return self._triton.decode(*args, **kwargs)

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        store_checkpoints: bool = False,
        **kwargs,
    ) -> tuple:
        """Returns (output, None, checkpoints). With store_checkpoints the
        per-64-token states are [1, num_chunks, HV, V, K], the layout the mamba
        prefix-cache tracker reads from the Triton kernel."""
        from sglang.kernels.ops.attention.linear.gdn_chunked_tilelang_sm70 import (
            chunked_gdn_sm70,
        )

        out, checkpoints = chunked_gdn_sm70(
            q,
            k,
            v,
            g,
            beta,
            scale=k.shape[-1] ** -0.5,
            state=ssm_states,
            state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            store_checkpoints=store_checkpoints,
        )
        return out, None, checkpoints

    def target_verify(self, *args, **kwargs) -> torch.Tensor:
        return self._triton.target_verify(*args, **kwargs)

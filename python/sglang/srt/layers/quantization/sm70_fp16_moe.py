"""V100 (SM70) FP16 MoE on TurboMind's s884 grouped GEMMs.

Routed tokens are permuted per expert, run through a gated-SiLU gate/up GEMM
and a down GEMM, then unpermuted with the router weights. Layers using features
this path does not implement keep the default Triton runner.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.ops.gemm import sm70_turbomind
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

# Decode batches up to this many tokens reuse per-layer buffers (CUDA-graph
# friendly); larger prefill batches allocate transiently, since retaining a
# prefill-sized set per MoE layer exhausts VRAM before the first request.
_PERSISTENT_MAX_TOKENS = 32
# Buffer leading dims: per token, per routed (token, expert) slot, or whole.
_TOKEN_ROWS = (
    "output",
    "inverse",
    "ids32",
    "token_expert",
    "permuted_experts",
    "sorted_rows",
    "ids_for_sort",
)
_SLOT_ROWS = ("permuted", "gate_up", "sorted_output", "permuted_idx")


def sm70_fp16_moe_supported(
    runner_config: MoeRunnerConfig, params_dtype: torch.dtype, with_bias: bool
) -> bool:
    """True on SM70 with the TurboMind library for layers this path covers."""
    if not envs.SGLANG_SM70_FP16_MOE.get():
        return False
    if params_dtype != torch.float16 or with_bias:
        return False
    if not (runner_config.activation == "silu" and runner_config.is_gated):
        return False
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
        or runner_config.swiglu_limit is not None
        or runner_config.apply_router_weight_on_input
        or runner_config.no_combine
        or runner_config.routed_scaling_factor not in (None, 1.0)
        or runner_config.num_experts != runner_config.num_local_experts
    ):
        return False
    return sm70_turbomind.ops() is not None


class SM70FP16MoEMethod(UnquantizedFusedMoEMethod):
    """Batched FP16 expert GEMMs specialized for V100."""

    def __init__(self):
        super().__init__(use_triton_kernels=False)
        self._ops = sm70_turbomind.ops()
        if self._ops is None:
            raise RuntimeError("SM70 TurboMind operators are unavailable.")

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        ops = self._ops
        num_experts, w13_n, hidden_size = layer.w13_weight.shape
        intermediate_size = w13_n // 2
        # TurboMind's gated epilogue reads gate/up rows interleaved.
        gate = layer.w13_weight[:, :intermediate_size]
        up = layer.w13_weight[:, intermediate_size:]
        w13 = torch.stack((gate, up), dim=2).reshape(-1, hidden_size)
        w13_tm, w13_ld = ops.f16_prepare(w13)
        w2_tm, w2_ld = ops.f16_prepare(layer.w2_weight.reshape(-1, intermediate_size))
        w13_tm = w13_tm.reshape(num_experts, 2 * intermediate_size, hidden_size)
        w2_tm = w2_tm.reshape(num_experts, hidden_size, intermediate_size)

        layer.w13_tm_weight = Parameter(w13_tm, requires_grad=False)
        layer.w2_tm_weight = Parameter(w2_tm, requires_grad=False)
        layer.w13_weight = None
        layer.w2_weight = None
        layer.w13_weight_ptrs = Parameter(
            ops.f16_moe_build_ptrs(w13_tm, int(w13_ld[0].item()), num_experts)[0],
            requires_grad=False,
        )
        layer.w2_weight_ptrs = Parameter(
            ops.f16_moe_build_ptrs(w2_tm, int(w2_ld[0].item()), num_experts)[0],
            requires_grad=False,
        )

        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_size = hidden_size
        layer.sm70_intermediate_size = intermediate_size
        layer.sm70_top_k = self.moe_runner_config.top_k
        layer.sm70_buffers = self._allocate_buffers(
            layer, _PERSISTENT_MAX_TOKENS, layer.w13_tm_weight.device
        )
        logger.info(
            "SM70 FP16 MoE: TurboMind grouped GEMM "
            "(experts=%d hidden=%d intermediate=%d topk=%d)",
            num_experts,
            hidden_size,
            intermediate_size,
            layer.sm70_top_k,
        )

    def _allocate_buffers(
        self, layer: torch.nn.Module, tokens: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        e, h, i, k = (
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            layer.sm70_intermediate_size,
            layer.sm70_top_k,
        )
        slots = tokens * k
        fp16 = dict(dtype=torch.float16, device=device)
        int32 = dict(dtype=torch.int32, device=device)
        return {
            "output": torch.empty(tokens, h, **fp16),
            "permuted": torch.empty(slots, h, **fp16),
            "gate_up": torch.empty(slots, i, **fp16),
            "sorted_output": torch.empty(slots, h, **fp16),
            "offsets64": torch.empty(e + 1, dtype=torch.int64, device=device),
            "offsets32": torch.empty(e + 1, **int32),
            "inverse": torch.empty(tokens, k, **int32),
            "ids32": torch.empty(tokens, k, **int32),
            "token_expert": torch.arange(slots, **int32).view(tokens, k),
            "permuted_idx": torch.empty(slots, **int32),
            "sort_workspace": torch.empty(
                self._ops.moe_permute_sort_workspace_size(slots, e),
                dtype=torch.int8,
                device=device,
            ),
            "permuted_experts": torch.empty(tokens, k, **int32),
            "sorted_rows": torch.empty(tokens, k, **int32),
            "ids_for_sort": torch.empty(tokens, k, **int32),
        }

    def apply(
        self, layer: torch.nn.Module, dispatch_output: StandardDispatchOutput
    ) -> StandardCombineInput:
        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        tokens = x.shape[0]
        k = layer.sm70_top_k
        if tokens <= _PERSISTENT_MAX_TOKENS:
            buffers = dict(layer.sm70_buffers)
            for name in _TOKEN_ROWS:
                buffers[name] = buffers[name][:tokens]
            for name in _SLOT_ROWS:
                buffers[name] = buffers[name][: tokens * k]
        else:
            buffers = self._allocate_buffers(layer, tokens, x.device)
        output = buffers["output"]
        output.zero_()
        if tokens == 0:
            return StandardCombineInput(hidden_states=output)

        ops = self._ops
        buffers["ids32"].copy_(topk.topk_ids)
        ops.moe_permute_with_scratch(
            x,
            buffers["ids32"],
            buffers["token_expert"],
            None,
            layer.sm70_num_experts,
            layer.sm70_num_experts,
            k,
            buffers["permuted"],
            buffers["offsets64"],
            buffers["inverse"],
            buffers["permuted_idx"],
            buffers["sort_workspace"],
            buffers["permuted_experts"],
            buffers["sorted_rows"],
            buffers["ids_for_sort"],
        )
        buffers["offsets32"].copy_(buffers["offsets64"])
        ops.f16_moe_gemm(
            buffers["gate_up"],
            buffers["permuted"],
            buffers["offsets32"],
            layer.w13_weight_ptrs,
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            2 * layer.sm70_intermediate_size,
            True,
        )
        ops.f16_moe_gemm(
            buffers["sorted_output"],
            buffers["gate_up"],
            buffers["offsets32"],
            layer.w2_weight_ptrs,
            layer.sm70_num_experts,
            layer.sm70_intermediate_size,
            layer.sm70_hidden_size,
            False,
        )
        ops.moe_unpermute(
            buffers["sorted_output"],
            topk.topk_weights,
            buffers["inverse"],
            buffers["offsets64"],
            k,
            output,
        )
        return StandardCombineInput(hidden_states=output)

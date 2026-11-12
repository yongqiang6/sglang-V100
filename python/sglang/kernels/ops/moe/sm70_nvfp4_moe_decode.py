"""Small-batch NVFP4 MoE decode and top-10 routing for V100 (SM70).

Marlin pads every routed row to a tensor-core tile, which leaves most of the
work empty at Qwen3.8 Flash Next's TP4 decode shape (at most four tokens, ten
experts each, a 160-wide expert slice). These kernels from the sglang-V100
fork stream the same Marlin-repacked NVFP4 weights with split-K FP16 dot
products instead. They read marlin_v100's SM70 repack layout and SGLang's SM70
NVFP4 scale encoding, and only this exact shape.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.kernels.jit.utils import KERNEL_PATH
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SRC_PATH = KERNEL_PATH / "csrc" / "moe" / "sm70_nvfp4_moe_decode.cu"
_EXT = None
_LOAD_ATTEMPTED = False

# The shape compiled into the kernels.
NUM_EXPERTS = 512
TOP_K = 10
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 160
# kMaxTokens in the source: several requests' MTP verify rows.
MAX_TOKENS = 64
# Marlin-repacked int32 weights, [experts, K/16, 2N] for 4-bit.
W13_SHAPE = (NUM_EXPERTS, HIDDEN_SIZE // 16, 2 * 2 * INTERMEDIATE_SIZE)
W2_SHAPE = (NUM_EXPERTS, INTERMEDIATE_SIZE // 16, 2 * HIDDEN_SIZE)
# Split counts of the gate/up and down partial kernels.
_GATE_UP_SPLITS = 40
_DOWN_SPLITS = 5


def max_tokens() -> int:
    """The most rows the kernels take in this process."""
    return min(MAX_TOKENS, envs.SGLANG_V100_NVFP4_MOE_DECODE_MAX_TOKENS.get())


def sm70_nvfp4_moe_decode_available() -> bool:
    """Whether the kernels are enabled and this is an SM70 GPU."""
    if not envs.SGLANG_V100_NVFP4_MOE_DECODE.get() or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() == (7, 0)
    except (AssertionError, RuntimeError):
        return False


def _load_ops():
    global _EXT, _LOAD_ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning("SM70 NVFP4 MoE decode source not found: %s", _SRC_PATH)
        return None
    from torch.utils.cpp_extension import load_inline

    build_directory = os.path.join(
        envs.SGLANG_V100_DECODE_CUDA_BUILD_DIR.get(), "nvfp4_moe"
    )
    os.makedirs(build_directory, exist_ok=True)
    try:
        _EXT = load_inline(
            name="sglang_sm70_nvfp4_moe_v100",
            cpp_sources="",
            cuda_sources=_SRC_PATH.read_text(),
            functions=None,
            is_python_module=True,
            verbose=False,
            build_directory=build_directory,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except Exception:  # pragma: no cover - build/environment failures
        logger.exception("SM70 NVFP4 MoE decode extension failed to build")
        return None
    logger.info("SM70 (V100): small-batch NVFP4 MoE decode kernel loaded.")
    return _EXT


def sm70_nvfp4_moe_ops_loaded() -> bool:
    """Build or load the extension; False when it cannot be built."""
    return _load_ops() is not None


def sm70_nvfp4_moe_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_global: torch.Tensor,
    w2_global: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Gated-SiLU NVFP4 experts, weighted and summed over the top-10 routes.

    hidden_states: FP16 [M, 2560], M <= MAX_TOKENS. w13/w2: Marlin int32 weights;
    scales: SM70-encoded E4M3 [E, K/16, N]; globals: FP32 [E]. topk_ids int32
    and topk_weights FP32, both [M, 10]; a negative id contributes nothing.
    """
    ext = _load_ops()
    if ext is None:
        raise RuntimeError("SM70 NVFP4 MoE decode extension is unavailable")
    num_routes = hidden_states.shape[0] * TOP_K
    device = hidden_states.device
    gate_up_partials = torch.empty(
        (_GATE_UP_SPLITS, num_routes, 2 * INTERMEDIATE_SIZE),
        dtype=torch.float32,
        device=device,
    )
    activated = torch.empty(
        (num_routes, INTERMEDIATE_SIZE), dtype=torch.float16, device=device
    )
    down_partials = torch.empty(
        (num_routes, _DOWN_SPLITS, HIDDEN_SIZE), dtype=torch.float32, device=device
    )
    output = torch.empty_like(hidden_states)
    ext.decode(
        hidden_states,
        w13,
        w2,
        w13_scales,
        w2_scales,
        w13_global,
        w2_global,
        topk_ids.reshape(-1),
        topk_weights.reshape(-1),
        gate_up_partials,
        activated,
        down_partials,
        output,
    )
    return output


def sm70_topk10_softmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over 512 router logits, top 10 renormalized, for M <= MAX_TOKENS rows.

    Returns (FP32 weights, int32 ids), both [M, 10]; ties pick the lower id.
    """
    ext = _load_ops()
    if ext is None:
        raise RuntimeError("SM70 NVFP4 MoE decode extension is unavailable")
    weights = torch.empty(
        (logits.shape[0], TOP_K), dtype=torch.float32, device=logits.device
    )
    ids = torch.empty((logits.shape[0], TOP_K), dtype=torch.int32, device=logits.device)
    ext.topk10_softmax(logits.contiguous(), weights, ids)
    return weights, ids

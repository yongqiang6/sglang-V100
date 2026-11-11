"""SM70 (V100) Marlin kernels from the external marlin_v100 library.

SGLang's JIT Marlin kernels compile to empty bodies below SM80, so on V100 every
Marlin GEMM and repack would write nothing. marlin_v100
(https://github.com/zhinianqin/marlin_v100, built and installed by
``scripts/v100/setup_v100_marlin.sh``) provides real SM70 kernels behind vLLM's
op schemas: ``torch.ops._C.{marlin_gemm,gptq_marlin_repack,awq_marlin_repack}``
and ``torch.ops._moe_C.moe_wna16_marlin_gemm``.

Its kernels read Marlin metadata differently from SM80+ Marlin: scales stay in
logical N-contiguous order (``[groups, N]``, ``[experts, groups, N]`` for MoE)
and zero points are float ``zero_point * scale`` values. The repack output keeps
the SM80+ shape, so callers that loop over experts work unchanged.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# scripts/v100/setup_v100_marlin.sh installs these names.
DENSE_LIBRARY = "_sm70_marlin_v100_dense.abi3.so"
MOE_LIBRARY = "_sm70_marlin_v100_moe.abi3.so"


def library_dirs() -> list[str]:
    dirs = [os.path.dirname(os.path.abspath(__file__))]
    extra = envs.SGLANG_V100_EXT_DIR.get()
    if extra:
        dirs.insert(0, extra)
    return dirs


def _load(library: str) -> bool:
    for directory in library_dirs():
        path = os.path.join(directory, library)
        if os.path.exists(path):
            torch.ops.load_library(path)
            return True
    return False


@lru_cache(maxsize=1)
def is_active() -> bool:
    """True on SM70 once both marlin_v100 libraries are loaded."""
    if not torch.cuda.is_available():
        return False
    cuda = current_arch().cuda
    if cuda is None or not cuda.is_sm(7, 0):
        return False
    if _load(DENSE_LIBRARY) and _load(MOE_LIBRARY):
        logger.info("SM70: Marlin kernels come from marlin_v100.")
        return True
    logger.warning(
        "SM70: marlin_v100 (%s, %s) not found in %s; GPTQ/AWQ/FP8 Marlin paths "
        "are disabled. Build it with scripts/v100/setup_v100_marlin.sh.",
        DENSE_LIBRARY,
        MOE_LIBRARY,
        library_dirs(),
    )
    return False


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_repack(
        b_q_weight, perm, size_k, size_n, num_bits, False
    )


def awq_marlin_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    return torch.ops._C.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits, False)


def _none_if_empty(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    # The kernels test presence, not size: an empty global_scale is read as an
    # NVFP4 input and rejected for integer weights.
    return None if t is None or t.numel() == 0 else t


def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    return torch.ops._C.marlin_gemm(
        a,
        c,
        b_q_weight,
        None,  # b_bias: SGLang adds the bias after the GEMM.
        b_scales,
        None,  # a_scales: weight-only quantization.
        _none_if_empty(global_scale),
        _none_if_empty(b_zeros),
        _none_if_empty(g_idx),
        _none_if_empty(perm),
        workspace,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        False,  # use_atomic_add: ignored by the SM70 kernels.
        use_fp32_reduce,
        is_zp_float,
    )


# marlin_v100 reads these overrides of its MoE kernel geometry at host launch;
# a captured CUDA graph keeps whatever was chosen at capture time.
_MOE_TUNING_ENV = (
    "SM70_MARLIN_MOE_CTA_GEOMETRY",
    "SM70_MARLIN_MOE_SPLIT_K",
    "SM70_MARLIN_MOE_METADATA_CACHE",
)
# Overrides the user set at startup apply to every MoE call; never replace them.
_USER_MOE_TUNING = any(os.environ.get(name) for name in _MOE_TUNING_ENV)
_active_moe_tuning: Optional[tuple[str, str, str]] = None


def qwen38_nvfp4_prefill_moe_tuning(
    b_scales: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Optional[tuple[str, str, str]]:
    """(CTA geometry, split-K, metadata) the sglang-V100 fork measured on V100
    for Qwen3.8 Flash Next's TP4 NVFP4 prefill MoE GEMMs, or None.

    marlin_v100's compiled-in selector already holds the fork's decode
    geometries; these two prefill shapes are not in it.
    """
    if b_scales.dtype != torch.float8_e4m3fn:
        return None
    # gate/up of roughly 1K-token chunks: [M, 2560] x [2560, 320], top-10 routes.
    if (
        moe_block_size == 32
        and top_k == 10
        and 512 <= size_m <= 2048
        and size_n == 320
        and size_k == 2560
    ):
        return ("32x64x64x4x32x32x32", "1", "vector_words")
    # down of chunks from 4K tokens: the routed rows [M * 10, 160] x [160, 2560].
    if (
        moe_block_size == 64
        and top_k == 1
        and size_m >= 40960
        and size_n == 2560
        and size_k == 160
    ):
        return ("64x256x32x4x64x64x32", "1", "vector_words")
    return None


def _apply_moe_tuning(tuning: Optional[tuple[str, str, str]]) -> None:
    """Point marlin_v100 at ``tuning``, or clear the overrides for None."""
    global _active_moe_tuning
    if _USER_MOE_TUNING or tuning == _active_moe_tuning:
        return
    if tuning is None:
        for name in _MOE_TUNING_ENV:
            os.environ.pop(name, None)
    else:
        for name, value in zip(_MOE_TUNING_ENV, tuning):
            os.environ[name] = value
    _active_moe_tuning = tuning


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_bias: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    """Blocks whose expert id is negative are skipped and their rows of ``c``
    are left untouched; only ``num_tokens_post_padded[0]`` is read."""
    _apply_moe_tuning(
        qwen38_nvfp4_prefill_moe_tuning(
            b_scales, moe_block_size, top_k, size_m, size_n, size_k
        )
    )
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        _none_if_empty(b_bias),
        b_scales,
        None,  # a_scales: weight-only quantization.
        _none_if_empty(global_scale),
        _none_if_empty(b_zeros),
        _none_if_empty(g_idx),
        _none_if_empty(perm),
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        False,  # use_atomic_add: ignored by the SM70 kernels.
        use_fp32_reduce,
        is_zp_float,
        -1,  # thread_k: -1 lets the library pick its measured SM70 geometry.
        -1,  # thread_n
        -1,  # blocks_per_sm
    )

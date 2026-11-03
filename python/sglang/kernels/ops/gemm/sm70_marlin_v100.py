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

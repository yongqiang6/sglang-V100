"""FP16 GEMV and two-to-seven-row GEMM for V100 (SM70) decode and MTP verify.

From the sglang-V100 fork, measured on V100 at Qwen3.8 Flash Next's TP4
shapes: one warp (or an 8/16-lane group) per output row with 16-byte loads and
FP32 accumulation beats cuBLAS's split-K kernels there. Shapes where cuBLAS
was faster are deliberately absent from the tables.
"""

from __future__ import annotations

import functools

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

# (output features, input features) -> (threads, lanes per row, vector width).
_GEMV_CONFIGS = {
    (4096, 2560): (64, 16, 8),
    (3584, 2560): (64, 32, 8),
    (2560, 1536): (128, 32, 8),
    (320, 2560): (256, 32, 8),
    (2560, 160): (128, 8, 4),
    (512, 2560): (256, 32, 8),
    (640, 2560): (128, 32, 8),
    (24, 2560): (64, 32, 8),
    (1, 2560): (256, 32, 8),
    (10240, 2560): (64, 32, 8),
    (2560, 2560): (128, 32, 8),
}
# A vocabulary shard at least this wide over the 2560 hidden size uses the
# default config (the LM head).
_LM_HEAD_MIN_ROWS = 32768
_HIDDEN_SIZE = 2560
_DEFAULT_GEMV_CONFIG = (64, 32, 8)

# (rows, output features, input features) -> (threads, lanes per output row).
_SMALL_GEMM_CONFIGS = {
    (2, 24, 2560): (64, 32),
    (2, 640, 2560): (256, 32),
    (2, 1, 2560): (64, 32),
    (2, 2560, 2560): (128, 32),
    (4, 24, 2560): (64, 32),
    (4, 640, 2560): (128, 32),
    (4, 2560, 2560): (64, 32),
    (2, 4096, 2560): (256, 32),
    (4, 4096, 2560): (64, 32),
    (2, 3584, 2560): (64, 32),
    (4, 3584, 2560): (64, 32),
    (2, 2560, 1536): (64, 32),
    (4, 2560, 1536): (128, 16),
    (2, 320, 2560): (256, 32),
    (4, 320, 2560): (256, 32),
    (2, 2560, 160): (128, 8),
    (4, 2560, 160): (128, 8),
    (2, 512, 2560): (256, 32),
    (4, 512, 2560): (256, 32),
    (2, 62080, 2560): (128, 32),
    (4, 62080, 2560): (128, 32),
}
# The most rows one launch of the row-templated kernels takes: an MTP verify
# step with seven draft tokens.
MAX_VERIFY_ROWS = 7
# Only two and four rows were measured; the others reuse the four-row launch,
# which loads each weight vector once whatever the row count.
_SMALL_GEMM_CONFIGS.update(
    {
        (rows, out_features, in_features): config
        for (measured_rows, out_features, in_features), config in tuple(
            _SMALL_GEMM_CONFIGS.items()
        )
        if measured_rows == 4
        for rows in (3, 5, 6, MAX_VERIFY_ROWS)
    }
)


def verify_row_chunks(rows: int, launch_rows: int = MAX_VERIFY_ROWS) -> tuple[int, ...]:
    """The rows of each launch covering ``rows`` with at most ``launch_rows``
    each, in as few and as even launches as that allows: a launch costs one
    pass over the weights whatever its rows. Every kernel computes its rows
    independently, so the chunking does not change any row's result."""
    launches = max(1, -(-rows // launch_rows))
    base, extra = divmod(rows, launches)
    return (base + 1,) * extra + (base,) * (launches - extra)


@functools.lru_cache(maxsize=None)
def _is_sm70_index(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 0)


def is_sm70(device: torch.device) -> bool:
    return device.type == "cuda" and _is_sm70_index(device.index)


def is_aligned_fp16(*tensors: torch.Tensor) -> bool:
    """Contiguous FP16 tensors on 16-byte boundaries, as the vector loads need."""
    return all(
        t.dtype == torch.float16 and t.is_contiguous() and t.data_ptr() % 16 == 0
        for t in tensors
    )


def supported(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """Whether ``x @ weight.T`` has a measured SM70 kernel for its shape."""
    if not (
        envs.SGLANG_SM70_DENSE_GEMV.get()
        and bias is None
        and x.is_cuda
        and x.ndim == 2
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
    ):
        return False
    rows = x.shape[0]
    if rows == 1:
        shape_ok = tuple(weight.shape) in _GEMV_CONFIGS or (
            weight.shape[1] == _HIDDEN_SIZE and weight.shape[0] >= _LM_HEAD_MIN_ROWS
        )
    else:
        shape_ok = (rows, *weight.shape) in _SMALL_GEMM_CONFIGS
    return (
        shape_ok
        and weight.device == x.device
        and is_aligned_fp16(x, weight)
        and is_sm70(x.device)
    )


@cache_once
def _gemv_module(threads: int, lanes: int, vector: int):
    return load_jit(
        "sm70_dense_gemv",
        str(threads),
        str(lanes),
        str(vector),
        cuda_files=["gemm/sm70_dense_gemv.cuh"],
        cuda_wrappers=[
            ("gemv", f"sglang::sm70_dense_gemv::gemv<{threads},{lanes},{vector}>")
        ],
    )


@cache_once
def _small_gemm_module(rows: int, threads: int, lanes: int):
    return load_jit(
        "sm70_small_gemm",
        str(rows),
        str(threads),
        str(lanes),
        cuda_files=["gemm/sm70_small_gemm.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_small_gemm::run<{rows},{threads},{lanes}>")
        ],
    )


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for a shape ``supported`` accepted; FP16 output."""
    rows = x.shape[0]
    out = torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=x.device)
    if rows == 1:
        config = _GEMV_CONFIGS.get(tuple(weight.shape), _DEFAULT_GEMV_CONFIG)
        _gemv_module(*config).gemv(x, weight, out)
    else:
        threads, lanes = _SMALL_GEMM_CONFIGS[(rows, *weight.shape)]
        _small_gemm_module(rows, threads, lanes).run(x, weight, out)
    return out

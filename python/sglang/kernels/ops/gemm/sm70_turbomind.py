"""SM70 (V100) TurboMind GEMMs: LMDeploy's s884 kernels carried by 1Cat-vLLM.

``scripts/v100/setup_v100_turbomind.sh`` builds them with SGLang's bindings
(``kernels/jit/csrc/sm70_turbomind``) into one library exposing
``torch.ops.sglang_sm70_turbomind``: FP16 MoE grouped GEMMs, block-FP8 W8A16 and
AWQ GEMMs. Without the library these layers keep their Triton/Marlin paths.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.gemm.sm70_marlin_v100 import library_dirs

logger = logging.getLogger(__name__)

# scripts/v100/build_sm70_turbomind.py installs this name.
LIBRARY = "_sm70_turbomind_v100.so"


@lru_cache(maxsize=1)
def ops() -> Optional[object]:
    """``torch.ops.sglang_sm70_turbomind`` on SM70 once the library is loaded."""
    if not torch.cuda.is_available():
        return None
    cuda = current_arch().cuda
    if cuda is None or not cuda.is_sm(7, 0):
        return None
    for directory in library_dirs():
        path = os.path.join(directory, LIBRARY)
        if os.path.exists(path):
            torch.ops.load_library(path)
            logger.info("SM70: TurboMind GEMMs loaded from %s", path)
            return torch.ops.sglang_sm70_turbomind
    logger.info(
        "SM70: %s not found in %s; FP16 MoE / FP8 / AWQ layers use Triton or "
        "Marlin. Build it with scripts/v100/setup_v100_turbomind.sh.",
        LIBRARY,
        library_dirs(),
    )
    return None

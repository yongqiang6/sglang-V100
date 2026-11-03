"""Build the TurboMind SM70 extension against the active torch.

Compiles LMDeploy's TurboMind SM70 GEMM kernels (from the 1Cat-vLLM source
subset fetched by setup_v100_turbomind.sh) together with SGLang's bindings in
python/sglang/kernels/jit/csrc/sm70_turbomind/, and copies the result to
--out-dir as _sm70_turbomind_v100.so.

Usage:
    python scripts/v100/build_sm70_turbomind.py --source-root DIR \\
        --cutlass-root DIR --out-dir DIR [--build-dir DIR]
"""

import argparse
import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parents[2]
SGLANG_SOURCES = REPO_ROOT / "python/sglang/kernels/jit/csrc/sm70_turbomind"
# Must match LIBRARY in sglang/kernels/ops/gemm/sm70_turbomind.py.
LIBRARY = "_sm70_turbomind_v100.so"
# The moe_permute ops in the 1Cat sources register under TORCH_EXTENSION_NAME.
EXTENSION_NAME = "sglang_sm70_turbomind"

LMDEPLOY_SOURCES = [
    "src/turbomind/core/check.cc",
    "src/turbomind/core/layout.cc",
    "src/turbomind/core/context.cc",
    "src/turbomind/core/allocator.cc",
    "src/turbomind/core/buffer.cc",
    "src/turbomind/core/stream.cc",
    "src/turbomind/utils/logger.cc",
    "src/turbomind/utils/parser.cc",
    "src/turbomind/kernels/gemm/gemm.cu",
    "src/turbomind/kernels/gemm/kernel.cu",
    "src/turbomind/kernels/gemm/dispatch_cache.cu",
    "src/turbomind/kernels/gemm/context.cu",
    "src/turbomind/kernels/gemm/convert_v3.cu",
    "src/turbomind/kernels/gemm/cast.cu",
    "src/turbomind/kernels/gemm/unpack.cu",
    "src/turbomind/kernels/gemm/tuner/cache_utils.cu",
    "src/turbomind/kernels/gemm/tuner/measurer.cu",
    "src/turbomind/kernels/gemm/tuner/sampler.cu",
    "src/turbomind/kernels/gemm/tuner/stopping_criterion.cc",
    "src/turbomind/kernels/gemm/tuner/params.cc",
    "src/turbomind/kernels/gemm/kernel/sm70_884_4.cu",
    "src/turbomind/kernels/gemm/kernel/sm70_884_8.cu",
    "src/turbomind/kernels/gemm/kernel/sm70_884_16.cu",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args()

    tm_root = args.source_root / "csrc" / "sm70_turbomind"
    lmdeploy = tm_root / "lmdeploy"
    if not lmdeploy.is_dir():
        raise SystemExit(f"TurboMind sources not found under {tm_root}.")
    if not (args.cutlass_root / "include" / "cutlass" / "cutlass.h").is_file():
        raise SystemExit(f"CUTLASS headers not found under {args.cutlass_root}.")

    sources = [
        SGLANG_SOURCES / "bindings.cpp",
        SGLANG_SOURCES / "fp16_moe_gemm.cu",
        *(lmdeploy / path for path in LMDEPLOY_SOURCES),
        tm_root / "ops" / "tm_registry_sm70.cu",
        tm_root / "ops" / "awq_sm70_gemm.cu",
        args.source_root / "csrc" / "moe" / "moe_permute_unpermute_op.cu",
        args.source_root
        / "csrc"
        / "moe"
        / "permute_unpermute_kernels"
        / "moe_permute_unpermute_kernel.cu",
    ]
    build_dir = args.build_dir
    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    library = load(
        name=EXTENSION_NAME,
        sources=[str(path) for path in sources],
        extra_include_paths=[
            str(lmdeploy),
            str(args.source_root / "csrc"),
            str(args.source_root / "csrc" / "moe"),
            str(args.cutlass_root / "include"),
            str(args.cutlass_root / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DENABLE_SM70_TURBOMIND=1",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        build_directory=None if build_dir is None else str(build_dir),
        is_python_module=False,
        verbose=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / LIBRARY
    shutil.copy2(library, destination)
    print(f"Installed {destination}")


if __name__ == "__main__":
    main()

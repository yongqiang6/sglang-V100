"""Validate a V100 (SM70) SGLang environment.

Static checks (no GPU needed): torch build, CUDA 12 runtime consistency, the
patched FlashInfer source and the sm_70 sglang-kernel build. GPU checks (skipped
with --no-gpu) load the kernels on the device and run a few small ops.

Usage:
    python scripts/v100/check_v100_env.py [--no-gpu]
"""

import argparse
import glob
import importlib.metadata as metadata
import importlib.util
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_results: list[tuple[bool, str, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((ok, name, detail))
    return ok


def _pinned_torch_version() -> str:
    project = tomllib.loads((REPO_ROOT / "python" / "pyproject.toml").read_text())
    for line in project["project"]["dependencies"]:
        if line.replace(" ", "").startswith("torch=="):
            return line.split("==", 1)[1].strip()
    raise RuntimeError("torch pin not found in python/pyproject.toml")


def _site_package_dirs() -> list[Path]:
    import site

    return [Path(p) for p in site.getsitepackages()]


def check_torch() -> None:
    import torch

    pinned = _pinned_torch_version()
    _check(
        "torch version matches python/pyproject.toml",
        torch.__version__.split("+")[0] == pinned,
        f"installed {torch.__version__}, pinned {pinned}",
    )
    cuda = torch.version.cuda or ""
    _check(
        "torch is a CUDA 12 build", cuda.startswith("12."), f"torch.version.cuda={cuda}"
    )
    arch_list = torch.cuda.get_arch_list()
    _check("torch ships sm_70 kernels", "sm_70" in arch_list, " ".join(arch_list))


def check_cuda_runtime_consistency() -> None:
    installed = {
        dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions()
    }
    version = installed.get("cuda-bindings", "missing")
    _check(
        "cuda-bindings is 12.x", version.startswith("12."), f"cuda-bindings {version}"
    )
    cudart13 = [
        str(p)
        for root in _site_package_dirs()
        for p in root.glob("nvidia/**/libcudart.so.13*")
    ]
    _check("no CUDA 13 runtime installed", not cudart13, ", ".join(cudart13))
    _check("nvidia-nccl-cu13 is not installed", "nvidia-nccl-cu13" not in installed)


def check_flashinfer() -> None:
    spec = importlib.util.find_spec("flashinfer")
    if not _check(
        "flashinfer is installed", spec is not None and spec.origin is not None
    ):
        return
    package_dir = Path(spec.origin).parent
    source = (package_dir / "jit" / "core.py").read_text()
    _check(
        "FlashInfer has the SM70 patch",
        "sm70 or higher" in source,
        f"flashinfer {metadata.version('flashinfer-python')} at {package_dir}",
    )


def check_tilelang() -> None:
    try:
        import tilelang
    except Exception as e:  # noqa: BLE001
        _check("tilelang imports (SM70 attention kernels)", False, repr(e))
        return
    _check("tilelang imports (SM70 attention kernels)", True, tilelang.__version__)


def _sgl_kernel_dir() -> Path | None:
    spec = importlib.util.find_spec("sgl_kernel")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def check_sgl_kernel_build() -> None:
    package_dir = _sgl_kernel_dir()
    if not _check("sgl_kernel is installed", package_dir is not None):
        return
    modules = glob.glob(str(package_dir / "sm70" / "common_ops*.so"))
    if not _check(
        "sgl_kernel has an sm70/common_ops build",
        len(modules) == 1,
        f"found {modules} under {package_dir}",
    ):
        return
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        _check("sgl_kernel sm70 cubins (cuobjdump not found, skipped)", True)
        return
    listing = subprocess.run(
        [cuobjdump, "--list-elf", modules[0]], capture_output=True, text=True
    ).stdout
    _check("sgl_kernel contains sm_70 cubins", ".sm_70." in listing)


def check_gpu() -> None:
    import torch

    if not _check("CUDA device available", torch.cuda.is_available()):
        return
    capability = torch.cuda.get_device_capability(0)
    _check("device is SM70", capability == (7, 0), f"capability={capability}")

    import sgl_kernel

    _check(
        "sgl_kernel loaded the sm70 module",
        "/sm70/" in Path(sgl_kernel.common_ops.__file__).as_posix(),
        sgl_kernel.common_ops.__file__,
    )

    torch.manual_seed(0)
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float16)
    out = sgl_kernel.rmsnorm(x, weight, 1e-6)
    ref = (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ) * weight.float()
    _check(
        "sgl_kernel.rmsnorm matches reference on SM70",
        torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2),
        f"max abs err {(out.float() - ref).abs().max().item():.3e}",
    )

    from flashinfer.sampling import top_k_top_p_sampling_from_probs

    probs = torch.softmax(torch.randn(2, 32000, device="cuda"), dim=-1)
    samples = top_k_top_p_sampling_from_probs(probs, top_k=20, top_p=0.9)
    torch.cuda.synchronize()
    _check("FlashInfer sampling JIT runs on SM70", samples.shape == (2,))

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    _check(
        "fused MoE uses the Triton moe_sum_reduce on SM70",
        not fused_moe._use_sgl_moe_sum_reduce,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-gpu", action="store_true", help="Skip device checks.")
    args = parser.parse_args()

    check_torch()
    check_cuda_runtime_consistency()
    check_flashinfer()
    check_tilelang()
    check_sgl_kernel_build()
    if not args.no_gpu:
        check_gpu()

    failed = 0
    for ok, name, detail in _results:
        failed += not ok
        suffix = f" ({detail})" if detail else ""
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

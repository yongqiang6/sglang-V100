"""Print the pip requirement list for a V100 (SM70, CUDA 12, torch cu126) install.

Starts from python/pyproject.toml so V100 tracks upstream pins, then removes
packages that are replaced by V100 source builds or only ship CUDA 13 / SM80+
binaries, and rewrites requirements that would pull CUDA 13 runtime wheels.

Usage:
    python scripts/v100/requirements.py [--extra diffusion] > requirements-v100.txt
    python scripts/v100/requirements.py --torch-index        # pins for the cu126 index
    python scripts/v100/requirements.py --torch-constraints  # installed torch family
"""

import argparse
import importlib.metadata as metadata
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "python" / "pyproject.toml"

# Installed from the torch CUDA index (download.pytorch.org/whl/cu126) together
# with torch. Their PyPI wheels are CUDA 13 builds: torch has no sm_70 kernels,
# and torchaudio/torchcodec link libcudart.so.13, which a cu126 install lacks.
TORCH_INDEX_PACKAGES = ("torch", "torchvision", "torchaudio", "torchcodec")

# Installed separately by install_v100.sh / v100.Dockerfile.
REPLACED_BY_V100_INSTALL = {
    *TORCH_INDEX_PACKAGES,
    "flashinfer-python",  # source checkout + patches/v100/flashinfer-0.7.2-sm70.patch
    "sglang-kernel",  # source build with -DSGL_KERNEL_V100_ONLY=ON
}

# CUDA 13-only or SM80+-only binaries. sglang imports all of them lazily.
UNSUPPORTED_ON_V100 = {
    "flash-attn-4",
    "humming-kernels",
    "nvshmem4py-cu13",
    "sgl-deep-ep",
    "sgl-deep-gemm",
    "sgl-flash-mla",
}

# A CUDA 13 floor installs libcudart.so.13 beside torch's libcudart.so.12.
OVERRIDES = {
    "cuda-python": "cuda-python>=12.6,<13",
}

# flashinfer is installed with --no-deps; these are its runtime requirements
# minus the CUDA 13-only EP transport (nccl4py, nccl-extensions) and
# nvidia-cudnn-frontend, which forces nvidia-cutlass-dsl[cu13] and is imported lazily.
FLASHINFER_RUNTIME_REQUIREMENTS = [
    "apache-tvm-ffi>=0.1.11,<0.2",
    "click",
    "cuda-tile>=1.4.0",
    "einops",
    "ninja",
    "numpy",
    "nvidia-cutlass-dsl>=4.6.2",
    "nvidia-ml-py",
    "packaging>=24.2",
    "requests",
    "tabulate",
    "tqdm",
]


def _rewrite(requirement: Requirement) -> Requirement | None:
    name = canonicalize_name(requirement.name)
    if name in REPLACED_BY_V100_INSTALL or name in UNSUPPORTED_ON_V100:
        return None
    if name in OVERRIDES:
        rewritten = Requirement(OVERRIDES[name])
        rewritten.marker = requirement.marker
        return rewritten
    requirement.extras.discard("cu13")
    return requirement


def build_requirements(extras: list[str]) -> list[str]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    raw = list(project["dependencies"])
    for extra in extras:
        raw.extend(project["optional-dependencies"][extra])
    raw.extend(FLASHINFER_RUNTIME_REQUIREMENTS)

    seen: set[str] = set()
    lines: list[str] = []
    for line in raw:
        requirement = _rewrite(Requirement(line))
        if requirement is None:
            continue
        key = f"{canonicalize_name(requirement.name)};{requirement.marker}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(str(requirement))
    return lines


def torch_index_requirements() -> list[str]:
    """pyproject pins of TORCH_INDEX_PACKAGES that apply to this platform."""
    pins = {}
    for line in tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]:
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name in TORCH_INDEX_PACKAGES and (
            requirement.marker is None or requirement.marker.evaluate()
        ):
            requirement.marker = None
            pins[name] = str(requirement)
    return [pins[name] for name in TORCH_INDEX_PACKAGES if name in pins]


def torch_index_constraints() -> list[str]:
    """Installed versions of TORCH_INDEX_PACKAGES, so other installs keep them."""
    lines = []
    for name in TORCH_INDEX_PACKAGES:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--torch-index",
        action="store_true",
        help="Print the torch-family pins to install from the torch CUDA index.",
    )
    mode.add_argument(
        "--torch-constraints",
        action="store_true",
        help="Print the installed torch-family versions as pip constraints.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Also include an optional-dependencies group (repeatable).",
    )
    args = parser.parse_args()
    if args.torch_index:
        lines = torch_index_requirements()
    elif args.torch_constraints:
        lines = torch_index_constraints()
    else:
        lines = build_requirements(args.extra)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

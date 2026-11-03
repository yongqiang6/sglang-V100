#!/usr/bin/env bash
# Host installer for SGLang on NVIDIA V100 (SM70): CUDA 12.8 toolkit, torch cu126,
# patched FlashInfer and an sm_70-only sglang-kernel build.
# Run as a child process; do not source it.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[install_v100] Do not source this file; run: bash %q\n' "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail
trap 'printf "\n[install_v100] FAILED (exit %d) at line %d: %s\n" "$?" "$LINENO" "$BASH_COMMAND" >&2' ERR

log() { printf '\n\033[1;34m[install_v100]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[install_v100] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
CONDA_ENV_NAME="${SGLANG_V100_CONDA_ENV:-sglang-v100}"
# The TileLang GDN kernels need nvcc >= 12.8 (tilelang 0.1.12 headers), and
# CUDA 13 dropped sm_70.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
# The PyPI torch wheel is CUDA 13 and has no sm_70 kernels; cu126 is the only
# torch 2.14 wheel that still ships them.
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
FLASHINFER_REPO="https://github.com/flashinfer-ai/flashinfer.git"
FLASHINFER_REV="ca0f620c9eca61759dc488e343e9e1e6ba9e43c3"  # v0.7.2, matches python/pyproject.toml
FLASHINFER_PATCH="$REPO_ROOT/patches/v100/flashinfer-0.7.2-sm70.patch"

[[ -f "$REPO_ROOT/python/pyproject.toml" ]] || die "$REPO_ROOT is not an SGLang checkout."

if [[ ${EUID} -eq 0 ]]; then SUDO=(); else SUDO=(sudo); fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  log "Installing CUDA 12.8 toolkit"
  # shellcheck disable=SC1091
  . /etc/os-release
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb
  "${SUDO[@]}" dpkg -i /tmp/cuda-keyring.deb
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y build-essential cmake git ninja-build pkg-config cuda-toolkit-12-8
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST=7.0

if ! command -v conda >/dev/null 2>&1; then
  log "Installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
# shellcheck disable=SC1091
. "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME" || \
  conda create -y -n "$CONDA_ENV_NAME" python=3.12 pip
conda activate "$CONDA_ENV_NAME"

if ! command -v cargo >/dev/null 2>&1; then
  log "Installing the Rust toolchain for SGLang's setuptools-rust extensions"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

MAX_JOBS="$(bash "$REPO_ROOT/scripts/v100/safe_jobs.sh")"
export MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" NVCC_THREADS="${NVCC_THREADS:-1}"
log "Build parallelism: MAX_JOBS=$MAX_JOBS"

python -m pip install --upgrade pip setuptools wheel setuptools-scm setuptools-rust \
  scikit-build-core "cmake>=3.31,<4" ninja packaging

TORCH_VERSION="$(python - "$REPO_ROOT/python/pyproject.toml" <<'PY'
import sys, tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
print(next(d.split("==")[1] for d in deps if d.replace(" ", "").startswith("torch==")))
PY
)"
log "Installing torch==$TORCH_VERSION (cu126)"
python -m pip install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX_URL"

log "Installing SGLang dependencies with V100 overrides"
mkdir -p "$DEPS_ROOT"
EXTRA_ARGS=()
# Space-separated optional-dependency groups, e.g. SGLANG_V100_EXTRAS=diffusion.
for extra in ${SGLANG_V100_EXTRAS:-}; do EXTRA_ARGS+=(--extra "$extra"); done
python "$REPO_ROOT/scripts/v100/requirements.py" "${EXTRA_ARGS[@]}" \
  >"$DEPS_ROOT/requirements-v100.txt"
python -c 'import torch; print(f"torch=={torch.__version__}")' >"$DEPS_ROOT/constraints-v100.txt"
python -m pip install -r "$DEPS_ROOT/requirements-v100.txt" -c "$DEPS_ROOT/constraints-v100.txt"
python -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT/python"

log "Installing FlashInfer v0.7.2 with the SM70 patch"
FLASHINFER_DIR="$DEPS_ROOT/flashinfer-sm70"
PATCH_STAMP="$(sha256sum "$FLASHINFER_PATCH" | awk '{print $1}')"
if [[ ! -d "$FLASHINFER_DIR/.git" ]]; then
  git clone "$FLASHINFER_REPO" "$FLASHINFER_DIR"
fi
if [[ "$(cat "$FLASHINFER_DIR/.sglang-v100-patch" 2>/dev/null)" != "$PATCH_STAMP-$FLASHINFER_REV" ]]; then
  if [[ -f "$FLASHINFER_DIR/.sglang-v100-patch" ]]; then
    # Only this script's patch is applied; restore it before re-patching.
    git -C "$FLASHINFER_DIR" checkout -- .
  fi
  git -C "$FLASHINFER_DIR" diff --quiet || die "$FLASHINFER_DIR has local changes; move it aside and rerun."
  git -C "$FLASHINFER_DIR" fetch origin "$FLASHINFER_REV"
  git -C "$FLASHINFER_DIR" checkout --detach "$FLASHINFER_REV"
  git -C "$FLASHINFER_DIR" submodule update --init --recursive
  git -C "$FLASHINFER_DIR" apply "$FLASHINFER_PATCH"
  printf '%s\n' "$PATCH_STAMP-$FLASHINFER_REV" >"$FLASHINFER_DIR/.sglang-v100-patch"
fi
python -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 \
  python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_DIR"

log "Building the sm_70-only sglang-kernel"
AOT_DIR="$REPO_ROOT/python/sglang/kernels/aot"
python -m pip uninstall -y sglang-kernel || true
# A stale in-tree build would otherwise be packaged next to the fresh module.
find "$AOT_DIR/python/sgl_kernel" -type f -name 'common_ops*.so' -delete
CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=$NVCC_THREADS" \
  python -m pip install --no-deps --no-build-isolation "$AOT_DIR"

log "Building marlin_v100 (SM70 GPTQ/AWQ/FP8 Marlin kernels)"
SGLANG_V100_DEPS_DIR="$DEPS_ROOT" bash "$REPO_ROOT/scripts/v100/setup_v100_marlin.sh"

log "Building the TurboMind SM70 extension (FP16 MoE, block-FP8 and AWQ GEMMs)"
SGLANG_V100_DEPS_DIR="$DEPS_ROOT" bash "$REPO_ROOT/scripts/v100/setup_v100_turbomind.sh"

log "Keeping only torch's CUDA 12 NCCL"
python -m pip uninstall -y nvidia-nccl-cu13 || true
NCCL_PIN="$(python - <<'PY'
from importlib.metadata import requires
print(next(r.split(";")[0].strip() for r in requires("torch") if r.startswith("nvidia-nccl-cu12")))
PY
)"
python -m pip install --force-reinstall --no-deps "$NCCL_PIN"

log "Validating the environment"
python "$REPO_ROOT/scripts/v100/check_v100_env.py" ${SGLANG_V100_CHECK_ARGS:-}

log "Complete. Run: conda activate $CONDA_ENV_NAME"

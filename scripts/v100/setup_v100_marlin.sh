#!/usr/bin/env bash
# Build marlin_v100 (SM70 Marlin dense + MoE kernels) against the active torch
# and install its two libraries next to sglang/kernels/ops/gemm/sm70_marlin_v100.py,
# which loads them on SM70. SGLang's own Marlin kernels are empty below SM80, so
# GPTQ/AWQ/FP8-Marlin serving on V100 needs this.
# Run as a child process; do not source it.
#
# Optional environment:
#   CUDA_HOME             CUDA 12.8/12.9 toolkit (default /usr/local/cuda-12.8)
#   SGLANG_V100_DEPS_DIR  source cache (default ~/.cache/sglang-v100-sources)
#   MARLIN_V100_REPO      marlin_v100 checkout (default $SGLANG_V100_DEPS_DIR/marlin-v100)
#   CUTLASS_DIR           CUTLASS checkout (default: the one bundled with tilelang)
#   SGLANG_V100_EXT_DIR   install here instead of the SGLang package; the runtime
#                         searches it too
#   CUDAHOSTCXX           host compiler for nvcc
#   MAX_JOBS, NVCC_THREADS build parallelism (default: scripts/v100/safe_jobs.sh, 1)

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[setup_v100_marlin] Do not source this file; run: bash %q\n' "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail
trap 'printf "\n[setup_v100_marlin] FAILED (exit %d) at line %d: %s\n" "$?" "$LINENO" "$BASH_COMMAND" >&2' ERR

log() { printf '\n\033[1;34m[setup_v100_marlin]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup_v100_marlin] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
MARLIN_REPO="${MARLIN_V100_REPO:-$DEPS_ROOT/marlin-v100}"
MARLIN_URL="https://github.com/zhinianqin/marlin_v100.git"
MARLIN_REF="6d72a49939701d26b15b617a4cd2423174adb2d1"
# C++20 for torch 2.14 headers, then the measured SM70 MoE CTA/split-K selections
# (Qwen3.5/3.6 AWQ/GPTQ, Qwen3.8 NVFP4).
MARLIN_PATCHES=(
  "$REPO_ROOT/patches/v100/marlin-v100-torch-cxx20.patch"
  "$REPO_ROOT/patches/v100/marlin-v100-qwen-sm70-tuning.patch"
  "$REPO_ROOT/patches/v100/marlin-v100-qwen38-nvfp4-tuning.patch"
)
# Must match DENSE_LIBRARY / MOE_LIBRARY in sglang/kernels/ops/gemm/sm70_marlin_v100.py.
DENSE_LIBRARY="_sm70_marlin_v100_dense.abi3.so"
MOE_LIBRARY="_sm70_marlin_v100_moe.abi3.so"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "no nvcc under CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | grep -Eq 'release 12\.[89]' \
  || die "marlin_v100 needs CUDA 12.8 or 12.9 (CUDA_HOME=$CUDA_HOME)."
"$PYTHON" -c 'import torch' || die "torch is not importable; activate the SGLang environment first."
log "torch $("$PYTHON" -c 'import torch; print(torch.__version__)'), nvcc $("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*')"

mkdir -p "$DEPS_ROOT"
if [[ ! -d "$MARLIN_REPO/.git" ]]; then
  git clone "$MARLIN_URL" "$MARLIN_REPO"
fi
PATCH_STAMP="$(cat "${MARLIN_PATCHES[@]}" | sha256sum | awk '{print $1}')-$MARLIN_REF"
if [[ "$(cat "$MARLIN_REPO/.sglang-v100-patch" 2>/dev/null)" != "$PATCH_STAMP" ]]; then
  if [[ -f "$MARLIN_REPO/.sglang-v100-patch" ]]; then
    # Only this script's patches are applied; restore them before re-patching.
    git -C "$MARLIN_REPO" checkout -- .
  fi
  git -C "$MARLIN_REPO" diff --quiet || die "$MARLIN_REPO has local changes; move it aside and rerun."
  git -C "$MARLIN_REPO" fetch origin "$MARLIN_REF"
  git -C "$MARLIN_REPO" checkout --detach "$MARLIN_REF"
  for patch in "${MARLIN_PATCHES[@]}"; do
    git -C "$MARLIN_REPO" apply "$patch"
  done
  printf '%s\n' "$PATCH_STAMP" >"$MARLIN_REPO/.sglang-v100-patch"
fi

if [[ -z "${CUTLASS_DIR:-}" ]]; then
  CUTLASS_DIR="$("$PYTHON" -c 'import os, tilelang; print(os.path.join(os.path.dirname(tilelang.__file__), "3rdparty", "cutlass"))')"
fi
[[ -f "$CUTLASS_DIR/include/cute/tensor.hpp" && -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]] \
  || die "CUTLASS_DIR=$CUTLASS_DIR has no include/{cute,cutlass}; set CUTLASS_DIR."
export CUTLASS_DIR

# marlin_v100's build.sh runs .venv/bin/python. A symlink to a venv interpreter
# loses the venv (pyvenv.cfg is looked up beside the link), so forward instead.
ACTIVE_PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
mkdir -p "$MARLIN_REPO/.venv/bin"
rm -f "$MARLIN_REPO/.venv/bin/python"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$ACTIVE_PYTHON" >"$MARLIN_REPO/.venv/bin/python"
chmod +x "$MARLIN_REPO/.venv/bin/python"

# CMake caches absolute torch paths; drop a build tree from another environment.
ACTIVE_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
CMAKE_CACHE="$(find "$MARLIN_REPO/build" -name CMakeCache.txt -print -quit 2>/dev/null || true)"
if [[ -n "$CMAKE_CACHE" ]] && ! grep -Fq "$ACTIVE_PREFIX" "$CMAKE_CACHE"; then
  log "Removing a CMake build tree from another Python environment"
  rm -rf "$(dirname "$CMAKE_CACHE")"
fi

export TORCH_CUDA_ARCH_LIST=7.0
# One marlin_v100 compiler process passed 7 GiB here (sm_70, nvcc 12.8).
export MAX_JOBS="${MAX_JOBS:-$(bash "$REPO_ROOT/scripts/v100/safe_jobs.sh" 10)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
# build.sh runs `cmake --build`, which ignores MAX_JOBS.
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
log "Building marlin_v100 (MAX_JOBS=$MAX_JOBS)"
(cd "$MARLIN_REPO" && bash ./build.sh)

DENSE_SO="$(ls "$MARLIN_REPO"/vllm/_C.*so 2>/dev/null | head -1 || true)"
MOE_SO="$(ls "$MARLIN_REPO"/vllm/_moe_C.*so 2>/dev/null | head -1 || true)"
[[ -f "$DENSE_SO" && -f "$MOE_SO" ]] || die "build did not produce vllm/_C and vllm/_moe_C."

if [[ -n "${SGLANG_V100_EXT_DIR:-}" ]]; then
  DEST="$SGLANG_V100_EXT_DIR"
  mkdir -p "$DEST"
else
  DEST="$("$PYTHON" -c 'import os, sglang.kernels.ops.gemm as g; print(os.path.dirname(g.__file__))')"
fi
DEST_DENSE="$DEST/$DENSE_LIBRARY"
DEST_MOE="$DEST/$MOE_LIBRARY"
cp -f "$DENSE_SO" "$DEST_DENSE"
cp -f "$MOE_SO" "$DEST_MOE"
log "Installed $DEST_DENSE and $DEST_MOE"

"$PYTHON" - "$DEST_DENSE" "$DEST_MOE" <<'PY'
import sys

import torch

for path in sys.argv[1:]:
    torch.ops.load_library(path)
for namespace, op in (
    ("_C", "marlin_gemm"),
    ("_C", "gptq_marlin_repack"),
    ("_C", "awq_marlin_repack"),
    ("_moe_C", "moe_wna16_marlin_gemm"),
):
    assert hasattr(getattr(torch.ops, namespace), op), f"torch.ops.{namespace}.{op} missing"
print("marlin_v100 ops registered")
PY

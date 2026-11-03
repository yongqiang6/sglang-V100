#!/usr/bin/env bash
# Build the TurboMind SM70 extension (LMDeploy GEMM kernels carried by 1Cat-vLLM)
# against the active torch and install _sm70_turbomind_v100.so next to
# sglang/kernels/ops/gemm/sm70_turbomind.py. It provides the V100 FP16 MoE,
# block-FP8 W8A16 and AWQ GEMMs; without it those layers use Triton/Marlin.
# Run as a child process; do not source it.
#
# Optional environment:
#   CUDA_HOME             CUDA 12.8/12.9 toolkit (default /usr/local/cuda-12.8)
#   SGLANG_V100_DEPS_DIR  source cache (default ~/.cache/sglang-v100-sources)
#   SGLANG_V100_EXT_DIR   install here instead of the SGLang package; the runtime
#                         searches it too
#   MAX_JOBS              build parallelism (default: scripts/v100/safe_jobs.sh)

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[setup_v100_turbomind] Do not source this file; run: bash %q\n' "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail
trap 'printf "\n[setup_v100_turbomind] FAILED (exit %d) at line %d: %s\n" "$?" "$LINENO" "$BASH_COMMAND" >&2' ERR

log() { printf '\n\033[1;34m[setup_v100_turbomind]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup_v100_turbomind] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
SOURCE_DIR="$DEPS_ROOT/turbomind-sm70-source"
SOURCE_URL="https://github.com/1CatAI/1Cat-vLLM.git"
SOURCE_REV="6ada86ed64af6d1a7b3cb0f34df237fd86f06d48"
CUTLASS_DIR="$DEPS_ROOT/cutlass-turbomind"
CUTLASS_URL="https://github.com/NVIDIA/cutlass.git"
CUTLASS_REV="da5e086dab31d63815acafdac9a9c5893b1c69e2"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "no nvcc under CUDA_HOME=$CUDA_HOME"
"$PYTHON" -c 'import torch' || die "torch is not importable; activate the SGLang environment first."
export PATH="$CUDA_HOME/bin:$PATH"

# Pinned sparse checkout: only the paths the build reads.
sparse_checkout() {
  local url=$1 rev=$2 destination=$3
  shift 3
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none --sparse --no-checkout "$url" "$destination"
  fi
  git -C "$destination" sparse-checkout set "$@"
  if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$rev" ]]; then
    git -C "$destination" diff --quiet || die "$destination has local changes; move it aside and rerun."
    git -C "$destination" fetch origin "$rev"
    git -C "$destination" checkout --detach "$rev"
  fi
}

mkdir -p "$DEPS_ROOT"
log "Fetching the 1Cat-vLLM TurboMind subset at $SOURCE_REV"
# Cone mode takes directories only and always keeps top-level files (LICENSE).
sparse_checkout "$SOURCE_URL" "$SOURCE_REV" "$SOURCE_DIR" \
  csrc/core csrc/sm70_turbomind csrc/moe
log "Fetching CUTLASS headers at $CUTLASS_REV"
sparse_checkout "$CUTLASS_URL" "$CUTLASS_REV" "$CUTLASS_DIR" include tools/util/include

if [[ -n "${SGLANG_V100_EXT_DIR:-}" ]]; then
  DEST="$SGLANG_V100_EXT_DIR"
else
  DEST="$("$PYTHON" -c 'import os, sglang.kernels.ops.gemm as g; print(os.path.dirname(g.__file__))')"
fi
export MAX_JOBS="${MAX_JOBS:-$(bash "$REPO_ROOT/scripts/v100/safe_jobs.sh")}"
log "Building the TurboMind SM70 extension (MAX_JOBS=$MAX_JOBS)"
"$PYTHON" "$REPO_ROOT/scripts/v100/build_sm70_turbomind.py" \
  --source-root "$SOURCE_DIR" --cutlass-root "$CUTLASS_DIR" \
  --build-dir "$DEPS_ROOT/turbomind-sm70-build" --out-dir "$DEST"

"$PYTHON" - "$DEST/_sm70_turbomind_v100.so" <<'PY'
import sys

import torch

torch.ops.load_library(sys.argv[1])
ops = torch.ops.sglang_sm70_turbomind
for op in (
    "f16_prepare",
    "f16_moe_build_ptrs",
    "f16_moe_gemm",
    "moe_permute_with_scratch",
    "moe_unpermute",
    "moe_permute_sort_workspace_size",
    "fp8_prepare",
    "fp8_gemm",
    "awq_prepare",
    "awq_gemm_out",
):
    assert hasattr(ops, op), f"torch.ops.sglang_sm70_turbomind.{op} missing"
print("TurboMind SM70 ops registered")
PY

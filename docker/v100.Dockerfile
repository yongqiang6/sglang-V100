# syntax=docker/dockerfile:1.7

# SGLang for NVIDIA V100 (Volta, SM70). Mirrors scripts/v100/install_v100.sh:
# CUDA 12.8 toolkit (the TileLang GDN kernels need >= 12.8; CUDA 13 dropped
# sm_70), torch cu126 (the PyPI torch wheel has no sm_70 kernels), patched
# FlashInfer and an sm_70-only sglang-kernel build. Expensive native builds are
# separate layers so later failures do not invalidate them.
#
#   DOCKER_BUILDKIT=1 docker build -f docker/v100.Dockerfile -t sglang-v100 .
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    SGLANG_V100_EXT_DIR=/opt/v100-ext \
    TORCH_CUDA_ARCH_LIST=7.0 \
    NVCC_THREADS=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git libnuma-dev ninja-build \
      pkg-config protobuf-compiler python3.12 python3.12-dev python3.12-venv \
    && python3.12 -m venv "$VIRTUAL_ENV"

FROM base AS builder

ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG FLASHINFER_REV=ca0f620c9eca61759dc488e343e9e1e6ba9e43c3

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal

WORKDIR /opt/sglang
COPY scripts/v100 /opt/sglang/scripts/v100
COPY python/pyproject.toml /opt/sglang/python/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip setuptools wheel setuptools-scm \
      setuptools-rust scikit-build-core "cmake>=3.31,<4" ninja packaging \
    && TORCH_VERSION="$(python -c 'import tomllib; deps = tomllib.load(open("/opt/sglang/python/pyproject.toml", "rb"))["project"]["dependencies"]; print(next(d.split("==")[1] for d in deps if d.replace(" ", "").startswith("torch==")))')" \
    && python -m pip install "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX_URL}" \
    && python /opt/sglang/scripts/v100/requirements.py >/tmp/requirements-v100.txt \
    && python -c 'import torch; print(f"torch=={torch.__version__}")' >/tmp/constraints-v100.txt \
    && python -m pip install -r /tmp/requirements-v100.txt -c /tmp/constraints-v100.txt

COPY patches/v100/flashinfer-0.7.2-sm70.patch /opt/sglang/patches/v100/
RUN git clone https://github.com/flashinfer-ai/flashinfer.git /opt/deps/flashinfer-sm70 \
    && git -C /opt/deps/flashinfer-sm70 checkout --detach "${FLASHINFER_REV}" \
    && git -C /opt/deps/flashinfer-sm70 submodule update --init --recursive \
    && git -C /opt/deps/flashinfer-sm70 apply /opt/sglang/patches/v100/flashinfer-0.7.2-sm70.patch \
    && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 \
       python -m pip install --no-deps --no-build-isolation -e /opt/deps/flashinfer-sm70

# marlin_v100: SM70 Marlin GEMM/repack kernels (SGLang's own are empty below SM80).
COPY patches/v100/marlin-v100-torch-cxx20.patch \
     patches/v100/marlin-v100-qwen-sm70-tuning.patch \
     patches/v100/marlin-v100-qwen38-nvfp4-tuning.patch \
     /opt/sglang/patches/v100/
RUN SGLANG_V100_DEPS_DIR=/opt/deps bash /opt/sglang/scripts/v100/setup_v100_marlin.sh \
    && rm -rf /opt/deps/marlin-v100/build

# TurboMind SM70 extension: FP16 MoE, block-FP8 W8A16 and AWQ GEMMs.
COPY python/sglang/kernels/jit/csrc/sm70_turbomind \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_turbomind
RUN SGLANG_V100_DEPS_DIR=/opt/deps bash /opt/sglang/scripts/v100/setup_v100_turbomind.sh \
    && rm -rf /opt/deps/turbomind-sm70-build

# Only the AOT kernel sources invalidate this layer.
COPY python/sglang/kernels/aot /opt/sglang/python/sglang/kernels/aot
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    find /opt/sglang/python/sglang/kernels/aot/python/sgl_kernel -type f -name 'common_ops*.so' -delete \
    && export MAX_JOBS="$(bash /opt/sglang/scripts/v100/safe_jobs.sh)" \
    && export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
    && CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=1" \
       python -m pip install --no-deps --no-build-isolation /opt/sglang/python/sglang/kernels/aot

# Application changes do not invalidate any native compilation layer.
COPY python /opt/sglang/python
COPY rust /opt/sglang/rust
COPY proto /opt/sglang/proto
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry \
    python -m pip install --no-deps --no-build-isolation -e /opt/sglang/python
RUN (python -m pip uninstall -y nvidia-nccl-cu13 || true) \
    && NCCL_PIN="$(python -c 'from importlib.metadata import requires; print(next(r.split(";")[0].strip() for r in requires("torch") if r.startswith("nvidia-nccl-cu12")))')" \
    && python -m pip install --force-reinstall --no-deps "${NCCL_PIN}" \
    && python /opt/sglang/scripts/v100/check_v100_env.py --no-gpu

FROM base AS runtime

ENV HF_HOME=/root/.cache/huggingface \
    FLASHINFER_WORKSPACE_BASE=/root/sglang-v100-jit \
    TRITON_CACHE_DIR=/root/sglang-v100-jit/triton \
    TORCHINDUCTOR_CACHE_DIR=/root/sglang-v100-jit/torchinductor

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/v100-ext /opt/v100-ext
COPY --from=builder /opt/deps/flashinfer-sm70 /opt/deps/flashinfer-sm70
COPY --from=builder /opt/sglang /opt/sglang
COPY docker/v100-entrypoint.sh /usr/local/bin/v100-entrypoint
RUN chmod +x /usr/local/bin/v100-entrypoint

WORKDIR /opt/sglang
ENTRYPOINT ["/usr/local/bin/v100-entrypoint"]
CMD ["--help"]

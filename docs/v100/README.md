# SGLang on NVIDIA V100 (SM70)

This branch re-ports the V100 work from the `sglang-V100` fork (forked from
upstream `cdd06011a`, 2026-06-01) onto current upstream SGLang, layer by layer.
Steps 1-4 (environment, base runtime, attention, GDN) are in place; the
remaining performance layers are not ported yet.

| Step | Scope | Status |
| --- | --- | --- |
| 1 | Environment: CUDA 12.8 toolkit, torch cu126, dependency overrides, patched FlashInfer, Docker/host installer, environment check | Done |
| 2 | Base runtime: SM70 floor, `SGL_KERNEL_V100_ONLY` AOT build, FlashInfer SM<80 paths, `moe_sum_reduce` fallback | Done |
| 3 | Attention: TileLang prefill/decode behind `triton`, `page_size=16` default, Triton SM70 decode tuning, FP8 KV on SM70 | Done |
| 4 | GDN: TileLang linear-attention backend, Triton chunk retune, FP16 Mamba state on SM70 | Done |
| 5 | Quantization: Marlin-V100 GPTQ/AWQ, TurboMind FP8 and FP16 MoE, QPN8 | Not started |
| 6 | Speculative decoding on upstream DFlash v2 / DSpark | Not started |
| 7 | Qwen3.8 Flash Next and MiniMax-H3 SM70 deltas | Not started |

## Why CUDA 12.8 with torch cu126

- Upstream pins `torch==2.14.1`. The PyPI wheel is a CUDA 13.0 build compiled
  for `sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120` only; CUDA 13 dropped Volta.
- PyTorch 2.14's `cu126` x86_64 wheel is built for
  `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`, so it is the torch build used here
  (`https://download.pytorch.org/whl/cu126`).
- The toolkit (`nvcc`, used by the sglang-kernel build, FlashInfer JIT and
  TileLang JIT) is CUDA 12.8. Upstream pins `tilelang==0.1.12`, whose
  `tl_templates/cuda/common.h` has
  `static_assert(false, "tl::st_bulk_shared requires CUDA >= 12.8")` for older
  toolkits. nvcc 12.6 trips it in the host pass of `nvcc --shared`, which is
  how TileLang's `cython` execution backend builds; the SM70 GDN kernels use
  that backend, so they need nvcc >= 12.8. (The attention kernels use the
  default `tvm_ffi` backend, which compiles device code only and works with
  12.6.) CUDA 12.8 still targets `sm_70`; CUDA 13 does not.
  `check_v100_env.py` requires nvcc 12.8 or 12.9.
- Torch keeps its CUDA 12.6 runtime wheels. A 12.8 toolkit against a 12.6
  torch is a minor-version mismatch, which torch's extension builder accepts;
  use a driver that supports CUDA 12.8 (R570 or newer, still Volta-capable).
  This mix has not been run on a V100 yet.
- torch 2.14.1 pins `triton~=3.8.0`; its default `ptxas` is CUDA 12.9 and
  assembles `sm_70` (the bundled `ptxas-blackwell`, CUDA 13.3, does not and is
  only used for Blackwell).

## Install

Host (conda env `sglang-v100`, installs CUDA 12.8 under `/usr/local/cuda-12.8` if missing):

```bash
bash scripts/v100/install_v100.sh
```

Docker:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/v100.Dockerfile -t sglang-v100 .
```

Validate an environment (add `--no-gpu` on a machine without a V100):

```bash
python scripts/v100/check_v100_env.py
```

## What the V100 environment changes

`scripts/v100/requirements.py` derives the dependency list from
`python/pyproject.toml`, so upstream pin bumps flow through automatically:

- Installed separately: `torch`/`torchvision` (cu126 index), `flashinfer-python`
  (v0.7.2 source + `patches/v100/flashinfer-0.7.2-sm70.patch`),
  `sglang-kernel` (source build with `-DSGL_KERNEL_V100_ONLY=ON`).
- Dropped (CUDA 13 or SM80+ only, imported lazily by SGLang): `flash-attn-4`,
  `humming-kernels`, `nvshmem4py-cu13`, `sgl-deep-ep`, `sgl-deep-gemm`, `sgl-flash-mla`.
- Rewritten: `cuda-python>=12.6,<13`, and every `[cu13]` extra is removed, so no
  `libcudart.so.13` lands next to torch's CUDA 12 runtime.
- `nvidia-nccl-cu13` is removed and torch's pinned `nvidia-nccl-cu12` reinstalled.

The FlashInfer patch (rebased from the fork's 0.6.12 patch onto v0.7.2):
accepts SM70 in `check_cuda_arch`, re-tunes the HMMA884 (SM70) FMHA-v2 kernel
specs for V100 shared memory, replaces `cuda::fast_mod_div` with plain division in
`fastdiv.cuh`, and gives `sampling.cuh` reduce functors for SM<75.
With v0.7.2's vendored CCCL, `sampling.cu`, `renorm.cu` and `fastdiv.cuh`
compile for `sm_70` (CUDA 12.6 and 12.8) with or without the last two hunks; only the
`check_cuda_arch` hunk is strictly required. The others are kept until a V100
run shows they can be dropped.

## What the runtime changes

- `model_runner_components/load_model_utils.py`: minimum compute capability is
  SM70 (was SM75); pre-SM80 GPUs still switch to float16.
- `kernels/aot/CMakeLists.txt`: `SGL_KERNEL_V100_ONLY=ON` builds one
  `sgl_kernel/sm70/common_ops` module for `sm_70` and skips FA3, FlashMLA,
  InfLLM-V2 and the SM80+ sources (`dsv4_norm_rope`, `es_*`,
  `fused_qknorm_rope_kernel`, `moe_sum_reduce`); their op registrations in
  `csrc/common_extension.cc` are guarded by the same macro.
- `sgl_kernel/load_utils.py` loads `sm70/` on SM70; `sgl_kernel/elementwise.py`
  keeps RMSNorm on the AOT kernels below SM80 (FlashInfer's CuTe-DSL norm uses `cp.async`).
- `layers/attention/flashinfer_backend.py`: below SM80 the wrappers get
  `backend="auto"` (FlashInfer still resolves it to FA2), which keeps SGLang's
  FA2-only fast prefill-plan paths off; decode tensor cores are off and
  `fast_decode_plan` is not installed during CUDA graph capture. These are the
  paths the fork saw crash on V100.
- `fused_moe.py`: below SM80 the Triton `moe_sum_reduce` replaces the AOT one.

Upstream already selects FlashInfer (or Triton) as the default MHA backend on
non-Hopper GPUs, so the fork's `server_args` default-backend change is not needed.

## Step 3: attention

The fork's standalone `tilelang_fa_v100` backend targeted the old CUDA-graph
API. Upstream gates many features on the backend *name* (`"triton"` appears in
hybrid-model, KV-dtype and model-override allowlists), so the port keeps the
name and changes what it builds on SM70:

- `layers/attention/tilelang_v100_backend.py`: `TileLangV100AttnBackend`
  subclasses `TritonAttnBackend`. Metadata, CUDA graphs, speculative verify, SWA
  and DCP stay on the inherited Triton code. It overrides only eager
  EXTEND/MIXED prefill (TileLang paged kernel; page table built from the
  physical slots `runner_ids.pack_read_stream` returns) and batch-1 grouped
  G6/D256 decode. Unsupported layers (SWA, logit cap, cross/encoder attention,
  MLA, other head dims) and any graph capture fall back to Triton.
- `attention_registry.create_triton_backend` builds it on SM70 when TileLang
  imports; `SGLANG_V100_DISABLE_TILELANG_ATTN=1` restores plain Triton.
- Resolution: SM70 defaults to `--attention-backend triton` and, with it,
  `--page-size 16`. `tilelang_fa_v100` / `flash_attn_v100` are deprecated
  aliases for `triton`.
- Kernels moved to `sglang/kernels/ops/attention/tilelang_fa_v100/` (TileLang)
  and `sglang/kernels/jit/csrc/attention/` (CUDA long-context decode, QSA
  combine). `T.gemm_v2` became `T.gemm` for tilelang 0.1.12.
- Triton decode/extend kernels take an `SM70_FP8` constexpr: FP8 KV (E4M3 or
  E5M2) is passed as bytes and decoded in-kernel, since Triton has no FP8
  operands on SM70. The grouped decode tile is 16 on SM70, and
  `TritonAttnBackend` raises `max_kv_splits` (up to 128, >=128 tokens per split)
  to fill 80 SMs.
- `runner/sm70_tilelang_warmup.py` compiles the prefill kernel for
  power-of-two batch sizes during runner warmup.
- The fork's `SGLANG_V100_*` / `SGLANG_SM70_QSA_COMBINE` knobs are registered in
  `environ.py` under "NVIDIA V100 (SM70)" with their old names. Boolean values
  now follow `EnvBool` (`true/1/yes/y`, `false/0/no/n`; `on/off` is no longer accepted).

Deferred: the fork's saturating E5M2 KV writer (needs the TurboMind extension,
step 5) and native linear speculative verify in the TileLang backend (step 6).

## Step 4: GDN (Qwen3.5 / Qwen3.6 linear attention)

Upstream reorganized GDN into per-mode kernels behind
`--linear-attn-{decode,prefill}-backend`, so the fork's hard-wired SM70 paths
become one more backend plus SM70 defaults:

- `layers/attention/linear/kernels/gdn_tilelang.py`: `TileLangGDNKernel`,
  selected with `--linear-attn-decode-backend tilelang` and/or
  `--linear-attn-prefill-backend tilelang` (SM70 only; other GPUs raise).
  Packed decode runs the persistent TileLang recurrence; decode calls carrying
  ReplaySSM ring caches, unpacked decode and target verify stay on Triton.
  Prefill runs the chunked TileLang kernels and, when the batch tracks Mamba
  prefix-cache states, returns the per-64-token checkpoints in the
  `[1, num_chunks, HV, V, K]` layout the tracker reads from Triton.
- Kernels live in `kernels/ops/attention/linear/gdn_tilelang_sm70.py` (packed
  decode) and `gdn_chunked_tilelang_sm70.py` (chunked prefill, K=V=128, FP16
  activations, FP16 or FP32 state pool). Their tuning knobs are
  `SGLANG_V100_GDN_*` in `environ.py`.
- Triton retune on SM70, kept for the default `triton` GDN backend:
  `chunk_delta_h.py` defaults to `BV=16`, 8 warps (the fork measured 12.6x over
  `BV=32`, 4 warps on Qwen3.6-27B TP4); `chunk_o.py` autotunes `BK`/`BV`/warps/
  stages on SM70 outside graph capture (the fixed H100 launch was 45% of V100
  prefill time). `SGLANG_GDN_CHUNK_H_*` still override.
- `configs/mamba_utils.py`: below SM80 the conv and SSM state default to FP16,
  since the model runs in FP16 and FP16 conv states cannot be written into a
  BF16 cache. `SGLANG_MAMBA_CONV_DTYPE` / `SGLANG_MAMBA_SSM_DTYPE` still win.

Not ported: the fork's `fused_sigmoid_gating` FP16 verify fix and SM70 `BV=8`
recurrent tweak (step 6, with speculative decoding) and the Qwen3.8 QSA/HC
fusions (step 7).

## Verification status

Done without V100 hardware:

- `-DSGL_KERNEL_V100_ONLY=ON` builds with CUDA 12.6 against torch 2.14.1:
  one `sgl_kernel/sm70/common_ops.abi3.so` whose 29 cubins are all `sm_70`, with
  the SM80+ ops absent and no `flash_ops` / `infllm_ops` / FlashMLA modules.
  `check_v100_env.py --no-gpu` passes on that build.
- The default (non-V100) configure still reaches and includes FlashMLA.
- Patched FlashInfer `sampling.cu` / `renorm.cu` compile for `sm_70`.
- The full serial build needs ~5.5 GiB per compiler process (see `safe_jobs.sh`).
- Step 3, compiled for `sm_70` without a GPU: the patched Triton decode
  (normal, grouped) and extend kernels with FP16, E5M2 and E4M3 KV and the
  split-count kernel (Triton 3.8, `GPUTarget("cuda", 70)`); all 13 TileLang
  kernel factories (paged prefill D64/128/256, sliding window, grouped decode,
  linear verify, FP8 gather, dense D256 variants) with tilelang 0.1.12 and
  nvcc 12.8 (and 12.6); and `sm70_longctx_decode.cu` with nvcc 12.6 and 12.8.
  Unpatched, the grouped decode kernel fails to compile for `sm_70` with E4M3
  or E5M2 KV.
- Step 4, compiled for `sm_70` without a GPU (nvcc 12.8, Qwen3.6-27B TP4
  shapes): the six TileLang GDN kernels (packed decode with FP16 and FP32
  state, KKT inverse, chunked forward with and without checkpoints, chunked
  state/output), the Triton `chunk_h` SM70 default and all 24 `chunk_o`
  autotune configs. With nvcc 12.6 the TileLang GDN kernels fail on the
  tilelang header described above.
- Resolution on a simulated SM70 platform is covered by
  `test_model_overrides.py::test_sm70_defaults_to_triton_with_16_token_pages`.

Still needs a V100: `check_v100_env.py` without `--no-gpu`, serving a model,
CUDA graph capture, the FlashInfer JIT attention paths, and numerical checks of
the TileLang and FP8 Triton attention and of the TileLang GDN kernels against
a reference.

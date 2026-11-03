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
| 5 | Quantization: Marlin-V100 GPTQ/AWQ (done); TurboMind FP8 and FP16 MoE, QPN8 | In progress |
| 6 | Speculative decoding on upstream DFlash v2 / DSpark | In progress |
| 7 | Qwen3.8 Flash Next and MiniMax-H3 SM70 deltas | In progress (NVFP4 done) |

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

Deferred: native linear speculative verify in the TileLang backend (step 6).

E5M2 KV writes: `Tensor.to(torch.float8_e5m2)` rounds FP16 magnitudes above
57344 to infinity. On SM70 `MHATokenToKVPool` writes E5M2 through
`kernels/ops/kvcache/e5m2_satfinite.py`, a Triton cast that matches torch's
round-to-nearest-even bit for bit where torch stays finite and saturates
elsewhere (the fork's TurboMind-built CUDA writer, reimplemented without the
extension; the scatter stays the existing store kernel).

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

## Step 5a: Marlin GPTQ / AWQ on SM70 (marlin_v100)

SGLang's JIT Marlin kernels (dense GEMM, MoE GEMM, GPTQ/AWQ repack) compile to
empty bodies below SM80: on V100 they launch and write nothing. The fork's fix
is kept: the external [marlin_v100](https://github.com/zhinianqin/marlin_v100)
library (vLLM op schemas, real SM70 kernels), pinned and patched by
`scripts/v100/setup_v100_marlin.sh` (called from `install_v100.sh` and the
Docker image; `SGLANG_V100_MARLIN_DIR` adds a search directory). The patches:
C++20 for torch 2.14's headers (marlin_v100 targets torch 2.10), and the fork's
measured MoE CTA/split-K selections. The fork's bf16-compat patch is dropped; it
is only needed below CUDA 12.8.

- `kernels/ops/gemm/sm70_marlin_v100.py` loads `_sm70_marlin_v100_{dense,moe}.abi3.so`
  on SM70; `gptq_marlin_gemm`, `moe_wna16_marlin_gemm`, `gptq_marlin_repack` and
  `awq_marlin_repack` dispatch to it when both are loaded. The fork only routed
  MoE and FP8 dense layers; dense GPTQ/AWQ layers would still have hit the stub.
- marlin_v100 reads metadata in its own layout: scales in logical `[groups, N]`
  order and zero points as fp16 `zero_point * scale`. `marlin_utils`
  (`marlin_permute_scales`, `marlin_zero_points`, `awq_to_marlin_zero_points`,
  `moe_awq_to_marlin_zero_points`) emit that layout on SM70, and `is_zp_float`
  follows the zero-point dtype.
- It skips MoE blocks with a negative expert id without writing their rows, so
  on SM70 a caller-provided output is zeroed when `zero_inactive_output` is set.
- `gptq_marlin` / `awq_marlin` report SM70 as supported when the library is
  loaded, so `gptq`/`awq` checkpoints are promoted as on SM80+, except act-order
  (`desc_act`) GPTQ, which marlin_v100 rejects. NVFP4 Marlin runs on SM70 (see
  step 7); MXFP4 Marlin raises there.
- `fused_marlin_moe` uses the Triton `moe_sum_reduce` below SM80.
- FP8 checkpoints: `Fp8Config` admits SM70 and FP8 linear layers auto-select
  weight-only FP8 Marlin (`can_auto_enable_marlin_fp8`) when marlin_v100 is
  loaded; the fork reached the same state through a generic loader bypass. FP8
  MoE layers are not supported on SM70 (no Triton FP8 there; the fork did not
  target them either).
- Kernel limits on SM70: FP16 activations/outputs only, `size_n % 64 == 0`,
  `size_k % 32 == 0`, groups -1/32/64/128 (FP8: -1/128).

Triton fused MoE: the fork's tuned `E=256,N=128,device_name=Tesla_V100-SXM2-32GB`
config (FP16 Qwen3.6 MoE) is kept under `triton_3_5_1/`, where it was measured;
upstream's lookup falls back to it on triton 3.8. The fork's `SGLANG_MOE_BLOCK_*`
sweep variables are not ported; tune with `benchmark/kernels/fused_moe_triton/`.

## Step 6: speculative decoding

Most of the fork's speculative commits backport upstream work (DFlash v2, relay
and WAR-barrier refactors) that current upstream already has. Ported deltas:

- GDN target verify with an FP16 state pool (SM70's default) now reloads each
  draft token's stored FP16 state before the next step, matching decode, which
  persists FP16 after every token. It used to carry the FP32 accumulator
  through the chain and could accept tokens decode would not produce. Upstream
  already rounds BF16 state when matching the CuTeDSL decode; that is unchanged.
- SM70 verify launch for Qwen3.6-27B TP4 MTP (`N=1, H=4, HV=12, K=V=128`,
  2 or 4 draft tokens): `BV=8` (192 CTAs instead of 48), bit-identical to BV=32.

Not ported, by design:

- The fork's lazy `TRTLLMMLABackend` import in `eagle_worker_v2`: it worked
  around a FlashInfer older than the MLA backend expected. The port builds the
  FlashInfer version upstream pins, so the module-level import is unchanged.
- The fork's DFlash sampling changes (penalties, min-p, seeded coins): they are
  model-agnostic and written against the fork's DFlash, which upstream's
  DFlash v2 has since replaced; they are not V100 adaptations.

## Step 7: Qwen3.8 Flash Next and MiniMax-H3

Upstream now ships Qwen3.8 Flash Next (`models/qwen4_exp.py`, MTP, PLE), so this
step is the fork's SM70 deltas on top of it. MiniMax-H3 is a diffusion model
(`multimodal_gen`).

- NVFP4 on SM70: upstream already falls back to weight-only FP4 Marlin below
  Blackwell (SM80-99). `fp4_utils.nvfp4_uses_marlin` extends that to SM70 when
  marlin_v100 is loaded (FP4 GEMM `auto` backend, the NVFP4 MoE Marlin fallback
  and `ModelOptFp4Config`'s capability floor). The rest of upstream's FP4 Marlin
  prep already produces marlin_v100's layout: logical scales on SM70 and the
  same S0E5M3 scale encoding as the fork's SM70 helper. The one difference is
  the global scale, which marlin_v100 reads as FP32 (one per expert for MoE);
  on SM70 it is computed from the FP32 checkpoint value. FP16 activations only
  (always the case on V100).

Not ported yet: the fork's Qwen3.8 NVFP4 decode kernel and per-shape Marlin
geometry selection (`sm70_nvfp4_moe_decode`), the SM70 Qwen fusions
(`sm70_qwen_fusions`, HC mix/gate, dense GEMV), PLE offload changes, the
PCIe P2P custom all-reduce, and MiniMax-H3 diffusion on V100.

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

- E5M2 KV cast: bit-exact with torch on every in-range FP16 value and
  saturating on the rest (all 65536 patterns, Triton interpreter); compiles for
  `sm_70`. `test/registered/kernels/ops/kvcache/test_e5m2_satfinite.py` runs it
  on GPU.
- Step 7: NVFP4 on SM70 is covered at the contract level (FP32 global scale,
  capability gates, MXFP4 rejection) in `test_sm70_marlin_v100.py`; no NVFP4
  checkpoint has been run.
- Step 6: in the Triton interpreter, FP16-state chain verify equals T
  sequential decodes bit for bit (23.6% of outputs differ without the reload),
  and the SM70 BV=8 verify launch equals BV=32
  (`test_fused_verify_triton_gdn.py`).
- Step 5a: the three marlin_v100 patches apply at the pinned revision. Without
  `marlin-v100-torch-cxx20.patch` its C++17 build fails on torch 2.14's headers;
  with it, `setup_v100_marlin.sh` compiles marlin_v100 for `sm_70` with nvcc 12.8
  (full build and op registration not yet confirmed here). The SM70 metadata
  and dispatch contracts are covered by
  `test/registered/unit/layers/quantization/test_sm70_marlin_v100.py`.

Still needs a V100: `check_v100_env.py` without `--no-gpu`, serving a model,
CUDA graph capture, the FlashInfer JIT attention paths, and numerical checks of
the TileLang and FP8 Triton attention and of the TileLang GDN kernels against
a reference.

# SGLang on NVIDIA V100 (SM70)

This branch re-ports the V100 work from the `sglang-V100` fork (forked from
upstream `cdd06011a`, 2026-06-01) onto current upstream SGLang, layer by layer.
All seven steps are ported. Each step lists what was left out on purpose and
what still needs a V100 to validate.

| Step | Scope | Status |
| --- | --- | --- |
| 1 | Environment: CUDA 12.8 toolkit, torch cu126, dependency overrides, patched FlashInfer, Docker/host installer, environment check | Done |
| 2 | Base runtime: SM70 floor, `SGL_KERNEL_V100_ONLY` AOT build, FlashInfer SM<80 paths, `moe_sum_reduce` fallback | Done |
| 3 | Attention: TileLang prefill/decode behind `triton`, `page_size=16` default, Triton SM70 decode tuning, FP8 KV on SM70 | Done |
| 4 | GDN: TileLang linear-attention backend, Triton chunk retune, FP16 Mamba state on SM70 | Done |
| 5 | Quantization: Marlin-V100 GPTQ/AWQ/FP8; TurboMind FP16 MoE, block-FP8 W8A16 and AWQ W4A16; QPN8 FP8 decode | Done |
| 6 | Speculative decoding on upstream DFlash v2 / DSpark | Done (TileLang native verify not ported) |
| 7 | Qwen3.8 Flash Next (QSA, PLE, NVFP4), PCIe custom all-reduce, MiniMax-H3 on V100 | Done (fork decode kernels, prefill Marlin geometry and greedy TP top-1 ported) |

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

Both build the two external V100 kernel libraries, marlin_v100
(`scripts/v100/setup_v100_marlin.sh`) and the TurboMind SM70 extension
(`scripts/v100/setup_v100_turbomind.sh`); either script can be rerun on its
own in an activated environment. They install into the SGLang package, or into
`SGLANG_V100_EXT_DIR` when it is set (the image uses `/opt/v100-ext`); the
runtime searches both.

Validate an environment (add `--no-gpu` on a machine without a V100):

```bash
python scripts/v100/check_v100_env.py
```

## What the V100 environment changes

`scripts/v100/requirements.py` derives the dependency list from
`python/pyproject.toml`, so upstream pin bumps flow through automatically:

- Installed separately: `torch`, `torchvision`, `torchaudio` and `torchcodec`
  from the cu126 index at their pyproject pins (`requirements.py --torch-index`),
  `flashinfer-python` (v0.7.2 source + `patches/v100/flashinfer-0.7.2-sm70.patch`)
  and `sglang-kernel` (source build with `-DSGL_KERNEL_V100_ONLY=ON`). The PyPI
  `torchaudio` 2.11.0 and `torchcodec` 0.17.0 wheels link `libcudart.so.13` (and
  `libnvrtc.so.13`), which a cu126 install does not have; `check_v100_env.py`
  fails on any torch extension package that links a CUDA 13 library.
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

Not ported: native linear speculative verify in the TileLang backend (see step 6).

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

The fork's FP16-state verify fix and SM70 `BV=8` verify launch are ported with
speculative decoding (step 6); its Qwen3.8 QSA/HC fusions are not (step 7).

## Step 5a: Marlin GPTQ / AWQ on SM70 (marlin_v100)

SGLang's JIT Marlin kernels (dense GEMM, MoE GEMM, GPTQ/AWQ repack) compile to
empty bodies below SM80: on V100 they launch and write nothing. The fork's fix
is kept: the external [marlin_v100](https://github.com/zhinianqin/marlin_v100)
library (vLLM op schemas, real SM70 kernels), pinned and patched by
`scripts/v100/setup_v100_marlin.sh` (called from `install_v100.sh` and the
Docker image; `SGLANG_V100_EXT_DIR` adds a search directory). The patches:
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

## Step 5b: TurboMind SM70 GEMMs and QPN8

The fork's fastest V100 paths use LMDeploy TurboMind GEMMs (built from
1Cat-vLLM's SM70 port) instead of Marlin. `setup_v100_turbomind.sh` builds them
as one torch library, `_sm70_turbomind_v100.so` (`torch.ops.sglang_sm70_turbomind`),
from pinned sparse checkouts of 1Cat-vLLM (`6ada86ed`) and CUTLASS v4.4.2
headers, plus the fork's two sources kept in
`kernels/jit/csrc/sm70_turbomind/`. `kernels/ops/gemm/sm70_turbomind.py` loads
it on SM70 only; without it every layer below keeps its Marlin or Triton path.

- FP16 MoE (`quantization/sm70_fp16_moe.py`): unquantized FP16 `FusedMoE`
  layers take TurboMind's grouped GEMM (gate/up interleaved at load, fused
  gated SiLU, permute/unpermute from the same library). It covers the contract
  the fork validated: gated SiLU, no bias, no clamps or `gemm1_alpha`, router
  weights applied after the experts, `routed_scaling_factor` 1 and no expert
  parallelism or A2A backend. Other layers stay on Triton.
  `SGLANG_SM70_FP16_MOE=0` disables it.
- Block-FP8 linear (`quantization/sm70_turbomind_fp8.py`): `[128, 128]`-block
  FP8 checkpoints run TurboMind W8A16 instead of FP8 Marlin
  (`SGLANG_SM70_FP8_BACKEND=auto|turbomind|marlin`; `turbomind` fails if it
  cannot be used). For the measured Qwen3.8-27B-FP8 TP4 shapes, decode
  (M <= 8) uses QPN8 and prefills of `SGLANG_SM70_FP8_PREFILL_MIN_TOKENS`
  (3920) tokens or more dequantize into a shared 85 MiB workspace and run a
  dense FP16 GEMM. `SGLANG_SM70_FP8_DECODE_QPN8`, `SGLANG_SM70_FP8_QPN8_FASTDEC`
  and `SGLANG_SM70_FP8_PREFILL_EXACT_DENSE` turn those off.
- QPN8 (`kernels/ops/gemm/sm70_qpn8.py`, `kernels/jit/csrc/gemm/sm70_fp8_qpn8_decode.cu`,
  adapted from [v100-skinny](https://github.com/dnv2003/v100-skinny), MIT):
  streams packed E4M3 codes and decodes them into `mma.sync.m8n8k4` operands.
  JIT-built on first use.
- AWQ (`quantization/sm70_turbomind_awq.py`): with the library loaded,
  `AWQConfig` admits SM70, and both the AWQ and AWQ-Marlin linear kernels
  repack group-32/64/128 weights once for TurboMind W4A16; selected
  Qwen3.6-27B-AWQ TP4 shapes run 4096-token prefill chunks as dequantize plus
  dense GEMM. `SGLANG_SM70_AWQ_TURBOMIND` / `SGLANG_SM70_AWQ_PREFILL_EXACT_DENSE`
  turn them off. AWQ MoE layers stay on marlin_v100.

Not ported: the fork's fused gated-SiLU epilogues for FP8 `gate_up_proj`
(QPN8 and TurboMind W8A16), which make the linear return `silu(gate) * up` and
need the model's MLP to skip its own activation; and its TurboMind E5M2 KV
writer, which step 3 replaced with a Triton cast.

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
- Native linear verify and the long-context DFlash paths in the TileLang
  attention backend. Upstream runs target verify inside CUDA graphs, where the
  ported backend uses the inherited Triton path; the fork's TileLang verify
  kernel still compiles for `sm_70` (step 3), but switching verify to it needs
  a V100 to check it against Triton first.

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

- NVFP4 MoE memory on SM70: a Marlin expert takes exactly the bytes of its
  checkpoint layout, so SM70 repacks the expert weights and block scales back
  into the loaded storage instead of building new tensors. With upstream's
  out-of-place repack, Qwen3.8 Flash Next ran out of memory on 32 GiB V100s at
  TP4 during this step: allocated memory rose from 21.8 GiB after loading to
  31.1 GiB. The per-layer allocation sizes in that log show the old expert
  tensors were not being released. Repacking in place leaves no second copy to
  hold. SM70 also skips the swizzled block-scale placeholders, which only
  FlashInfer reads (2.1 GiB per GPU for that model at TP4); the fork freed the
  same placeholders after loading.

- PCIe custom all-reduce: upstream disables custom all-reduce above two GPUs
  without full NVLink. `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` keeps it (V100 PCIe
  cards, or SXM2 boards whose GPUs are not all NVLink peers) and restricts
  `CustomAllReduceV2` to one-shot push for messages up to
  `SGLANG_CUSTOM_AR_PCIE_MAX_BYTES` (128 KiB, decode-sized); pull and two-shot
  modes are off and larger messages use NCCL. Off by default, as in the fork.
- MiniMax-H3 (`multimodal_gen`):
  - On SM70, `validate_server_args` runs every BF16 component (DiT, VAEs, text
    encoders) in FP16 and keeps FP32 ones; an explicit BF16
    `--component-precisions` entry is rejected.
  - `--attention-backend tilelang_fa_v100` selects a diffusion backend whose
    packed varlen attention for FP16, non-causal, head-dim-128 inputs (H3's
    contract) runs the fork's TileLang kernel
    (`kernels/ops/attention/tilelang_dense_varlen_sm70.py`); every other call
    is SDPA. It resolves to SDPA on other GPUs.
  - `--quantization v100_w4a16_awq` quantizes the rank-local DiT linear
    weights online (round-to-nearest, asymmetric UINT4, group 128, AWQ
    layout) and runs them on TurboMind's W4A16 GEMM; `--quantization
    v100_w8a16` stores per-channel INT8 and dequantizes for cuBLAS. Both load
    the DiT in FP16 (`QuantizationConfig.requires_fp16_source`), honor
    `--quantization-ignored-layers` and saturate FP16 overflow. The deployment
    command is in the MiniMax-H3 cookbook page.

### Qwen3.8 Flash Next on SM70

Its full-attention layers use QSA sparse attention, which upstream serves with
FlashAttention (FA2 is SM80+, FA4 is Hopper/Blackwell) and FP8 indexer
GEMMs, none of which run on Volta. The fork's SM70 routes are ported onto
upstream's QSA code; the routing lives in
`layers/attention/qsa/sm70.py`:

- Prefix-free prefills up to `SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS`
  (8192) tokens run dense causal attention with the TileLang D256 kernel; the
  indexer still compresses keys but skips the selection. The fork measured the
  sparse Triton gather at 29.5 ms per layer against 3.55 ms dense (4K tokens,
  TP4). Longer prefills stay sparse (Triton).
- When `--context-length` fits that threshold, decode reads the paged E5M2
  cache densely (TileLang grouped decode).
- Otherwise prefill with a cached prefix (one request), decode and MTP verify,
  at the exact TP4 shape (6 query heads, 1 KV head, head dim 256) with an E5M2
  cache, read the selected keys from the cache with the fork's CUDA kernels.
- Any other decode runs upstream's Triton packed-decode kernel instead of
  FlashAttention, as on ROCm.
- The indexer runs FP16 (Volta's MMA is FP16-only): `--qsa-indexer-dtype auto`
  resolves to float16 and the other values raise. Decode pads index heads to
  16, except at the TP4 shape (4 heads, dim 128), where the fork's CUDA scorer
  reads them unpadded.
- Pages are 16 tokens (64 elsewhere), so a ratio-4 compressed page is 4 rows,
  which the TileLang scorer now accepts on SM70.
- PLE: the N-gram table is offloaded to host memory by default even at FP16
  (it does not fit a 32 GB V100 next to the weights), and the host gather
  decodes its FP8 bytes as Triton's software `fp8e4b15` times 256, which is
  exact for every finite E4M3 code. HC and PLE tensors use FP16.

The deployment command is in the Qwen3.8 Flash Next cookbook page. Upstream
already initializes the offloaded table on the meta device, so the fork's
transient-VRAM fix (`qwen4_exp_v100.py`) is not needed.

Ported from the fork's decode-speed work: the NVFP4 MoE decode kernel and its
top-10 router (`kernels/ops/moe/sm70_nvfp4_moe_decode.py`; the fork's CUDA
source, only clang-formatted). At the TP4 decode shape (at most 4 tokens, 512 experts, top-10,
hidden 2560, expert slice 160) the Marlin runner and `fused_topk` use them in
place of Marlin and the Triton router; anything else, or a failed build, stays
on those. They read the Marlin-repacked weights and SM70 scale bytes that
upstream's prep produces, which match the fork's for FP16 (same marlin_v100
pin and repack, same scale encoding, a scale factor of 1). The extension builds
at first use under `SGLANG_V100_DECODE_CUDA_BUILD_DIR/nvfp4_moe`;
`SGLANG_V100_NVFP4_MOE_DECODE=0` turns it off.

Also ported, with the fork's CUDA sources (clang-formatted only) behind new
hooks in upstream's restructured code:

- Dense GEMV and few-row GEMM (`kernels/ops/gemm/sm70_dense_gemv.py`): FP16
  projections at one row, and at two to seven rows for MTP verify, for the
  fork's measured TP4 shapes (attention and GDN projections, shared expert,
  router, LM head shard). `UnquantizedLinearMethod` and the LM head take them
  for bias-free, aligned inputs; other shapes stay on cuBLAS.
  `SGLANG_SM70_DENSE_GEMV=0` turns them off.
- Hyper-connection mix (`sm70_hc_mix.py`): down projection with SiLU, then an
  up projection fusing sigmoid, multiply and the stream mean, in place of the
  persistent Triton mix, which is pathologically slow on SM70. From two rows
  up the down projection also computes the combine's inject-gate partials.
- Qwen projections (`sm70_qwen_fusions.py`): the shared expert's gate_up with
  SiLU-and-mul, the GDN qkvz + ba projections in one launch at decode, and at
  MTP verify both projections written straight into the split GDN inputs.
- The PLE gate/value kernel's FP16 branch, which reproduces eager FP16
  rounding, on SM70 only (BF16 code generation is unchanged).

`SGLANG_SM70_QWEN_FUSIONS=0` turns off the last three. Left out on purpose:
the fork's shared-expert gate kernel (upstream fuses that gate already) and
its GDN QKV-prefix reuse (upstream's fused ratio-three split replaces it).
Two smaller pieces from the fork:

- Prefill MoE geometry (`sm70_marlin_v100.py`): two NVFP4 prefill shapes, the
  512 to 2048 token gate/up GEMM and the down GEMM of a 4096-token chunk, launch
  with the fork's tuned CTA geometry through marlin_v100's
  `SM70_MARLIN_MOE_*` overrides, cleared again for every other shape. Decode
  shapes already use the geometry compiled into marlin_v100. Setting any of
  those variables yourself turns this off.
- Greedy TP top-1 (`layers/greedy_tp_top1.py`, opt-in with
  `SGLANG_V100_GREEDY_TP_TOP1=1`): when every request is greedy and nothing
  reads the logits (no logprobs, penalties, grammar, logit bias or custom
  processor), each TP rank exchanges its best (logit, token id) pair instead
  of all-gathering the vocabulary logits. The result is the exact argmax, with
  the lowest id on ties. It runs on eager steps only, never inside a CUDA
  graph, and not under speculative decoding, so an MTP deployment with CUDA
  graphs never takes it.

MTP chains of two to seven draft tokens stay on these paths: the few-row GEMM,
the hyper-connection mix and inject gates, the GDN projections and verify
launch, the NVFP4 MoE kernel with its router, and the QSA split combine all
take up to seven verify rows (`--speculative-num-steps` 1 to 6 with
`--speculative-num-draft-tokens` one higher). The fork ran and measured two
and four rows only; three and five to seven rows reuse the four-row launch
configurations and are untested on a V100. Whether a longer chain is faster
depends on the acceptance rate.

Concurrent requests multiply the verify rows (requests x draft tokens), past
what one launch of those kernels takes. Three of them cover such batches, none
measured on a V100 yet, each with a switch for A/B runs:

- The NVFP4 MoE kernel and its router take up to 32 rows; they are indexed by
  route, so they scale with the rows as Marlin does.
  `SGLANG_V100_NVFP4_MOE_DECODE_MAX_TOKENS=7` restores Marlin above seven rows.
- The hyper-connection mix and inject gates run rows above seven as launches
  of three or four rows, in place of the persistent Triton mix. Each launch
  re-reads the mix weights. `SGLANG_SM70_HC_MIX_MAX_ROWS=7` turns this off.
- The GDN verify projections can run the same way, but stay off by default:
  cuBLAS reads those weights once for the whole batch.
  `SGLANG_SM70_GDN_PROJ_MAX_ROWS=32` tries it.

The native QSA split combine also takes up to 32 rows. The few-row GEMM stays
at seven rows; larger batches use cuBLAS. Chunked launches compute each row
exactly as a single request's launch does.

On a 4x V100 PCIe server, upstream without any of the fork's
decode work ran Qwen3.8 Flash Next at about 38 tok/s with MTP (accept length
3.05), against about 70 on the fork's image. With the ported kernels the same
server was reported at about 65 tok/s single-stream (MTP 3 steps / 4 draft
tokens, `NCCL_P2P_LEVEL=NVL`, so NCCL all-reduces went through host memory).

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
  capability gates, MXFP4 rejection, the MoE repack reusing the loaded storage
  and matching the out-of-place result) in `test_sm70_marlin_v100.py`. Run on
  CPU at Qwen3.8 Flash Next TP4 shapes (512 experts) with stubbed kernels, the
  repack holds memory flat across layers. No NVFP4 checkpoint has completed
  loading on a V100 with this change yet.
- NVFP4 MoE decode kernel: all seven kernels compile for `sm_70` with
  PyTorch's extension flags (nvcc 12.8, no spills). `test_sm70_marlin_v100.py`
  pins the scale bytes against the kernel's reader for every E4M3 code, and
  the gates that keep every other shape and MoE variant on Marlin. It has not
  run on a V100 in this tree.
- Decode GEMV and Qwen fusions: all 23 kernel instantiations build for `sm_70`
  through the JIT loader, and the FP16 PLE branch compiles for `sm_70` (its
  BF16 PTX is identical to before). `test_sm70_decode_fusions.py` pins each
  gate's refusals and the routing of SM70 inject-gate partials. Not run on a
  V100 in this tree.
- Step 6: in the Triton interpreter, FP16-state chain verify equals T
  sequential decodes bit for bit (23.6% of outputs differ without the reload),
  and the SM70 BV=8 verify launch equals BV=32
  (`test_fused_verify_triton_gdn.py`).
- Step 5a: the three marlin_v100 patches apply at the pinned revision. Without
  `marlin-v100-torch-cxx20.patch` its C++17 build fails on torch 2.14's headers;
  with it, `setup_v100_marlin.sh` builds marlin_v100 for `sm_70` with nvcc 12.8.
  Here 5 of its 25 objects were compiled, including the dense u4 and u4b8 GEMMs.
  Each GEMM unit takes 45-50 minutes serially and peaks at 7.3 GiB in `cicc`,
  so the script budgets 10 GiB per job. The link and op registration were not
  reached. The SM70 metadata
  and dispatch contracts are covered by
  `test/registered/unit/layers/quantization/test_sm70_marlin_v100.py`.

- Step 5b: the fork's `fp16_moe_gemm.cu`, the ported `bindings.cpp` and the
  QPN8 kernel compile for `sm_70` with nvcc 12.8 against torch 2.14.1's
  headers. Layer selection (which FP16 MoE, FP8 and AWQ layers take TurboMind,
  and that a forced `turbomind` backend fails without it) is covered by
  `test/registered/unit/layers/quantization/test_sm70_turbomind.py`.
  From empty source directories, `setup_v100_turbomind.sh` fetches both
  pinned sparse checkouts and builds the library with nvcc 12.8. That took 14
  minutes at `MAX_JOBS=1` and under 3 GiB per compiler process, producing only
  `sm_70` cubins. Every op the Python layers call is registered, and
  `check_v100_env.py --no-gpu` finds the library through `SGLANG_V100_EXT_DIR`.
- Step 7, Qwen3.8 Flash Next: compiled for `sm_70` in FP16 at the TP4 shapes:
  the QSA TileLang scorers (prefill, relative prefill, mask, decode with 4- and
  16-row compressed pages), the QSA sparse-attention, gather, block-expansion
  and PLE Triton kernels (all within 96 KiB of shared memory, including the
  E5M2-to-FP16 gathers), the HC, PLE and gate Triton kernels, and the JIT CUDA
  grouped Gemma RMSNorm, HC combine, QSA indexer and fast top-k. The FP8 PLE
  decode is checked exactly against E4M3 (`test_qwen4_ple_offload.py`); the
  SM70 routing, indexer dtype and head padding by
  `test/registered/unit/layers/attention/test_qsa_sm70_routing.py`; the
  16-token pages and FP16 offload default by `test_model_overrides.py`.
- Step 7: the PCIe all-reduce mode is covered by
  `test_custom_all_reduce_v2_capability.py::test_pcie_config_selects_only_one_shot_push`.
  For MiniMax-H3, the TileLang dense varlen kernel compiles for `sm_70`;
  backend selection, FP16 admission and the W4A16/W8A16 contracts (AWQ nibble
  order, error within half a quantization step, FP16 DiT loading) are covered
  by `test_cuda_attention_backend.py`, `test_minimax_h3_admission.py` and
  `test_v100_quantization.py` under `multimodal_gen/test/unit/`.

Still needs a V100: `check_v100_env.py` without `--no-gpu`, serving a model,
CUDA graph capture, the FlashInfer JIT attention paths, numerical checks of
the TileLang and FP8 Triton attention, the TileLang GDN kernels and the
TurboMind and QPN8 GEMMs against a reference, and an H3 generation.

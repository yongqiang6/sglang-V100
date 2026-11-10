// SPDX-License-Identifier: Apache-2.0
// Batch-one Qwen3.8 shared-expert gate_up projection fused with SiLU-and-mul
// on Volta.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
namespace sglang::sm70_qwen_fusions {
__global__ void gate_up_kernel(const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ out) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * 4 + warp / 2;
  const int row = n + (warp % 2) * 160;
  float a[8] = {};
#pragma unroll
  for (int j = 0; j < 10; ++j) {
    const int k = lane * 8 + j * 256;
    const int4 xv = *reinterpret_cast<const int4*>(x + k), wv = *reinterpret_cast<const int4*>(w + row * 2560 + k);
    const half* hx = reinterpret_cast<const half*>(&xv);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      a[i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), a[i]);
  }
  float sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    sum += a[i];
#pragma unroll
  for (int d = 16; d > 0; d /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, d);
  __shared__ float vals[8];
  if (lane == 0) vals[warp] = __half2float(__float2half_rn(sum));
  __syncthreads();
  if (lane == 0 && warp % 2 == 0) {
    float g = vals[warp], u = vals[warp + 1];
    out[n] = __float2half_rn(g / (1.f + __expf(-g)) * u);
  }
}
void gate_up(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, tvm::ffi::TensorView out) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({1, 2560}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({320, 2560}).with_dtype<half>().with_device(dev).verify(weight);
  TensorMatcher({1, 160}).with_dtype<half>().with_device(dev).verify(out);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
      "Operands must be 16-byte aligned");
  LaunchKernel(40, 256, dev.unwrap())(
      gate_up_kernel,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(weight.data_ptr()),
      static_cast<half*>(out.data_ptr()));
}
}  // namespace sglang::sm70_qwen_fusions

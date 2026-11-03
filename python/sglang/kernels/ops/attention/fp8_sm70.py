"""Software FP8 helpers for Volta attention kernels.

SM70 can store FP8 bytes, but PTX/Triton cannot use native FP8 operand types on
this architecture. V100 attention kernels therefore receive the cache as raw
uint8 bytes, decode to FP32, and use FP16 tensor-core MMA.
"""

import triton
import triton.language as tl


@triton.jit
def fp8_e4m3fn_to_fp32(raw):
    """Decode raw IEEE-like E4M3FN bytes, including subnormals.

    E4M3FN uses exponent bias 7. Exponent 15 remains finite for mantissas 0-6
    (up to 448); the all-ones mantissa is NaN and is never emitted by SGLang's
    finite KV-cache cast.
    """
    raw = raw.to(tl.int32)
    sign = tl.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    subnormal = mantissa.to(tl.float32) * (2.0**-9)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    return sign * tl.where(exponent == 0, subnormal, normal)


@triton.jit
def fp8_e5m2_to_fp32(raw):
    """Decode raw IEEE E5M2 bytes, including subnormals.

    SGLang's SM70 cache writer saturates finite FP16 inputs, so attention does
    not encounter the exponent-31 Inf/NaN encodings in normal operation.
    """
    raw = raw.to(tl.int32)
    sign = tl.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 2) & 0x1F
    mantissa = raw & 0x03
    subnormal = mantissa.to(tl.float32) * (2.0**-16)
    normal = (1.0 + mantissa.to(tl.float32) * 0.25) * tl.exp2(
        exponent.to(tl.float32) - 15.0
    )
    return sign * tl.where(exponent == 0, subnormal, normal)


@triton.jit
def fp8_sm70_to_fp32(raw, is_e5m2: tl.constexpr):
    if is_e5m2:
        return fp8_e5m2_to_fp32(raw)
    return fp8_e4m3fn_to_fp32(raw)

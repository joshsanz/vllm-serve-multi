"""Opt-in GB10 kernels used inside SGLang's native Ripple model.

Keep scheduling, attention, KV allocation and speculative acceptance in SGLang.
These kernels retain the standalone runtime's BF16 Q/g and fused NVFP4 work.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _head_norm(x, weight, rows: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, D)
    value = tl.load(x + row * D + col).to(tl.float32)
    scale = tl.rsqrt(tl.sum(value * value, 0) / D + EPS)
    result = value * scale * tl.load(weight + col).to(tl.float32)
    tl.store(x + row * D + col, result)


def head_norm_(x, weight, eps):
    """Per-head RMSNorm with FP32 reduction; leave RoPE to SGLang."""
    d = weight.numel()
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
            and weight.device == x.device and weight.is_contiguous()
            and d == 128 and x.numel() % d == 0):
        raise ValueError("Unsupported head RMSNorm shape or layout")
    _head_norm[(x.numel() // d,)](x, weight, x.numel() // d, d, float(eps),
                                  num_warps=4, enable_fp_fusion=False)
    return x


@triton.jit
def _gate_mul(out, gate, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(out + i, i < N, other=0).to(tl.float32)
    g = tl.load(gate + i, i < N, other=0).to(tl.float32)
    # Keep the model's BF16 rounding between sigmoid and multiplication.
    sigmoid = (1. / (1. + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out + i, value * sigmoid, i < N)


def gate_mul_(out, gate):
    """Fuse the FP32 sigmoid, BF16 cast and attention-output multiplication."""
    if not (out.is_cuda and out.device == gate.device and out.shape == gate.shape
            and out.dtype == gate.dtype == torch.bfloat16
            and out.is_contiguous() and gate.is_contiguous()):
        raise ValueError("Unsupported attention gate shape or layout")
    _gate_mul[(triton.cdiv(out.numel(), 1024),)](out, gate, out.numel(), 1024,
                                              num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _stream(x, w, y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SX: tl.constexpr, BM: tl.constexpr):
    n = tl.program_id(0) * 64 + tl.arange(0, 64)
    m = tl.arange(0, BM)
    k = tl.arange(0, 128)
    acc = tl.zeros((64, BM), tl.float32)
    for start in range(0, K, 128):
        weight = tl.load(w + n[:, None] * K + (start + k)[None, :], eviction_policy="evict_first")
        inp = tl.load(x + m[:, None] * SX + (start + k)[None, :], mask=m[:, None] < M, other=0.)
        acc = tl.dot(weight, tl.trans(inp), acc)
    tl.store(y + m[None, :] * N + n[:, None], acc.to(tl.bfloat16), mask=m[None, :] < M)


def stream_qg(x, weight):
    """FP32 accumulation, one contiguous weight panel per CTA; Q/g only."""
    m, k = x.shape
    n = weight.shape[0]
    if not (x.is_cuda and x.device == weight.device and x.dtype == weight.dtype == torch.bfloat16
            and weight.shape == (10240, 5120) and k == 5120 and 1 <= m <= 128
            and x.stride(1) == 1 and weight.is_contiguous()):
        raise ValueError("Unsupported BF16 Q/g streaming shape or layout")
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _stream[(n // 64,)](x, weight, y, m, n, k, x.stride(0),
                       max(16, triton.next_power_of_2(m)), num_stages=3, num_warps=4)
    return y


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x],
                                    dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _e2m1(v):
    a = tl.abs(v)
    c = tl.where(a <= .25, 0, tl.where(a < .75, 1, tl.where(a <= 1.25, 2,
        tl.where(a < 1.75, 3, tl.where(a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5., 6, 7)))))))
    # Hardware E2M1 preserves negative zero; a floating comparison loses its sign.
    sign = (v.to(tl.int32, bitcast=True) < 0).to(tl.int32) * 8
    return (c + sign).to(tl.uint8)


@triton.jit
def _silu_fp4(x, q, sf, gs, I: tl.constexpr, SX: tl.constexpr, KT: tl.constexpr,
              CLAMP: tl.constexpr):
    # int64 also keeps long-prefill offsets safe at M * 2I >= 2^31.
    m = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * 512 + tl.arange(0, 512)
    g = tl.minimum(tl.load(x + m * SX + col).to(tl.float32), CLAMP)
    u = tl.load(x + m * SX + I + col).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -CLAMP), CLAMP)
    y = (tl.div_rn(g, 1. + tl.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    blocks = tl.reshape(y, (32, 16))
    vmax = tl.max(tl.abs(blocks), axis=1)
    global_scale = tl.load(gs)
    scale = (global_scale * (vmax * _rcp(tl.full((32,), 6., tl.float32)))).to(
        tl.float8e4nv, fp_downcast_rounding="rtne")
    inverse = tl.where(vmax != 0, _rcp(scale.to(tl.float32) * _rcp(
        global_scale + tl.zeros((32,), tl.float32))), 0.)
    codes = _e2m1(blocks * inverse[:, None])
    lo, hi = tl.split(tl.reshape(codes, (256, 2)))
    tl.store(q + m * (I // 2) + tl.program_id(1) * 256 + tl.arange(0, 256), lo | (hi << 4))
    # FlashInfer's swizzled 128-row × 4-block scale layout.
    k = tl.program_id(1) * 32 + tl.arange(0, 32)
    offset = (((m // 128) * KT + k // 4) * 32 + m % 32) * 16 + (m % 128) // 32 * 4 + k % 4
    tl.store(sf + offset, scale.to(tl.uint8, bitcast=True))


def silu_fp4(gate_up, clamp, global_scale):
    """Clamp → BF16 SwiGLU → packed NVFP4, using the native prequantized-input hook."""
    m, two_i = gate_up.shape
    i = two_i // 2
    if not (gate_up.is_cuda and gate_up.dtype == torch.bfloat16 and m > 0
            and two_i % 1024 == 0 and gate_up.stride(1) == 1
            and global_scale.device == gate_up.device and global_scale.dtype == torch.float32
            and global_scale.numel() == 1):
        raise ValueError("Unsupported SwiGLU/NVFP4 shape or scale")
    q = torch.empty((m, i // 2), device=gate_up.device, dtype=torch.uint8)
    sf = torch.zeros((triton.cdiv(m, 128) * 128, i // 16), device=gate_up.device, dtype=torch.uint8)
    _silu_fp4[(m, i // 512)](gate_up, q, sf, global_scale, i, gate_up.stride(0),
                            i // 64, float(clamp), num_warps=4)
    return q, sf

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


if triton is not None:
    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['height', 'width', 'channels'],
    )
    @triton.jit
    def _source_merge_kernel(
            first, second, logits, first_valid, second_valid, output,
            first_stride_b, first_stride_c, first_stride_h, first_stride_w,
            second_stride_b, second_stride_c, second_stride_h, second_stride_w,
            logits_stride_b, logits_stride_c, logits_stride_h, logits_stride_w,
            first_valid_stride_b, first_valid_stride_h, first_valid_stride_w,
            second_valid_stride_b, second_valid_stride_h, second_valid_stride_w,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            height: tl.constexpr, width: tl.constexpr,
            channels: tl.constexpr, block_channels: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        channel = tl.arange(0, block_channels)
        channel_mask = channel < channels
        valid_offset0 = (batch * first_valid_stride_b + y * first_valid_stride_h +
                         x * first_valid_stride_w)
        valid_offset1 = (batch * second_valid_stride_b + y * second_valid_stride_h +
                         x * second_valid_stride_w)
        valid0 = tl.load(first_valid + valid_offset0).to(tl.int1)
        valid1 = tl.load(second_valid + valid_offset1).to(tl.int1)
        logits_base = batch * logits_stride_b + y * logits_stride_h + x * logits_stride_w
        logit0 = tl.load(logits + logits_base).to(tl.float32)
        logit1 = tl.load(logits + logits_base + logits_stride_c).to(tl.float32)
        maximum = tl.maximum(tl.where(valid0, logit0, -float('inf')),
                             tl.where(valid1, logit1, -float('inf')))
        maximum = tl.where(valid0 | valid1, maximum, 0.0)
        exp0 = tl.where(valid0, libdevice.exp(logit0 - maximum), 0.0)
        exp1 = tl.where(valid1, libdevice.exp(logit1 - maximum), 0.0)
        denominator = tl.maximum(exp0 + exp1, 1e-12)
        weight0 = (exp0 / denominator).to(tl.float16)
        weight1 = (exp1 / denominator).to(tl.float16)
        weight_sum = tl.inline_asm_elementwise(
            asm='add.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[weight0, weight1], dtype=tl.float16, is_pure=True, pack=1)
        weight_sum = tl.maximum(weight_sum, 6.103515625e-05)
        weight0 = (weight0.to(tl.float32) / weight_sum.to(tl.float32)).to(tl.float16)
        weight1 = (weight1.to(tl.float32) / weight_sum.to(tl.float32)).to(tl.float16)
        offset0 = (batch * first_stride_b + channel * first_stride_c +
                   y * first_stride_h + x * first_stride_w)
        offset1 = (batch * second_stride_b + channel * second_stride_c +
                   y * second_stride_h + x * second_stride_w)
        value0 = tl.load(first + offset0, mask=channel_mask, other=0.0).to(tl.float16)
        value1 = tl.load(second + offset1, mask=channel_mask, other=0.0).to(tl.float16)
        scaled0 = tl.inline_asm_elementwise(
            asm='mul.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[value0, weight0], dtype=tl.float16, is_pure=True, pack=1)
        scaled1 = tl.inline_asm_elementwise(
            asm='mul.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[value1, weight1], dtype=tl.float16, is_pure=True, pack=1)
        merged = tl.inline_asm_elementwise(
            asm='add.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[scaled0, scaled1], dtype=tl.float16, is_pure=True, pack=1)
        output_offset = (batch * output_stride_b + channel * output_stride_c +
                         y * output_stride_h + x * output_stride_w)
        tl.store(output + output_offset, merged, mask=channel_mask)


def fused_source_merge(first, second, logits, first_valid, second_valid):
    """Fuse two-source masked softmax and feature reduction for FP16 inference."""
    if triton is None:
        raise RuntimeError('Triton is unavailable')
    batch, channels, height, width = first.shape
    output = torch.empty_like(first, memory_format=torch.preserve_format)
    block_channels = triton.next_power_of_2(channels)
    _source_merge_kernel[(batch * height * width,)](
        first, second, logits, first_valid, second_valid, output,
        *first.stride(), *second.stride(), *logits.stride(),
        first_valid.stride(0), first_valid.stride(2), first_valid.stride(3),
        second_valid.stride(0), second_valid.stride(2), second_valid.stride(3),
        *output.stride(), height, width, channels, block_channels,
    )
    return output


def fused_brr_available():
    return triton is not None

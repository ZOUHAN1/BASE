import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # The eager implementation remains the portable fallback.
    triton = None
    tl = None


if triton is not None:
    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['source_h', 'source_w', 'channels', 'positions'],
    )
    @triton.jit
    def _bilinear_sample_kernel(values, coords, output,
                                value_stride_b, value_stride_c,
                                value_stride_h, value_stride_w,
                                coord_stride_b, coord_stride_h, coord_stride_w,
                                coord_stride_p, coord_stride_xy,
                                output_stride_b, output_stride_c,
                                output_stride_h, output_stride_w,
                                source_h: tl.constexpr, source_w: tl.constexpr,
                                anchor_w: tl.constexpr,
                                channels: tl.constexpr, positions: tl.constexpr,
                                block_channels: tl.constexpr):
        position = tl.program_id(0)
        batch = position // positions
        spatial = position - batch * positions
        anchor_y = spatial // anchor_w
        anchor_x = spatial - anchor_y * anchor_w
        channel = tl.arange(0, block_channels)
        channel_mask = channel < channels
        coord_base = (batch * coord_stride_b + anchor_y * coord_stride_h +
                      anchor_x * coord_stride_w)
        x = tl.load(coords + coord_base).to(tl.float32)
        y = tl.load(coords + coord_base + coord_stride_xy).to(tl.float32)
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0
        value_base = batch * value_stride_b + channel * value_stride_c
        in00 = (x0 >= 0) & (x0 < source_w) & (y0 >= 0) & (y0 < source_h)
        in01 = (x1 >= 0) & (x1 < source_w) & (y0 >= 0) & (y0 < source_h)
        in10 = (x0 >= 0) & (x0 < source_w) & (y1 >= 0) & (y1 < source_h)
        in11 = (x1 >= 0) & (x1 < source_w) & (y1 >= 0) & (y1 < source_h)
        v00 = tl.load(values + value_base + y0 * value_stride_h + x0 * value_stride_w,
                      mask=channel_mask & in00, other=0.0).to(tl.float32)
        v01 = tl.load(values + value_base + y0 * value_stride_h + x1 * value_stride_w,
                      mask=channel_mask & in01, other=0.0).to(tl.float32)
        v10 = tl.load(values + value_base + y1 * value_stride_h + x0 * value_stride_w,
                      mask=channel_mask & in10, other=0.0).to(tl.float32)
        v11 = tl.load(values + value_base + y1 * value_stride_h + x1 * value_stride_w,
                      mask=channel_mask & in11, other=0.0).to(tl.float32)
        result = ((1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v01 +
                  (1.0 - dx) * dy * v10 + dx * dy * v11)
        output_base = (batch * output_stride_b + channel * output_stride_c +
                       anchor_y * output_stride_h + anchor_x * output_stride_w)
        tl.store(output + output_base, result, mask=channel_mask)


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['source_h', 'source_w', 'channels', 'positions', 'derive_coords',
             'implicit_valid', 'store_valid'],
    )
    @triton.jit
    def _sample_aggregate_kernel(values, mask, coords, control, template, offset_scale,
                                 output, support_valid,
                                 value_stride_b, value_stride_c,
                                 value_stride_h, value_stride_w,
                                 mask_stride_b, mask_stride_h, mask_stride_w,
                                 coord_stride_b, coord_stride_h, coord_stride_w,
                                 coord_stride_p, coord_stride_xy,
                                 control_stride_b, control_stride_c,
                                 control_stride_h, control_stride_w,
                                 output_stride_b, output_stride_c,
                                 output_stride_h, output_stride_w,
                                 valid_stride_b, valid_stride_h, valid_stride_w,
                                 source_h: tl.constexpr, source_w: tl.constexpr,
                                 anchor_w: tl.constexpr,
                                 channels: tl.constexpr, positions: tl.constexpr,
                                 points: tl.constexpr, control_channels: tl.constexpr,
                                 block_channels: tl.constexpr, block_points: tl.constexpr,
                                 derive_coords: tl.constexpr,
                                 implicit_valid: tl.constexpr,
                                 store_valid: tl.constexpr):
        position = tl.program_id(0)
        batch = position // positions
        spatial = position - batch * positions
        anchor_y = spatial // anchor_w
        anchor_x = spatial - anchor_y * anchor_w
        point = tl.arange(0, block_points)
        channel = tl.arange(0, block_channels)
        point_mask = point < points

        control_base = (batch * control_stride_b + anchor_y * control_stride_h +
                        anchor_x * control_stride_w)
        coord_base = (batch * coord_stride_b + anchor_y * coord_stride_h +
                      anchor_x * coord_stride_w)
        if derive_coords:
            center_x = tl.load(coords + coord_base).to(tl.float32)
            center_y = tl.load(coords + coord_base + coord_stride_xy).to(tl.float32)
            raw_x = tl.load(control + control_base + (2 * point) * control_stride_c,
                            mask=point_mask, other=0.0).to(tl.float32)
            raw_y = tl.load(control + control_base + (2 * point + 1) * control_stride_c,
                            mask=point_mask, other=0.0).to(tl.float32)
            offset_x = (2.0 / (1.0 + tl.exp(-2.0 * raw_x)) - 1.0).to(tl.float16)
            offset_y = (2.0 / (1.0 + tl.exp(-2.0 * raw_y)) - 1.0).to(tl.float16)
            scale = tl.load(offset_scale + point, mask=point_mask, other=0.0).to(tl.float16)
            template_x = tl.load(template + point * 2,
                                 mask=point_mask, other=0.0).to(tl.float16)
            template_y = tl.load(template + point * 2 + 1,
                                 mask=point_mask, other=0.0).to(tl.float16)
            x = ((center_x.to(tl.float16) + template_x).to(tl.float16) +
                 (scale * offset_x).to(tl.float16)).to(tl.float16).to(tl.float32)
            y = ((center_y.to(tl.float16) + template_y).to(tl.float16) +
                 (scale * offset_y).to(tl.float16)).to(tl.float16).to(tl.float32)
        else:
            point_coord_base = coord_base + point * coord_stride_p
            x = tl.load(coords + point_coord_base, mask=point_mask, other=0.0).to(tl.float32)
            y = tl.load(coords + point_coord_base + coord_stride_xy,
                        mask=point_mask, other=0.0).to(tl.float32)
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0

        mask_base = batch * mask_stride_b
        in00 = (x0 >= 0) & (x0 < source_w) & (y0 >= 0) & (y0 < source_h) & point_mask
        in01 = (x1 >= 0) & (x1 < source_w) & (y0 >= 0) & (y0 < source_h) & point_mask
        in10 = (x0 >= 0) & (x0 < source_w) & (y1 >= 0) & (y1 < source_h) & point_mask
        in11 = (x1 >= 0) & (x1 < source_w) & (y1 >= 0) & (y1 < source_h) & point_mask
        index00 = y0 * mask_stride_h + x0 * mask_stride_w
        index01 = y0 * mask_stride_h + x1 * mask_stride_w
        index10 = y1 * mask_stride_h + x0 * mask_stride_w
        index11 = y1 * mask_stride_h + x1 * mask_stride_w
        if implicit_valid:
            m00 = in00.to(tl.float32)
            m01 = in01.to(tl.float32)
            m10 = in10.to(tl.float32)
            m11 = in11.to(tl.float32)
        else:
            m00 = tl.load(mask + mask_base + index00, mask=in00, other=0.0).to(tl.float32)
            m01 = tl.load(mask + mask_base + index01, mask=in01, other=0.0).to(tl.float32)
            m10 = tl.load(mask + mask_base + index10, mask=in10, other=0.0).to(tl.float32)
            m11 = tl.load(mask + mask_base + index11, mask=in11, other=0.0).to(tl.float32)
        sampled_mask = ((1.0 - dx) * (1.0 - dy) * m00 + dx * (1.0 - dy) * m01 +
                        (1.0 - dx) * dy * m10 + dx * dy * m11)
        valid = sampled_mask >= 0.999999

        logits = tl.load(control + control_base + (34 + point) * control_stride_c,
                         mask=point_mask, other=0.0).to(tl.float32)
        is_support = valid & (point < 9)
        is_reference = valid & (point >= 9) & point_mask
        has_support = tl.sum(is_support.to(tl.int32), axis=0) > 0
        has_reference = tl.sum(is_reference.to(tl.int32), axis=0) > 0
        support_max = tl.max(tl.where(is_support, logits, -float('inf')), axis=0)
        reference_max = tl.max(tl.where(is_reference, logits, -float('inf')), axis=0)
        support_max = tl.where(has_support, support_max, 0.0)
        reference_max = tl.where(has_reference, reference_max, 0.0)
        support_exp = tl.where(is_support, tl.exp(logits - support_max), 0.0)
        reference_exp = tl.where(is_reference, tl.exp(logits - reference_max), 0.0)
        support_weight = support_exp / tl.maximum(tl.sum(support_exp, axis=0), 1e-12)
        reference_weight = reference_exp / tl.maximum(tl.sum(reference_exp, axis=0), 1e-12)

        value_base = batch * value_stride_b
        value_channel = channel[None, :] * value_stride_c
        channel_mask = channel[None, :] < channels
        value_index00 = y0 * value_stride_h + x0 * value_stride_w
        value_index01 = y0 * value_stride_h + x1 * value_stride_w
        value_index10 = y1 * value_stride_h + x0 * value_stride_w
        value_index11 = y1 * value_stride_h + x1 * value_stride_w
        v00 = tl.load(values + value_base + value_channel + value_index00[:, None],
                      mask=in00[:, None] & channel_mask, other=0.0).to(tl.float32)
        v01 = tl.load(values + value_base + value_channel + value_index01[:, None],
                      mask=in01[:, None] & channel_mask, other=0.0).to(tl.float32)
        v10 = tl.load(values + value_base + value_channel + value_index10[:, None],
                      mask=in10[:, None] & channel_mask, other=0.0).to(tl.float32)
        v11 = tl.load(values + value_base + value_channel + value_index11[:, None],
                      mask=in11[:, None] & channel_mask, other=0.0).to(tl.float32)
        sampled = ((1.0 - dx[:, None]) * (1.0 - dy[:, None]) * v00 +
                   dx[:, None] * (1.0 - dy[:, None]) * v01 +
                   (1.0 - dx[:, None]) * dy[:, None] * v10 +
                   dx[:, None] * dy[:, None] * v11)
        support = tl.sum(sampled * support_weight[:, None], axis=0)
        reference = tl.sum(sampled * reference_weight[:, None], axis=0)
        gate = tl.load(control + control_base + 51 * control_stride_c).to(tl.float32)
        gate = 2.0 / (1.0 + tl.exp(-2.0 * gate)) - 1.0
        result = (support + gate * (support - reference) * has_reference) * has_support
        output_base = (batch * output_stride_b + channel * output_stride_c +
                       anchor_y * output_stride_h + anchor_x * output_stride_w)
        tl.store(output + output_base, result, mask=channel < channels)
        valid_base = (batch * valid_stride_b + anchor_y * valid_stride_h +
                      anchor_x * valid_stride_w)
        if store_valid:
            tl.store(support_valid + valid_base, has_support)


def fused_sample_aggregate(values, mask, coords, control, implicit_valid=False,
                           store_valid=True):
    """Fused inference form of BRR sampling, masked softmax, and point reduction."""
    if triton is None:
        raise RuntimeError('Triton is unavailable')
    batch, channels, source_h, source_w = values.shape
    _, anchor_h, anchor_w, points, _ = coords.shape
    positions = anchor_h * anchor_w
    memory_format = (torch.channels_last if values.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty((batch, channels, anchor_h, anchor_w), device=values.device,
                         dtype=values.dtype, memory_format=memory_format)
    support_valid = (torch.empty((batch, 1, anchor_h, anchor_w), device=values.device,
                                 dtype=torch.bool, memory_format=memory_format)
                     if store_valid else None)
    valid_storage = support_valid if support_valid is not None else output[:, :1]
    block_channels = triton.next_power_of_2(channels)
    _sample_aggregate_kernel[(batch * positions,)](
        values, mask, coords, control, coords, coords, output, valid_storage,
        *values.stride(), mask.stride(0), mask.stride(2), mask.stride(3),
        *coords.stride(), *control.stride(), *output.stride(),
        valid_storage.stride(0), valid_storage.stride(2), valid_storage.stride(3),
        source_h, source_w, anchor_w, channels, positions, points, control.shape[1],
        block_channels, triton.next_power_of_2(points), False, implicit_valid, store_valid,
    )
    return output, support_valid


def fused_sample_aggregate_control(values, mask, center, template, offset_scale, control,
                                   implicit_valid=False, store_valid=True):
    """Fused BRR sampling that derives all point coordinates inside the kernel."""
    if triton is None:
        raise RuntimeError('Triton is unavailable')
    batch, channels, source_h, source_w = values.shape
    _, anchor_h, anchor_w, _, _ = center.shape
    points = template.shape[0]
    positions = anchor_h * anchor_w
    memory_format = (torch.channels_last if values.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty((batch, channels, anchor_h, anchor_w), device=values.device,
                         dtype=values.dtype, memory_format=memory_format)
    support_valid = (torch.empty((batch, 1, anchor_h, anchor_w), device=values.device,
                                 dtype=torch.bool, memory_format=memory_format)
                     if store_valid else None)
    valid_storage = support_valid if support_valid is not None else output[:, :1]
    block_channels = triton.next_power_of_2(channels)
    _sample_aggregate_kernel[(batch * positions,)](
        values, mask, center, control, template, offset_scale, output, valid_storage,
        *values.stride(), mask.stride(0), mask.stride(2), mask.stride(3),
        *center.stride(), *control.stride(), *output.stride(),
        valid_storage.stride(0), valid_storage.stride(2), valid_storage.stride(3),
        source_h, source_w, anchor_w, channels, positions, points, control.shape[1],
        block_channels, triton.next_power_of_2(points), True, implicit_valid, store_valid,
    )
    return output, support_valid


def fused_bilinear_sample(values, coords):
    """Bilinear zero-padded point sampling for the BRR center-key fast path."""
    if triton is None:
        raise RuntimeError('Triton is unavailable')
    batch, channels, source_h, source_w = values.shape
    _, anchor_h, anchor_w, points, _ = coords.shape
    if points != 1:
        raise ValueError('The fused center sampler expects one point per location')
    positions = anchor_h * anchor_w
    memory_format = (torch.channels_last if values.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty((batch, channels, anchor_h, anchor_w), device=values.device,
                         dtype=values.dtype, memory_format=memory_format)
    block_channels = triton.next_power_of_2(channels)
    _bilinear_sample_kernel[(batch * positions,)](
        values, coords, output, *values.stride(), *coords.stride(), *output.stride(),
        source_h, source_w, anchor_w, channels, positions,
        block_channels,
    )
    return output


def fused_sampling_available():
    return triton is not None

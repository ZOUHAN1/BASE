import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # The reference PyTorch implementation remains the fallback.
    triton = None
    tl = None


if triton is not None:
    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)],
        key=['height', 'width'],
    )
    @triton.jit
    def _gate_kernel(
            query, key, first_weight, first_bias, last_weight, last_bias, gate,
            query_stride_b, query_stride_c, query_stride_h, query_stride_w,
            key_stride_b, key_stride_c, key_stride_h, key_stride_w,
            first_stride_o, first_stride_i,
            gate_stride_b, gate_stride_h, gate_stride_w,
            height: tl.constexpr, width: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        input_channel = tl.arange(0, 32)
        hidden_channel = tl.arange(0, 32)
        query_value = tl.load(
            query + batch * query_stride_b + input_channel * query_stride_c +
            y * query_stride_h + x * query_stride_w,
        ).to(tl.float16)
        key_value = tl.load(
            key + batch * key_stride_b + input_channel * key_stride_c +
            y * key_stride_h + x * key_stride_w,
        ).to(tl.float16)
        query_weight = tl.load(
            first_weight + hidden_channel[None, :] * first_stride_o +
            input_channel[:, None] * first_stride_i,
        ).to(tl.float16)
        key_weight = tl.load(
            first_weight + hidden_channel[None, :] * first_stride_o +
            (32 + input_channel[:, None]) * first_stride_i,
        ).to(tl.float16)
        hidden = tl.sum(query_value[:, None].to(tl.float32) *
                        query_weight.to(tl.float32), axis=0)
        hidden += tl.sum(key_value[:, None].to(tl.float32) *
                         key_weight.to(tl.float32), axis=0)
        hidden += tl.load(first_bias + hidden_channel).to(tl.float32)
        hidden = hidden.to(tl.float16).to(tl.float32)
        hidden = (hidden / (1.0 + tl.exp(-hidden))).to(tl.float16)
        final_weight = tl.load(last_weight + hidden_channel).to(tl.float16)
        result = tl.sum(hidden.to(tl.float32) * final_weight.to(tl.float32), axis=0)
        result += tl.load(last_bias).to(tl.float32)
        result = result.to(tl.float16).to(tl.float32)
        result = (1.0 / (1.0 + tl.exp(-result))).to(tl.float16)
        gate_offset = batch * gate_stride_b + y * gate_stride_h + x * gate_stride_w
        tl.store(gate + gate_offset, result)


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (2, 4, 8)],
        key=['height', 'width', 'channels'],
    )
    @triton.jit
    def _relay_kernel(
            base, value, gate, output,
            base_stride_b, base_stride_c, base_stride_h, base_stride_w,
            value_stride_b, value_stride_c, value_stride_h, value_stride_w,
            gate_stride_b, gate_stride_h, gate_stride_w,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            height: tl.constexpr, width: tl.constexpr,
            channels: tl.constexpr, block_channels: tl.constexpr):
        position = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        channel = channel_block * block_channels + tl.arange(0, block_channels)
        channel_mask = channel < channels
        base_offset = (batch * base_stride_b + channel * base_stride_c +
                       y * base_stride_h + x * base_stride_w)
        value_offset = (batch * value_stride_b + channel * value_stride_c +
                        y * value_stride_h + x * value_stride_w)
        gate_offset = batch * gate_stride_b + y * gate_stride_h + x * gate_stride_w
        base_value = tl.load(base + base_offset, mask=channel_mask).to(tl.float16)
        relay_value = tl.load(value + value_offset, mask=channel_mask).to(tl.float16)
        gate_value = tl.load(gate + gate_offset).to(tl.float16)
        scaled = tl.inline_asm_elementwise(
            asm='mul.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[gate_value, relay_value], dtype=tl.float16, is_pure=True, pack=1)
        result = tl.inline_asm_elementwise(
            asm='add.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[base_value, scaled], dtype=tl.float16, is_pure=True, pack=1)
        output_offset = (batch * output_stride_b + channel * output_stride_c +
                         y * output_stride_h + x * output_stride_w)
        tl.store(output + output_offset, result, mask=channel_mask)


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['factor', 'height', 'width'],
    )
    @triton.jit
    def _score_kernel(
            query, key, first_weight, first_bias, last_weight, last_bias, scores,
            query_stride_b, query_stride_c, query_stride_h, query_stride_w,
            key_stride_b, key_stride_c, key_stride_h, key_stride_w,
            first_stride_o, first_stride_i,
            scores_stride_b, scores_stride_k, scores_stride_h, scores_stride_w,
            source_h: tl.constexpr, source_w: tl.constexpr,
            height: tl.constexpr, width: tl.constexpr,
            factor: tl.constexpr, candidates: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        candidate_index = tl.arange(0, 16)
        candidate = candidate_index[:, None]
        input_channel = tl.arange(0, 32)[None, :]
        hidden_channel = tl.arange(0, 32)[None, :]
        offset_y = candidate // factor
        offset_x = candidate - offset_y * factor
        source_y = y * factor + offset_y
        source_x = x * factor + offset_x
        candidate_mask = candidate_index < candidates
        inside = (candidate_mask[:, None] & (source_y < source_h) & (source_x < source_w))
        key_offset = (batch * key_stride_b + input_channel * key_stride_c +
                      source_y * key_stride_h + source_x * key_stride_w)
        key_value = tl.load(key + key_offset, mask=inside, other=0.0).to(tl.float16)
        weight_key = tl.load(
            first_weight + (32 + input_channel.T) * first_stride_i +
            hidden_channel * first_stride_o,
        ).to(tl.float16)
        key_score = tl.dot(key_value, weight_key)

        query_value = tl.load(
            query + batch * query_stride_b + input_channel * query_stride_c +
            y * query_stride_h + x * query_stride_w,
        ).to(tl.float16)
        weight_query = tl.load(
            first_weight + input_channel.T * first_stride_i + hidden_channel * first_stride_o,
        ).to(tl.float16)
        query_score = tl.sum(query_value.T * weight_query, axis=0)[None, :]
        bias = tl.load(first_bias + hidden_channel).to(tl.float32)
        hidden = (key_score + query_score + bias).to(tl.float16).to(tl.float32)
        hidden = (hidden / (1.0 + tl.exp(-hidden))).to(tl.float16)
        final_weight = tl.load(last_weight + tl.arange(0, 32)[None, :]).to(tl.float16)
        score = tl.sum(hidden.to(tl.float32) * final_weight.to(tl.float32), axis=1)
        score += tl.load(last_bias).to(tl.float32)
        score_offset = (batch * scores_stride_b + candidate_index * scores_stride_k +
                        y * scores_stride_h + x * scores_stride_w)
        tl.store(scores + score_offset, score, mask=candidate_mask)


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['factor', 'height', 'width', 'implicit_valid', 'store_weights'],
    )
    @triton.jit
    def _aggregate_kernel(
            scores, position_bias, correction, key, valid, weights, output, key_output,
            scores_stride_b, scores_stride_k, scores_stride_h, scores_stride_w,
            correction_stride_b, correction_stride_c, correction_stride_h, correction_stride_w,
            key_stride_b, key_stride_c, key_stride_h, key_stride_w,
            valid_stride_b, valid_stride_h, valid_stride_w,
            weights_stride_b, weights_stride_k, weights_stride_h, weights_stride_w,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            key_output_stride_b, key_output_stride_c, key_output_stride_h, key_output_stride_w,
            source_h: tl.constexpr, source_w: tl.constexpr,
            height: tl.constexpr, width: tl.constexpr,
            factor: tl.constexpr, candidates: tl.constexpr,
            channels: tl.constexpr, block_channels: tl.constexpr,
            implicit_valid: tl.constexpr, store_weights: tl.constexpr):
        position = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        candidate = tl.arange(0, 16)
        candidate_mask = candidate < candidates
        offset_y = candidate // factor
        offset_x = candidate - offset_y * factor
        source_y = y * factor + offset_y
        source_x = x * factor + offset_x
        inside = ((source_y >= 0) & (source_y < source_h) &
                  (source_x >= 0) & (source_x < source_w) & candidate_mask)
        valid_offset = (batch * valid_stride_b + source_y * valid_stride_h +
                        source_x * valid_stride_w)
        if implicit_valid:
            candidate_valid = inside
        else:
            candidate_valid = tl.load(valid + valid_offset, mask=inside, other=0).to(tl.int1)
        score_offset = (batch * scores_stride_b + candidate * scores_stride_k +
                        y * scores_stride_h + x * scores_stride_w)
        logits = tl.load(scores + score_offset, mask=candidate_mask, other=0.0).to(tl.float32)
        logits += tl.load(position_bias + candidate, mask=candidate_mask, other=0.0).to(tl.float32)
        has_valid = tl.sum(candidate_valid.to(tl.int32), axis=0) > 0
        maximum = tl.max(tl.where(candidate_valid, logits, -float('inf')), axis=0)
        maximum = tl.where(has_valid, maximum, 0.0)
        exponent = tl.where(candidate_valid, tl.exp(logits - maximum), 0.0)
        weight = exponent / tl.maximum(tl.sum(exponent, axis=0), 1e-12)

        channel = channel_block * block_channels + tl.arange(0, block_channels)
        channel_mask = channel < channels
        correction_offset = (batch * correction_stride_b +
                             channel[None, :] * correction_stride_c +
                             source_y[:, None] * correction_stride_h +
                             source_x[:, None] * correction_stride_w)
        candidate_value = tl.load(
            correction + correction_offset,
            mask=inside[:, None] & channel_mask[None, :], other=0.0,
        ).to(tl.float32)
        aggregated = tl.sum(candidate_value * weight[:, None], axis=0)
        output_offset = (batch * output_stride_b + channel * output_stride_c +
                         y * output_stride_h + x * output_stride_w)
        tl.store(output + output_offset, aggregated, mask=channel_mask)

        if channel_block == 0:
            if store_weights:
                weight_offset = (batch * weights_stride_b + candidate * weights_stride_k +
                                 y * weights_stride_h + x * weights_stride_w)
                tl.store(weights + weight_offset, weight, mask=candidate_mask)
            key_channel = tl.arange(0, 32)
            key_offset = (batch * key_stride_b + key_channel[None, :] * key_stride_c +
                          source_y[:, None] * key_stride_h + source_x[:, None] * key_stride_w)
            candidate_key = tl.load(
                key + key_offset, mask=inside[:, None], other=0.0,
            ).to(tl.float32)
            aggregated_key = tl.sum(candidate_key * weight[:, None], axis=0)
            key_output_offset = (batch * key_output_stride_b +
                                 key_channel * key_output_stride_c +
                                 y * key_output_stride_h + x * key_output_stride_w)
            tl.store(key_output + key_output_offset, aggregated_key)


def fused_arr_aggregate(scores, position_bias, correction, key, valid, factor,
                        implicit_valid=False, store_weights=False):
    """Fuse ARR candidate packing, masked softmax, and weighted reductions."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if scores.dtype != torch.float16:
        raise ValueError("The fused ARR inference kernel currently expects FP16 scores")
    batch, candidates, height, width = scores.shape
    source_h, source_w = correction.shape[-2:]
    memory_format = (torch.channels_last
                     if correction.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    weights = (torch.empty_like(scores, memory_format=torch.preserve_format)
               if store_weights else None)
    weight_storage = weights if weights is not None else scores
    output = torch.empty((batch, 256, height, width), device=correction.device,
                         dtype=correction.dtype, memory_format=memory_format)
    key_output = torch.empty((batch, 32, height, width), device=key.device,
                             dtype=key.dtype, memory_format=memory_format)
    block_channels = 128
    _aggregate_kernel[(batch * height * width, triton.cdiv(256, block_channels))](
        scores, position_bias, correction, key, valid, weight_storage, output, key_output,
        *scores.stride(), *correction.stride(), *key.stride(),
        valid.stride(0), valid.stride(2), valid.stride(3), *weight_storage.stride(),
        *output.stride(), *key_output.stride(), source_h, source_w, height, width,
        factor, candidates, 256, block_channels, implicit_valid, store_weights,
    )
    return output, key_output, weights


def fused_arr_score(query, key, first, last, factor):
    """Evaluate the ARR candidate score MLP without materializing packed keys."""
    batch, _, height, width = query.shape
    source_h, source_w = key.shape[-2:]
    candidates = factor * factor
    scores = torch.empty((batch, candidates, height, width), device=query.device,
                         dtype=query.dtype, memory_format=torch.channels_last)
    _score_kernel[(batch * height * width,)](
        query, key, first.weight, first.bias, last.weight, last.bias, scores,
        *query.stride(), *key.stride(), first.weight.stride(0), first.weight.stride(1),
        *scores.stride(), source_h, source_w, height, width, factor, candidates,
    )
    return scores


def fused_arr_relay(base, value, gate):
    """Fuse FP16 gate multiplication and residual addition without changing rounding order."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, height, width = base.shape
    memory_format = (torch.channels_last
                     if base.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty_like(base, memory_format=torch.preserve_format)
    block_channels = 128
    _relay_kernel[(batch * height * width, triton.cdiv(channels, block_channels))](
        base, value, gate, output, *base.stride(), *value.stride(),
        gate.stride(0), gate.stride(2), gate.stride(3), *output.stride(),
        height, width, channels, block_channels,
    )
    return output


def fused_arr_gate(query, key, first, last):
    """Fuse the ARR 64 -> 32 -> 1 gate MLP, SiLU, and sigmoid."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, height, width = query.shape
    if channels != 32 or key.shape[1] != 32:
        raise ValueError("The fused ARR gate expects two 32-channel inputs")
    memory_format = (torch.channels_last
                     if query.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    gate = torch.empty((batch, 1, height, width), device=query.device,
                       dtype=query.dtype, memory_format=memory_format)
    _gate_kernel[(batch * height * width,)](
        query, key, first.weight, first.bias, last.weight, last.bias, gate,
        *query.stride(), *key.stride(), first.weight.stride(0), first.weight.stride(1),
        gate.stride(0), gate.stride(2), gate.stride(3), height, width,
    )
    return gate


def fused_arr_available():
    return triton is not None

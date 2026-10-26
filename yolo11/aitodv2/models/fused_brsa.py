import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # The reference PyTorch implementation remains the fallback.
    triton = None
    tl = None
    libdevice = None


if triton is not None:
    @triton.jit
    def _reference_input_kernel(
            frequency, semantic, output,
            frequency_stride_b, frequency_stride_c, frequency_stride_h, frequency_stride_w,
            semantic_stride_b, semantic_stride_c, semantic_stride_h, semantic_stride_w,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            height: tl.constexpr, width: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        channel = tl.arange(0, 128)
        frequency_value = tl.load(
            frequency + batch * frequency_stride_b + channel * frequency_stride_c +
            y * frequency_stride_h + x * frequency_stride_w,
            mask=channel < 64, other=0.0,
        ).to(tl.float16)
        semantic_channel = channel - 64
        semantic_value = tl.load(
            semantic + batch * semantic_stride_b + semantic_channel * semantic_stride_c +
            y * semantic_stride_h + x * semantic_stride_w,
            mask=(channel >= 64) & (channel < 80), other=0.0,
        ).to(tl.float16)
        value = tl.where(channel < 64, frequency_value, semantic_value)
        tl.store(
            output + batch * output_stride_b + channel * output_stride_c +
            y * output_stride_h + x * output_stride_w,
            value, mask=channel < 80,
        )


    @triton.jit
    def _control_activation_kernel(
            control, activated,
            control_stride_b, control_stride_c, control_stride_h, control_stride_w,
            activated_stride_b, activated_stride_c, activated_stride_h, activated_stride_w,
            height: tl.constexpr, width: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        control_base = (batch * control_stride_b + y * control_stride_h +
                        x * control_stride_w)
        occupancy = tl.load(control + control_base).to(tl.float32)
        alpha0 = tl.load(control + control_base + control_stride_c).to(tl.float32)
        alpha1 = tl.load(control + control_base + 2 * control_stride_c).to(tl.float32)
        probability = 1.0 / (1.0 + tl.exp(-occupancy))
        maximum = tl.maximum(alpha0, alpha1)
        exp0 = tl.exp(alpha0 - maximum)
        exp1 = tl.exp(alpha1 - maximum)
        denominator = exp0 + exp1
        activated_base = (batch * activated_stride_b + y * activated_stride_h +
                          x * activated_stride_w)
        tl.store(activated + activated_base, probability)
        tl.store(activated + activated_base + activated_stride_c, exp0 / denominator)
        tl.store(activated + activated_base + 2 * activated_stride_c, exp1 / denominator)


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)],
        key=['input_h', 'input_w'],
    )
    @triton.jit
    def _silu_avg_pool2d_kernel(
            x, output,
            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            input_h: tl.constexpr, input_w: tl.constexpr,
            output_h: tl.constexpr, output_w: tl.constexpr,
            channels: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (output_h * output_w)
        spatial = position - batch * output_h * output_w
        output_y = spatial // output_w
        output_x = spatial - output_y * output_w
        channel = tl.arange(0, channels)
        total = tl.zeros((channels,), tl.float32)
        for offset_y in tl.static_range(0, 2):
            for offset_x in tl.static_range(0, 2):
                input_y = output_y * 2 + offset_y
                input_x = output_x * 2 + offset_x
                value = tl.load(
                    x + batch * x_stride_b + channel * x_stride_c +
                    input_y * x_stride_h + input_x * x_stride_w,
                ).to(tl.float32)
                activated = value / (1.0 + tl.exp(-value))
                total += activated.to(tl.float16).to(tl.float32)
        pooled = total * 0.25
        tl.store(
            output + batch * output_stride_b + channel * output_stride_c +
            output_y * output_stride_h + output_x * output_stride_w,
            pooled,
        )


    @triton.autotune(
        configs=[
            triton.Config({'block_positions': 8}, num_warps=2),
            triton.Config({'block_positions': 16}, num_warps=4),
            triton.Config({'block_positions': 32}, num_warps=8),
        ],
        key=['height', 'width'],
    )
    @triton.jit
    def _modulate_frequency_kernel(
            z, modulation, rho, output,
            z_stride_b, z_stride_c, z_stride_f, z_stride_h, z_stride_w,
            modulation_stride_b, modulation_stride_c,
            modulation_stride_h, modulation_stride_w,
            rho_stride_b, rho_stride_h, rho_stride_w,
            output_stride_b, output_stride_c, output_stride_f,
            output_stride_h, output_stride_w,
            height: tl.constexpr, width: tl.constexpr,
            block_positions: tl.constexpr):
        group_frequency = tl.program_id(0)
        batch = tl.program_id(1)
        group = group_frequency // 16
        frequency = group_frequency - group * 16
        spatial = tl.program_id(2) * block_positions + tl.arange(0, block_positions)
        valid = spatial < height * width
        y = spatial // width
        x = spatial - y * width
        channel = group * 8 + tl.arange(0, 8)
        modulation_offset = (batch * modulation_stride_b +
                             (group * 16 + frequency) * modulation_stride_c +
                             y * modulation_stride_h + x * modulation_stride_w)
        modulation_value = tl.load(
            modulation + modulation_offset, mask=valid, other=0.0).to(tl.float32)
        modulation_value = libdevice.tanh(modulation_value).to(tl.float16)
        rho_offset = batch * rho_stride_b + y * rho_stride_h + x * rho_stride_w
        rho_value = tl.load(rho + rho_offset, mask=valid, other=0.0).to(tl.float16)
        z_offset = (batch * z_stride_b + channel[None, :] * z_stride_c +
                    frequency * z_stride_f + y[:, None] * z_stride_h +
                    x[:, None] * z_stride_w)
        z_value = tl.load(z + z_offset, mask=valid[:, None], other=0.0).to(tl.float16)
        scaled = tl.inline_asm_elementwise(
            asm='mul.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[z_value, modulation_value[:, None]], dtype=tl.float16,
            is_pure=True, pack=1)
        scaled = tl.inline_asm_elementwise(
            asm='mul.rn.f16 $0, $1, $2;', constraints='=h,h,h',
            args=[scaled, rho_value[:, None]], dtype=tl.float16,
            is_pure=True, pack=1)
        output_offset = (batch * output_stride_b + channel[None, :] * output_stride_c +
                         frequency * output_stride_f + y[:, None] * output_stride_h +
                         x[:, None] * output_stride_w)
        tl.store(output + output_offset, scaled, mask=valid[:, None])


    @triton.autotune(
        configs=[
            triton.Config({'block_x': 4}, num_warps=1),
            triton.Config({'block_x': 8}, num_warps=2),
            triton.Config({'block_x': 16}, num_warps=4),
        ],
        key=['input_h', 'input_w', 'output_h', 'output_w'],
    )
    @triton.jit
    def _reconstruct_down_kernel(
            z, kernel, output,
            z_stride_b, z_stride_c, z_stride_f, z_stride_h, z_stride_w,
            kernel_stride_i, kernel_stride_h, kernel_stride_w,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            input_h: tl.constexpr, input_w: tl.constexpr,
            output_h: tl.constexpr, output_w: tl.constexpr,
            channels: tl.constexpr, frequencies: tl.constexpr,
            block_x: tl.constexpr):
        row = tl.program_id(0)
        batch = row // (channels * output_h)
        channel_row = row - batch * channels * output_h
        channel = channel_row // output_h
        output_y = channel_row - channel * output_h
        output_x_parity = tl.program_id(2)
        output_x = ((tl.program_id(1) * block_x + tl.arange(0, block_x)) * 2 +
                    output_x_parity)
        frequency = tl.arange(0, frequencies)
        accumulator = tl.zeros((block_x, frequencies), tl.float32)
        output_y_parity = output_y % 2
        for iy in tl.static_range(0, 2):
            ky = output_y_parity + 2 * iy
            input_y = (output_y - ky) // 2
            valid_y = (ky < 3) & (input_y >= 0) & (input_y < input_h)
            for ix in tl.static_range(0, 2):
                kx = output_x_parity + 2 * ix
                input_x = (output_x - kx) // 2
                valid_x = (kx < 3) & (input_x >= 0) & (input_x < input_w)
                valid = valid_y & valid_x & (output_x < output_w)
                z_offset = (batch * z_stride_b + channel * z_stride_c +
                            frequency[None, :] * z_stride_f + input_y * z_stride_h +
                            input_x[:, None] * z_stride_w)
                values = tl.load(z + z_offset, mask=valid[:, None], other=0.0).to(tl.float32)
                kernel_offset = ((channel * frequencies + frequency) * kernel_stride_i +
                                 ky * kernel_stride_h + kx * kernel_stride_w)
                coefficients = tl.load(kernel + kernel_offset).to(tl.float32)
                accumulator += values * coefficients[None, :]
        result = tl.sum(accumulator, axis=1)
        output_offset = (batch * output_stride_b + channel * output_stride_c +
                         output_y * output_stride_h + output_x * output_stride_w)
        tl.store(output + output_offset, result, mask=output_x < output_w)


    @triton.autotune(
        configs=[
            triton.Config({'block_size': 256}, num_warps=4),
            triton.Config({'block_size': 512}, num_warps=8),
            triton.Config({'block_size': 1024}, num_warps=8),
        ],
        key=['elements'],
    )
    @triton.jit
    def _enhance_delta_kernel(semantic, delta, enhanced, recovered, elements,
                              block_size: tl.constexpr):
        offset = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offset < elements
        semantic_value = tl.load(semantic + offset, mask=valid).to(tl.float16)
        delta_value = tl.load(delta + offset, mask=valid).to(tl.float16)
        enhanced_value = (semantic_value + delta_value).to(tl.float16)
        recovered_value = (enhanced_value - semantic_value).to(tl.float16)
        tl.store(enhanced + offset, enhanced_value, mask=valid)
        tl.store(recovered + offset, recovered_value, mask=valid)


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (4, 8)],
        key=['height', 'width', 'input_channels', 'hidden_channels', 'output_channels'],
    )
    @triton.jit
    def _pointwise_mlp_kernel(
            x, first_weight, first_bias, last_weight, last_bias, output,
            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
            first_stride_o, first_stride_i, last_stride_o, last_stride_i,
            output_stride_b, output_stride_c, output_stride_h, output_stride_w,
            total_rows, height: tl.constexpr, width: tl.constexpr,
            positions: tl.constexpr, input_channels: tl.constexpr,
            hidden_channels: tl.constexpr, output_channels: tl.constexpr,
            block_positions: tl.constexpr, block_input: tl.constexpr,
            block_hidden: tl.constexpr, block_output: tl.constexpr):
        row = tl.program_id(0) * block_positions + tl.arange(0, block_positions)
        spatial = row % positions
        batch = row // positions
        y = spatial // width
        x_position = spatial - y * width
        input_channel = tl.arange(0, block_input)
        hidden_channel = tl.arange(0, block_hidden)
        output_channel = tl.arange(0, block_output)
        input_offset = (batch[:, None] * x_stride_b +
                        input_channel[None, :] * x_stride_c +
                        y[:, None] * x_stride_h + x_position[:, None] * x_stride_w)
        inputs = tl.load(x + input_offset,
                         mask=(row[:, None] < total_rows) &
                              (input_channel[None, :] < input_channels),
                         other=0.0).to(tl.float16)
        first_offset = (input_channel[:, None] * first_stride_i +
                        hidden_channel[None, :] * first_stride_o)
        first_values = tl.load(
            first_weight + first_offset,
            mask=(input_channel[:, None] < input_channels) &
                 (hidden_channel[None, :] < hidden_channels),
            other=0.0,
        ).to(tl.float16)
        hidden = tl.dot(inputs, first_values)
        hidden += tl.load(first_bias + hidden_channel[None, :],
                          mask=hidden_channel[None, :] < hidden_channels,
                          other=0.0).to(tl.float32)
        hidden = hidden.to(tl.float16).to(tl.float32)
        hidden = (hidden / (1.0 + tl.exp(-hidden))).to(tl.float16)
        last_offset = (hidden_channel[:, None] * last_stride_i +
                       output_channel[None, :] * last_stride_o)
        last_values = tl.load(
            last_weight + last_offset,
            mask=(hidden_channel[:, None] < hidden_channels) &
                 (output_channel[None, :] < output_channels),
            other=0.0,
        ).to(tl.float16)
        result = tl.dot(hidden, last_values)
        result += tl.load(last_bias + output_channel[None, :],
                          mask=output_channel[None, :] < output_channels,
                          other=0.0).to(tl.float32)
        output_offset = (batch[:, None] * output_stride_b +
                         output_channel[None, :] * output_stride_c +
                         y[:, None] * output_stride_h + x_position[:, None] * output_stride_w)
        tl.store(output + output_offset, result,
                 mask=(row[:, None] < total_rows) &
                      (output_channel[None, :] < output_channels))


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['output_h', 'output_w'],
    )
    @triton.jit
    def _dct_energy_kernel(
            x, dct_matrix, z, energy,
            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
            z_stride_b, z_stride_c, z_stride_f, z_stride_h, z_stride_w,
            energy_stride_b, energy_stride_c, energy_stride_h, energy_stride_w,
            output_h: tl.constexpr, output_w: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (output_h * output_w)
        spatial = position - batch * output_h * output_w
        output_y = spatial // output_w
        output_x = spatial - output_y * output_w
        channel = tl.arange(0, 32)[:, None]
        frequency = tl.arange(0, 16)[None, :]
        frequency_y = frequency // 4
        frequency_x = frequency - frequency_y * 4
        transformed = tl.zeros((32, 16), tl.float32)
        for iy in tl.static_range(0, 4):
            coefficient_y = tl.load(dct_matrix + frequency_y * 4 + iy).to(tl.float32)
            for ix in tl.static_range(0, 4):
                coefficient_x = tl.load(dct_matrix + frequency_x * 4 + ix).to(tl.float32)
                input_offset = (batch * x_stride_b + channel * x_stride_c +
                                (output_y * 4 + iy) * x_stride_h +
                                (output_x * 4 + ix) * x_stride_w)
                value = tl.load(x + input_offset).to(tl.float32)
                transformed += value * coefficient_y * coefficient_x
        z_offset = (batch * z_stride_b + channel * z_stride_c + frequency * z_stride_f +
                    output_y * z_stride_h + output_x * z_stride_w)
        tl.store(z + z_offset, transformed)
        grouped = tl.reshape(transformed, (4, 8, 16))
        power = tl.sum(grouped * grouped, axis=1) * 0.125
        power = tl.log(1.0 + power)
        energy_channel = tl.arange(0, 64)
        energy_offset = (batch * energy_stride_b + energy_channel * energy_stride_c +
                         output_y * energy_stride_h + output_x * energy_stride_w)
        tl.store(energy + energy_offset, tl.reshape(power, (64,)))


    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
        key=['height', 'width'],
    )
    @triton.jit
    def _statistics_descriptor_kernel(
            p, q, alpha, mask, semantic, tau, descriptor, rho,
            p_stride_b, p_stride_c, p_stride_h, p_stride_w,
            q_stride_b, q_stride_h, q_stride_w,
            alpha_stride_b, alpha_stride_c, alpha_stride_h, alpha_stride_w,
            mask_stride_b, mask_stride_h, mask_stride_w,
            semantic_stride_b, semantic_stride_c, semantic_stride_h, semantic_stride_w,
            descriptor_stride_b, descriptor_stride_c, descriptor_stride_h, descriptor_stride_w,
            rho_stride_b, rho_stride_h, rho_stride_w,
            height: tl.constexpr, width: tl.constexpr,
            channels: tl.constexpr, block_channels: tl.constexpr,
            eps: tl.constexpr):
        position = tl.program_id(0)
        batch = position // (height * width)
        spatial = position - batch * height * width
        y = spatial // width
        x = spatial - y * width
        channel = tl.arange(0, block_channels)
        channel_mask = channel < channels

        denominator3 = 0.0
        denominator5 = 0.0
        count3 = 0.0
        count5 = 0.0
        sum3 = tl.zeros((block_channels,), tl.float32)
        sum5 = tl.zeros((block_channels,), tl.float32)
        square3 = tl.zeros((block_channels,), tl.float32)
        square5 = tl.zeros((block_channels,), tl.float32)
        for oy in tl.static_range(-2, 3):
            for ox in tl.static_range(-2, 3):
                if oy != 0 or ox != 0:
                    sy, sx = y + oy, x + ox
                    inside = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
                    mask_offset = batch * mask_stride_b + sy * mask_stride_h + sx * mask_stride_w
                    valid = tl.load(mask + mask_offset, mask=inside, other=0.0).to(tl.float32)
                    q_offset = batch * q_stride_b + sy * q_stride_h + sx * q_stride_w
                    probability = tl.load(q + q_offset, mask=inside, other=0.0).to(tl.float32)
                    weight = valid * (1.0 - probability)
                    p_offset = (batch * p_stride_b + channel * p_stride_c +
                                sy * p_stride_h + sx * p_stride_w)
                    value = tl.load(p + p_offset, mask=inside & channel_mask,
                                    other=0.0).to(tl.float32)
                    denominator5 += weight
                    count5 += valid
                    sum5 += weight * value
                    square5 += weight * value * value
                    if oy >= -1:
                        if oy <= 1:
                            if ox >= -1:
                                if ox <= 1:
                                    denominator3 += weight
                                    count3 += valid
                                    sum3 += weight * value
                                    square3 += weight * value * value

        mean3 = sum3 / tl.maximum(denominator3, eps)
        mean5 = sum5 / tl.maximum(denominator5, eps)
        second3 = square3 / tl.maximum(denominator3, eps)
        second5 = square5 / tl.maximum(denominator5, eps)
        alpha_base = batch * alpha_stride_b + y * alpha_stride_h + x * alpha_stride_w
        alpha3 = tl.load(alpha + alpha_base).to(tl.float32)
        alpha5 = tl.load(alpha + alpha_base + alpha_stride_c).to(tl.float32)
        mean = alpha3 * mean3 + alpha5 * mean5
        second = alpha3 * second3 + alpha5 * second5
        sigma = tl.sqrt(tl.maximum(second - mean * mean, 0.0) + eps)
        support3 = denominator3 / tl.maximum(count3, 1.0)
        support5 = denominator5 / tl.maximum(count5, 1.0)
        support = alpha3 * support3 + alpha5 * support5
        raw_tau = tl.load(tau).to(tl.float32)
        positive_tau = tl.log(1.0 + tl.exp(raw_tau))
        reliability = support / (support + positive_tau)
        reliability = tl.where((count3 > 0.0) | (count5 > 0.0), reliability, 0.0)

        current = tl.load(
            p + batch * p_stride_b + channel * p_stride_c + y * p_stride_h + x * p_stride_w,
            mask=channel_mask, other=0.0,
        ).to(tl.float32)
        residual = (current - mean) / (sigma + eps)
        descriptor_base = batch * descriptor_stride_b + y * descriptor_stride_h + x * descriptor_stride_w
        tl.store(descriptor + descriptor_base + channel * descriptor_stride_c,
                 current, mask=channel_mask)
        tl.store(descriptor + descriptor_base + (channel + channels) * descriptor_stride_c,
                 mean, mask=channel_mask)
        tl.store(descriptor + descriptor_base + (channel + 2 * channels) * descriptor_stride_c,
                 sigma, mask=channel_mask)
        tl.store(descriptor + descriptor_base + (channel + 3 * channels) * descriptor_stride_c,
                 residual, mask=channel_mask)
        semantic_channel = tl.arange(0, 16)
        semantic_value = tl.load(
            semantic + batch * semantic_stride_b + semantic_channel * semantic_stride_c +
            y * semantic_stride_h + x * semantic_stride_w,
        )
        tl.store(descriptor + descriptor_base + (256 + semantic_channel) * descriptor_stride_c,
                 semantic_value)
        tl.store(rho + batch * rho_stride_b + y * rho_stride_h + x * rho_stride_w,
                 reliability)


def fused_statistics_descriptor(p, q, alpha, mask, semantic, tau, eps):
    """Fuse BRSA neighborhood statistics and descriptor construction for inference."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, height, width = p.shape
    if channels != 64 or semantic.shape[1] != 16:
        raise ValueError("The fused BRSA kernel expects 64 frequency and 16 semantic channels")
    memory_format = (torch.channels_last
                     if semantic.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    descriptor = torch.empty((batch, 272, height, width), device=p.device,
                             dtype=semantic.dtype, memory_format=memory_format)
    rho = torch.empty((batch, 1, height, width), device=p.device, dtype=torch.float32,
                      memory_format=memory_format)
    block_channels = triton.next_power_of_2(channels)
    _statistics_descriptor_kernel[(batch * height * width,)](
        p, q, alpha, mask, semantic, tau, descriptor, rho,
        *p.stride(), q.stride(0), q.stride(2), q.stride(3), *alpha.stride(),
        mask.stride(0), mask.stride(2), mask.stride(3), *semantic.stride(),
        *descriptor.stride(), rho.stride(0), rho.stride(2), rho.stride(3),
        height, width, channels, block_channels, eps,
    )
    return descriptor, rho


def fused_reference_input(frequency, semantic):
    """Cast and concatenate the BRSA frequency and semantic descriptors once."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, height, width = frequency.shape
    if (frequency.dtype != torch.float32 or channels != 64 or
            semantic.dtype != torch.float16 or semantic.shape != (batch, 16, height, width)):
        raise ValueError("The fused BRSA reference input expects FP32 C64 and FP16 C16 inputs")
    memory_format = (torch.channels_last
                     if semantic.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty((batch, 80, height, width), device=frequency.device,
                         dtype=semantic.dtype, memory_format=memory_format)
    _reference_input_kernel[(batch * height * width,)](
        frequency, semantic, output, *frequency.stride(), *semantic.stride(),
        *output.stride(), height, width, num_warps=4,
    )
    return output


def fused_control_activation(control):
    """Fuse BRSA occupancy sigmoid and two-way scale softmax in FP32."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, height, width = control.shape
    if control.dtype != torch.float16 or channels != 3:
        raise ValueError("The fused BRSA control activation expects a 3-channel FP16 input")
    memory_format = (torch.channels_last
                     if control.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    activated = torch.empty((batch, 3, height, width), device=control.device,
                            dtype=torch.float32, memory_format=memory_format)
    _control_activation_kernel[(batch * height * width,)](
        control, activated, *control.stride(), *activated.stride(), height, width,
        num_warps=1,
    )
    return activated[:, :1], activated[:, 1:]


def fused_silu_avg_pool2d(x):
    """Fuse SiLU with the fixed 2x2/stride-2 BRSA semantic pooling."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, input_h, input_w = x.shape
    if x.dtype != torch.float16 or channels != 16 or input_h % 2 or input_w % 2:
        raise ValueError("The fused BRSA semantic pool expects even FP16 16-channel inputs")
    output_h, output_w = input_h // 2, input_w // 2
    memory_format = (torch.channels_last
                     if x.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty((batch, channels, output_h, output_w), device=x.device,
                         dtype=x.dtype, memory_format=memory_format)
    _silu_avg_pool2d_kernel[(batch * output_h * output_w,)](
        x, output, *x.stride(), *output.stride(),
        input_h, input_w, output_h, output_w, channels,
    )
    return output


def fused_dct_energy(x, dct_matrix):
    """Compute the BRSA block DCT and grouped log energy in one kernel."""
    batch, channels, height, width = x.shape
    if channels != 32 or height % 4 or width % 4:
        raise ValueError("The fused BRSA DCT expects 32 channels and 4-aligned spatial axes")
    output_h, output_w = height // 4, width // 4
    z = torch.empty((batch, 32, 16, output_h, output_w), device=x.device,
                    dtype=torch.float32)
    memory_format = (torch.channels_last
                     if x.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    energy = torch.empty((batch, 64, output_h, output_w), device=x.device,
                         dtype=torch.float32, memory_format=memory_format)
    _dct_energy_kernel[(batch * output_h * output_w,)](
        x, dct_matrix, z, energy, *x.stride(), *z.stride(), *energy.stride(),
        output_h, output_w,
    )
    return z, energy


def fused_pointwise_mlp(x, first, last):
    """Fuse two pointwise convolutions and their intermediate SiLU activation."""
    batch, input_channels, height, width = x.shape
    hidden_channels, output_channels = first.out_channels, last.out_channels
    memory_format = (torch.channels_last
                     if x.is_contiguous(memory_format=torch.channels_last)
                     else torch.contiguous_format)
    output = torch.empty((batch, output_channels, height, width), device=x.device,
                         dtype=x.dtype, memory_format=memory_format)
    block_positions = 16
    total_rows = batch * height * width
    _pointwise_mlp_kernel[(triton.cdiv(total_rows, block_positions),)](
        x, first.weight, first.bias, last.weight, last.bias, output,
        *x.stride(), first.weight.stride(0), first.weight.stride(1),
        last.weight.stride(0), last.weight.stride(1), *output.stride(),
        total_rows, height, width, height * width, input_channels,
        hidden_channels, output_channels, block_positions,
        triton.next_power_of_2(input_channels), triton.next_power_of_2(hidden_channels),
        max(16, triton.next_power_of_2(output_channels)),
    )
    return output


def fused_enhance_delta(semantic, delta):
    """Create the enhanced P3 feature and its rounded residual in one pass."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if semantic.dtype != torch.float16 or semantic.stride() != delta.stride():
        raise ValueError("The fused enhance kernel expects layout-matched FP16 tensors")
    enhanced = torch.empty_like(semantic, memory_format=torch.preserve_format)
    recovered = torch.empty_like(semantic, memory_format=torch.preserve_format)
    elements = semantic.numel()
    grid = lambda meta: (triton.cdiv(elements, meta['block_size']),)
    _enhance_delta_kernel[grid](semantic, delta, enhanced, recovered, elements)
    return enhanced, recovered


def fused_reconstruct_down(z, kernel, output_shape):
    """Specialized grouped transposed convolution for BRSA frequency reconstruction."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, frequencies, input_h, input_w = z.shape
    output_h, output_w = output_shape
    output = torch.empty((batch, channels, output_h, output_w), device=z.device,
                         dtype=z.dtype, memory_format=torch.channels_last)
    grid = lambda meta: (
        batch * channels * output_h,
        triton.cdiv((output_w + 1) // 2, meta['block_x']),
        2,
    )
    _reconstruct_down_kernel[grid](
        z, kernel, output, *z.stride(), kernel.stride(0), kernel.stride(2),
        kernel.stride(3), *output.stride(), input_h, input_w, output_h, output_w,
        channels, frequencies,
    )
    return output


def fused_modulate_frequency(z, modulation, rho):
    """Fuse BRSA tanh, FP16 casts, and two rounded frequency-domain products."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, channels, frequencies, height, width = z.shape
    if channels != 32 or frequencies != 16:
        raise ValueError("The fused BRSA modulation expects 32 channels and 16 frequencies")
    output = torch.empty((batch, channels, frequencies, height, width), device=z.device,
                         dtype=modulation.dtype)
    grid = lambda meta: (
        4 * frequencies,
        batch,
        triton.cdiv(height * width, meta['block_positions']),
    )
    _modulate_frequency_kernel[grid](
        z, modulation, rho, output, *z.stride(), *modulation.stride(),
        rho.stride(0), rho.stride(2), rho.stride(3), *output.stride(),
        height, width,
    )
    return output


def fused_brsa_available():
    return triton is not None

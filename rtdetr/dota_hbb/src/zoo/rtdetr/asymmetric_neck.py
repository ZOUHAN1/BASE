import math

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_act(c1, c2, kernel=1):
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(),
    )


def valid_cells(pixel_mask, stride):
    return -F.max_pool2d(-pixel_mask.float(), stride, stride) >= 1 - 1e-6


def masked_softmax(logits, valid, dim):
    valid = valid.bool()
    masked = logits.float().masked_fill(~valid, -1e30)
    weights = torch.softmax(masked, dim=dim) * valid
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def canonical_grid(h, w, source_h, source_w, anchor_stride, source_stride, device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij',
    )
    x = (x.float() + 0.5) * anchor_stride / source_stride - 0.5
    y = (y.float() + 0.5) * anchor_stride / source_stride - 0.5
    return torch.stack((x, y), -1)


def sample_points(values, coords):
    b, h, w, k, _ = coords.shape
    source_h, source_w = values.shape[-2:]
    scale = coords.new_tensor([source_w, source_h])
    grid = (2 * (coords + 0.5) / scale - 1).reshape(b, h, w * k, 2)
    sampled = F.grid_sample(
        values.float(), grid.float(), mode='bilinear',
        padding_mode='zeros', align_corners=False,
    )
    return sampled.reshape(b, values.shape[1], h, w, k).permute(0, 4, 1, 2, 3)


class BackgroundReferencedReading(nn.Module):
    def __init__(self, channels, anchor_stride, source_strides, value_channels=64):
        super().__init__()
        self.anchor_stride = anchor_stride
        self.source_strides = tuple(source_strides)
        self.query = conv_bn_act(channels, 32)
        self.keys = nn.ModuleList([conv_bn_act(channels, 32) for _ in source_strides])
        self.values = nn.ModuleList([
            nn.Conv2d(channels, value_channels, 1, bias=False) for _ in source_strides
        ])
        self.controllers = nn.ModuleList([
            nn.Sequential(conv_bn_act(64, 64, 3), nn.Conv2d(64, 52, 1, bias=True))
            for _ in source_strides
        ])
        self.scale_logits = nn.Conv2d(32, len(source_strides), 1) \
            if len(source_strides) > 1 else None
        self.output = nn.Conv2d(value_channels, channels, 1, bias=False)

        support = [(x, y) for y in (-0.5, 0.0, 0.5) for x in (-0.5, 0.0, 0.5)]
        reference = [(x, y) for y in (-2.0, 0.0, 2.0)
                     for x in (-2.0, 0.0, 2.0) if x or y]
        self.register_buffer('template', torch.tensor(support + reference).float())
        self.register_buffer(
            'offset_scale',
            torch.tensor([0.25] * 9 + [0.5] * 8).view(1, 1, 1, 17, 1),
        )
        for controller in self.controllers:
            nn.init.zeros_(controller[-1].weight)
            nn.init.zeros_(controller[-1].bias)
        if self.scale_logits is not None:
            nn.init.zeros_(self.scale_logits.weight)
            nn.init.zeros_(self.scale_logits.bias)

    def forward(self, anchor, sources, masks):
        q = self.query(anchor)
        b, _, h, w = anchor.shape
        outputs, source_valid = [], []
        for source, mask, key_proj, value_proj, controller, source_stride in zip(
                sources, masks, self.keys, self.values, self.controllers, self.source_strides):
            center = canonical_grid(
                h, w, *source.shape[-2:], self.anchor_stride, source_stride, source.device
            )
            center = center[None, :, :, None].expand(b, -1, -1, -1, -1)
            key, value = key_proj(source), value_proj(source)
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                center_key = sample_points(key, center)[:, 0]
            control = controller(torch.cat((q, center_key.to(q.dtype)), 1))
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                ctrl = control.float()
                offset = ctrl[:, :34].reshape(b, 17, 2, h, w).permute(0, 3, 4, 1, 2)
                coords = center + self.template.float()[None, None, None] \
                    + self.offset_scale.float() * offset.tanh()
                values_at_points = sample_points(value, coords)
                valid = sample_points(mask.float(), coords)[:, :, 0] >= 1 - 1e-6
                support_weight = masked_softmax(ctrl[:, 34:43], valid[:, :9], 1)
                reference_weight = masked_softmax(ctrl[:, 43:51], valid[:, 9:], 1)
                support = (values_at_points[:, :9] * support_weight[:, :, None]).sum(1)
                reference = (values_at_points[:, 9:] * reference_weight[:, :, None]).sum(1)
                has_support = valid[:, :9].any(1, keepdim=True)
                has_reference = valid[:, 9:].any(1, keepdim=True)
                contrast = (support - reference) * has_reference
                gate = ctrl[:, 51:52].tanh()
                outputs.append((support + gate * contrast) * has_support)
                source_valid.append(has_support)

        logits = self.scale_logits(q) if self.scale_logits is not None else q[:, :1] * 0
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            weights = masked_softmax(logits, torch.cat(source_valid, 1), 1)
            merged = (torch.stack(outputs, 1) * weights[:, :, None]).sum(1)
        return self.output(merged.to(anchor.dtype))


class OriginalBRSA(nn.Module):
    def __init__(self, shallow_channels=256, channels=256):
        super().__init__()
        self.project = conv_bn_act(shallow_channels, 32)
        self.semantic = nn.Sequential(conv_bn_act(channels, 16), nn.AvgPool2d(2, 2))
        self.reference = nn.Sequential(
            nn.Conv2d(80, 16, 1), nn.SiLU(), nn.Conv2d(16, 3, 1)
        )
        self.modulation = nn.Sequential(
            nn.Conv2d(272, 32, 1), nn.SiLU(), nn.Conv2d(32, 64, 1)
        )
        self.down = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False)
        self.output = nn.Conv2d(32, channels, 1, bias=False)
        self.tau = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        self.eps = 1e-6

        u, x = torch.meshgrid(torch.arange(4).float(), torch.arange(4).float(), indexing='ij')
        dct = torch.cos(math.pi * (2 * x + 1) * u / 8) * math.sqrt(0.5)
        dct[0] = 0.5
        self.register_buffer('dct_matrix', dct)
        for radius in (3, 5):
            kernel = torch.ones(1, 1, radius, radius)
            kernel[0, 0, radius // 2, radius // 2] = 0
            self.register_buffer('kernel%d' % radius, kernel)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def dct(self, x):
        b, c, h, w = x.shape
        assert h % 4 == 0 and w % 4 == 0
        blocks = x.float().reshape(b, c, h // 4, 4, w // 4, 4)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5)
        matrix = self.dct_matrix.float()
        coeff = torch.matmul(torch.matmul(matrix, blocks), matrix.T)
        return coeff.permute(0, 1, 4, 5, 2, 3).reshape(b, c, 16, h // 4, w // 4)

    def idct(self, coeff):
        b, c, _, h, w = coeff.shape
        blocks = coeff.reshape(b, c, 4, 4, h, w).permute(0, 1, 4, 5, 2, 3)
        matrix = self.dct_matrix.float()
        output = torch.matmul(torch.matmul(matrix.T, blocks), matrix)
        return output.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h * 4, w * 4)

    def neighbor_sum(self, x, radius):
        kernel = getattr(self, 'kernel%d' % radius).float().expand(x.shape[1], 1, -1, -1)
        return F.conv2d(x, kernel, padding=radius // 2, groups=x.shape[1])

    def statistics(self, power, foreground, alpha, mask):
        weight = mask.float() * (1 - foreground.detach())
        means, moments, supports, nonempty = [], [], [], []
        for radius in (3, 5):
            denominator = self.neighbor_sum(weight, radius)
            count = self.neighbor_sum(mask.float(), radius)
            means.append(self.neighbor_sum(weight * power, radius) /
                         denominator.clamp_min(self.eps))
            moments.append(self.neighbor_sum(weight * power.square(), radius) /
                           denominator.clamp_min(self.eps))
            supports.append(denominator / count.clamp_min(1))
            nonempty.append(count > 0)
        mean = alpha[:, 0:1] * means[0] + alpha[:, 1:2] * means[1]
        moment = alpha[:, 0:1] * moments[0] + alpha[:, 1:2] * moments[1]
        sigma = ((moment - mean.square()).clamp_min(0) + self.eps).sqrt()
        support = alpha[:, 0:1] * supports[0] + alpha[:, 1:2] * supports[1]
        rho = support / (support + F.softplus(self.tau.float()))
        rho = rho * (nonempty[0] | nonempty[1])
        return mean, sigma, rho

    @staticmethod
    def occupancy_target(targets, shape, input_shape, device):
        batch, _, h, w = shape
        input_h, input_w = input_shape
        result = torch.zeros(batch, h, w, device=device, dtype=torch.bool)
        for batch_idx, target in enumerate(targets or []):
            boxes = target.get('boxes')
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device).float()
            center = boxes[:, :2] * boxes.new_tensor([input_w, input_h])
            size = boxes[:, 2:] * boxes.new_tensor([input_w, input_h])
            low = torch.floor((center - size / 2) / 16).long()
            high = torch.ceil((center + size / 2) / 16).long()
            low[:, 0].clamp_(0, w)
            low[:, 1].clamp_(0, h)
            high[:, 0].clamp_(0, w)
            high[:, 1].clamp_(0, h)
            for lo, hi in zip(low, high):
                if hi[0] > lo[0] and hi[1] > lo[1]:
                    result[batch_idx, lo[1]:hi[1], lo[0]:hi[0]] = True
        return result[:, None]

    def occupancy_loss(self, logits, mask, targets, input_shape):
        target = self.occupancy_target(targets, logits.shape, input_shape, logits.device)
        valid = mask
        if targets and all('aux_valid_mask' in item for item in targets):
            aux_valid = torch.stack([
                item['aux_valid_mask'].to(logits.device).float() for item in targets
            ])
            if aux_valid.ndim == 3:
                aux_valid = aux_valid[:, None]
            valid = valid & valid_cells(aux_valid, 16)
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
        groups = [loss[valid & target], loss[valid & ~target]]
        means = [group.mean() for group in groups if group.numel()]
        return torch.stack(means).mean() if means else logits.sum() * 0

    def forward(self, shallow, semantic, pixel_valid, targets=None):
        projected = self.project(shallow)
        semantic_descriptor = self.semantic(semantic)
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            coeff = self.dct(projected)
            b, _, _, h, w = coeff.shape
            power = coeff.reshape(b, 4, 8, 16, h, w).square().mean(2).log1p()
            power = power.reshape(b, 64, h, w)
            valid = valid_cells(pixel_valid, 16)
        control = self.reference(torch.cat((power.to(semantic_descriptor.dtype), semantic_descriptor), 1))
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            logits = control[:, :1].float()
            foreground = logits.sigmoid()
            alpha = control[:, 1:].float().softmax(1)
            mean, sigma, rho = self.statistics(power, foreground, alpha, valid)
            residual = (power - mean) / (sigma + self.eps)
            descriptor = torch.cat((power, mean, sigma, residual, semantic_descriptor.float()), 1)
        modulation = self.modulation(descriptor.to(semantic_descriptor.dtype))
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            modulation = modulation.float().reshape(b, 4, 1, 16, h, w).tanh()
            delta_coeff = coeff.reshape(b, 4, 8, 16, h, w) * modulation * rho[:, None, None]
            reconstructed = self.idct(delta_coeff.reshape(b, 32, 16, h, w))
        delta = self.output(self.down(reconstructed.to(projected.dtype)))
        enhanced = semantic + delta.to(semantic.dtype)
        aux_loss = None
        if targets is not None:
            aux_loss = self.occupancy_loss(logits, valid, targets, pixel_valid.shape[-2:])
        return enhanced, aux_loss


def pack_candidates(x, factor, target_shape=None):
    b, c, h, w = x.shape
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    out_h, out_w = x.shape[-2] // factor, x.shape[-1] // factor
    packed = x.reshape(b, c, out_h, factor, out_w, factor)
    packed = packed.permute(0, 3, 5, 1, 2, 4)
    packed = packed.reshape(b, factor * factor, c, out_h, out_w)
    if target_shape is not None:
        packed = packed[..., :target_shape[0], :target_shape[1]]
    return packed


class AdaptiveResidualRelay(nn.Module):
    def __init__(self, channels, factor, refine):
        super().__init__()
        self.factor = factor
        self.query = conv_bn_act(channels, 32)
        self.key = nn.Conv2d(channels, 32, 1, bias=False)
        self.score = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1)
        )
        self.position_bias = nn.Parameter(torch.zeros(factor * factor))
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1)
        )
        self.refine = refine
        for network in (self.score, self.gate):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def forward(self, base, correction, valid):
        query, key = self.query(base), self.key(correction)
        keys = pack_candidates(key, self.factor, base.shape[-2:])
        candidates = pack_candidates(correction, self.factor, base.shape[-2:])
        mask = pack_candidates(valid.float(), self.factor, base.shape[-2:])[:, :, 0].bool()
        b, count, _, h, w = keys.shape
        query_keys = torch.cat((query[:, None].expand(-1, count, -1, -1, -1), keys), 2)
        scores = self.score(query_keys.reshape(b * count, 64, h, w)).reshape(b, count, h, w)
        with torch.autocast(device_type=base.device.type, enabled=False):
            weights = masked_softmax(
                scores.float() + self.position_bias.float()[None, :, None, None], mask, 1
            )
            aggregate = (weights[:, :, None] * candidates.float()).sum(1)
            mean_key = (weights[:, :, None] * keys.float()).sum(1)
        value = self.value(aggregate.to(base.dtype))
        gate = self.gate(torch.cat((query, mean_key.to(query.dtype)), 1)).sigmoid()
        return self.refine(base + gate * value)

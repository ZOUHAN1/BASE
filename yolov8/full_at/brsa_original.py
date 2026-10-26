import math
import torch
from torch import nn
import torch.nn.functional as F
from .background_referenced_reading import conv_bn_act
from .geometry import valid_cells


class OriginalBRSA(nn.Module):
    def __init__(self, shallow_channels=256, semantic_channels=256):
        super().__init__()
        self.project = conv_bn_act(shallow_channels, 32)
        self.semantic = nn.Sequential(conv_bn_act(semantic_channels, 16), nn.AvgPool2d(2, 2))
        self.reference = nn.Sequential(nn.Conv2d(80, 16, 1), nn.SiLU(), nn.Conv2d(16, 3, 1))
        self.modulation = nn.Sequential(nn.Conv2d(272, 32, 1), nn.SiLU(), nn.Conv2d(32, 64, 1))
        self.down = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False)
        self.output = nn.Conv2d(32, semantic_channels, 1, bias=False)
        self.tau = nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.eps = 1e-6
        u, x = torch.meshgrid(torch.arange(4).float(), torch.arange(4).float(), indexing='ij')
        a = torch.cos(math.pi * (2 * x + 1) * u / 8) * math.sqrt(.5)
        a[0] = .5
        self.register_buffer('dct_matrix', a)
        for r in (3, 5):
            kernel = torch.ones(1, 1, r, r)
            kernel[0, 0, r // 2, r // 2] = 0
            self.register_buffer('kernel%d' % r, kernel)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def dct(self, x):
        b, c, h, w = x.shape
        assert h % 4 == w % 4 == 0
        blocks = x.float().reshape(b, c, h // 4, 4, w // 4, 4).permute(0, 1, 2, 4, 3, 5)
        a = self.dct_matrix.float()
        z = torch.matmul(torch.matmul(a, blocks), a.T)
        return z.permute(0, 1, 4, 5, 2, 3).reshape(b, c, 16, h // 4, w // 4)

    def idct(self, z):
        b, c, _, h, w = z.shape
        blocks = z.reshape(b, c, 4, 4, h, w).permute(0, 1, 4, 5, 2, 3)
        a = self.dct_matrix.float()
        out = torch.matmul(torch.matmul(a.T, blocks), a)
        return out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h * 4, w * 4)

    def neighbor_sum(self, x, radius):
        kernel = getattr(self, 'kernel%d' % radius).float().expand(x.shape[1], 1, -1, -1)
        return F.conv2d(x, kernel, padding=radius // 2, groups=x.shape[1])

    def statistics(self, p, q, alpha, mask):
        weight = mask.float() * (1 - q.detach())
        mus, nus, supports, nonempty = [], [], [], []
        for radius in (3, 5):
            denominator = self.neighbor_sum(weight, radius)
            count = self.neighbor_sum(mask.float(), radius)
            mus.append(self.neighbor_sum(weight * p, radius) / denominator.clamp_min(self.eps))
            nus.append(self.neighbor_sum(weight * p.square(), radius) / denominator.clamp_min(self.eps))
            supports.append(denominator / count.clamp_min(1))
            nonempty.append(count > 0)
        mu = alpha[:, 0:1] * mus[0] + alpha[:, 1:2] * mus[1]
        nu = alpha[:, 0:1] * nus[0] + alpha[:, 1:2] * nus[1]
        sigma = ((nu - mu.square()).clamp_min(0) + self.eps).sqrt()
        support = alpha[:, 0:1] * supports[0] + alpha[:, 1:2] * supports[1]
        rho = support / (support + F.softplus(self.tau.float()))
        rho = rho * (nonempty[0] | nonempty[1])
        return mu, sigma, rho

    def forward(self, shallow, semantic, pixel_valid, return_aux=False, diagnostics=False):
        u, s = self.project(shallow), self.semantic(semantic)
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            z = self.dct(u)
            b, c, _, h, w = z.shape
            p = z.reshape(b, 4, 8, 16, h, w).square().mean(2).log1p().reshape(b, 64, h, w)
            mask = valid_cells(pixel_valid, 16)
        control = self.reference(torch.cat((p.to(s.dtype), s), 1))
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            logits = control[:, :1].float()
            q = logits.sigmoid()
            alpha = control[:, 1:].float().softmax(1)
            mu, sigma, rho = self.statistics(p, q, alpha, mask)
            residual = (p - mu) / (sigma + self.eps)
            descriptor = torch.cat((p, mu, sigma, residual, s.float()), 1)
        modulation = self.modulation(descriptor.to(s.dtype))
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            mod = modulation.float().reshape(b, 4, 1, 16, h, w).tanh()
            delta_z = z.reshape(b, 4, 8, 16, h, w) * mod * rho[:, None, None]
            reconstructed = self.idct(delta_z.reshape(b, 32, 16, h, w))
        delta = self.output(self.down(reconstructed.to(u.dtype)))
        enhanced = semantic + delta.to(semantic.dtype)
        aux = dict(logits=logits, mask=mask) if return_aux else None
        if diagnostics:
            aux = {} if aux is None else aux
            aux['diagnostics'] = {k: v.detach().cpu() for k, v in dict(q=q, rho=rho, delta=delta).items()}
        return enhanced, aux

    @staticmethod
    def occupancy_target(boxes, batch_idx, shape, input_shape):
        b, _, h, w = shape
        ih, iw = input_shape
        device = boxes.device
        y = torch.zeros(b, h, w, device=device, dtype=torch.bool)
        if boxes.numel():
            # Union of strictly positive box-cell intersections via a 2D difference array.
            xy = boxes[:, :2] * boxes.new_tensor([iw, ih])
            wh = boxes[:, 2:] * boxes.new_tensor([iw, ih])
            lo = torch.floor((xy - wh / 2) / 16).long()
            hi = torch.ceil((xy + wh / 2) / 16).long()
            lo[:, 0].clamp_(0, w); lo[:, 1].clamp_(0, h)
            hi[:, 0].clamp_(0, w); hi[:, 1].clamp_(0, h)
            good = (hi > lo).all(1) & (wh > 0).all(1)
            ids, lo, hi = batch_idx.long()[good], lo[good], hi[good]
            diff = torch.zeros(b * (h + 1) * (w + 1), device=device)
            for xx, yy, sign in ((lo[:, 0], lo[:, 1], 1), (hi[:, 0], lo[:, 1], -1),
                                 (lo[:, 0], hi[:, 1], -1), (hi[:, 0], hi[:, 1], 1)):
                index = ids * (h + 1) * (w + 1) + yy * (w + 1) + xx
                diff.scatter_add_(0, index, torch.full_like(index, sign, dtype=torch.float32))
            y = diff.reshape(b, h + 1, w + 1).cumsum(1).cumsum(2)[:, :h, :w] > 0
        return y[:, None]

    def occupancy_loss(self, aux, batch):
        a, valid = aux['logits'], aux['mask']
        if 'aux_valid_mask' in batch:
            valid = valid & valid_cells(batch['aux_valid_mask'], 16)
        target = self.occupancy_target(batch['bboxes'].to(a.device), batch['batch_idx'].to(a.device),
                                       a.shape, batch['img'].shape[-2:])
        loss = F.binary_cross_entropy_with_logits(a, target.float(), reduction='none')
        groups = [loss[valid & target], loss[valid & ~target]]
        nonempty = [x.mean() for x in groups if x.numel()]
        return torch.stack(nonempty).mean() if nonempty else a.sum() * 0

import math
import torch
from torch import nn
import torch.nn.functional as F
from .background_referenced_reading import conv_bn_act
from .fused_brsa import (fused_brsa_available, fused_control_activation,
                         fused_dct_energy, fused_enhance_delta, fused_modulate_frequency,
                         fused_pointwise_mlp,
                         fused_reference_input,
                         fused_reconstruct_down,
                         fused_silu_avg_pool2d, fused_statistics_descriptor)
from .geometry import valid_cells


class OriginalBRSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = conv_bn_act(256, 32)
        self.semantic = nn.Sequential(conv_bn_act(256, 16), nn.AvgPool2d(2, 2))
        self.reference = nn.Sequential(nn.Conv2d(80, 16, 1), nn.SiLU(), nn.Conv2d(16, 3, 1))
        self.modulation = nn.Sequential(nn.Conv2d(272, 32, 1), nn.SiLU(), nn.Conv2d(32, 64, 1))
        self.down = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False)
        self.output = nn.Conv2d(32, 256, 1, bias=False)
        self.fast_reconstruct = True
        self.fast_statistics = True
        self.fast_dct = True
        self.fast_mlp = True
        self.fast_enhance = True
        self.fast_reconstruct_down = True
        self.fast_modulated_reconstruct = True
        self.fast_semantic_pool = True
        self.fast_control_activation = True
        self.fast_reference_input = True
        self.tau = nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.eps = 1e-6
        u, x = torch.meshgrid(torch.arange(4).float(), torch.arange(4).float(), indexing='ij')
        a = torch.cos(math.pi * (2 * x + 1) * u / 8) * math.sqrt(.5)
        a[0] = .5
        self.register_buffer('dct_matrix', a)
        dct_kernel = torch.einsum('ui,vj->uvij', a, a).reshape(16, 1, 4, 4)
        self.register_buffer('_dct_kernel', dct_kernel.repeat(32, 1, 1, 1), persistent=False)
        self._reconstruct_down_cache = {}
        for r in (3, 5):
            kernel = torch.ones(1, 1, r, r)
            kernel[0, 0, r // 2, r // 2] = 0
            self.register_buffer('kernel%d' % r, kernel)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _apply(self, fn):
        self._reconstruct_down_cache.clear()
        return super()._apply(fn)

    def dct(self, x):
        b, c, h, w = x.shape
        assert h % 4 == w % 4 == 0
        assert c == 32
        z = F.conv2d(x.float(), self._dct_kernel.float(), stride=4, groups=c)
        return z.reshape(b, c, 16, h // 4, w // 4)

    def idct(self, z):
        b, c, _, h, w = z.shape
        assert c == 32
        return F.conv_transpose2d(z.reshape(b, c * 16, h, w).float(),
                                  self._dct_kernel.float(), stride=4, groups=c)

    def combined_reconstruct_down_kernel(self, dtype):
        try:
            version = self.down.weight._version
        except RuntimeError:
            # Parameters created under inference_mode do not expose a version
            # counter. They are immutable for this deployment-only fast path.
            version = None
        key = (self.down.weight.device, dtype, version)
        cached = self._reconstruct_down_cache.get(key)
        if cached is not None:
            return cached
        # Compose the fixed 4x4 IDCT and learned 3x3/stride-2 depthwise
        # convolution into one grouped transposed convolution. Its 101x101
        # boundary is cropped by reconstruct_down() to the original 100x100.
        basis = F.pad(self._dct_kernel[:, 0].float().unsqueeze(0), (1, 2, 1, 2))
        down = self.down.weight.float().repeat_interleave(16, 0)
        kernel = F.conv2d(basis, down, stride=2, groups=down.shape[0])
        kernel = kernel.squeeze(0).unsqueeze(1).to(dtype)
        self._reconstruct_down_cache.clear()
        self._reconstruct_down_cache[key] = kernel
        return kernel

    def reconstruct_down(self, z, output_shape, dtype):
        b, c, _, h, w = z.shape
        kernel = self.combined_reconstruct_down_kernel(dtype)
        if (self.fast_reconstruct_down and z.is_cuda and z.dtype == torch.float16 and
                fused_brsa_available()):
            return fused_reconstruct_down(z, kernel, output_shape)
        result = F.conv_transpose2d(z.reshape(b, c * 16, h, w).to(dtype), kernel,
                                    stride=2, groups=c)
        return result[..., :output_shape[0], :output_shape[1]]

    def neighbor_sum(self, x, radius):
        kernel = getattr(self, 'kernel%d' % radius).float().expand(x.shape[1], 1, -1, -1)
        return F.conv2d(x, kernel, padding=radius // 2, groups=x.shape[1])

    def statistics(self, p, q, alpha, mask):
        weight = mask.float() * (1 - q.detach())
        weighted_p = weight * p
        terms = torch.cat((weight, mask.float(), weighted_p, weighted_p * p), 1)
        mus, nus, supports, nonempty = [], [], [], []
        sums_by_radius = (self.neighbor_sum(terms, 3), self.neighbor_sum(terms, 5))
        for sums in sums_by_radius:
            denominator, count, sum_p, sum_p2 = sums.split((1, 1, 64, 64), 1)
            mus.append(sum_p / denominator.clamp_min(self.eps))
            nus.append(sum_p2 / denominator.clamp_min(self.eps))
            supports.append(denominator / count.clamp_min(1))
            nonempty.append(count > 0)
        mu = alpha[:, 0:1] * mus[0] + alpha[:, 1:2] * mus[1]
        nu = alpha[:, 0:1] * nus[0] + alpha[:, 1:2] * nus[1]
        sigma = ((nu - mu.square()).clamp_min(0) + self.eps).sqrt()
        support = alpha[:, 0:1] * supports[0] + alpha[:, 1:2] * supports[1]
        rho = support / (support + F.softplus(self.tau.float()))
        rho = rho * (nonempty[0] | nonempty[1])
        return mu, sigma, rho

    def forward(self, shallow, semantic, pixel_valid, return_aux=False, diagnostics=False,
                block_valid=None, return_recovered_delta=False, precomputed_project=None):
        u = self.project(shallow) if precomputed_project is None else precomputed_project
        semantic_projection = self.semantic[0]
        fused_semantic_pool = (
            self.fast_semantic_pool and not self.training and semantic.is_cuda and
            not torch.is_grad_enabled() and semantic.dtype == torch.float16 and
            isinstance(semantic_projection[1], nn.Identity) and
            fused_brsa_available())
        if fused_semantic_pool:
            s = fused_silu_avg_pool2d(semantic_projection[0](semantic))
        else:
            s = self.semantic(semantic)
        fast_reconstruct = (self.fast_reconstruct and not self.training and shallow.is_cuda and
                            not torch.is_grad_enabled() and
                            u.dtype in (torch.float16, torch.bfloat16, torch.float32))
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            if self.fast_dct and fast_reconstruct and fused_brsa_available():
                z, p = fused_dct_energy(u, self.dct_matrix)
                b, c, _, h, w = z.shape
            else:
                z = self.dct(u)
                b, c, _, h, w = z.shape
                p = z.reshape(b, 4, 8, 16, h, w).square().mean(2).log1p().reshape(b, 64, h, w)
            mask = valid_cells(pixel_valid, 16) if block_valid is None else block_valid
        fused_reference = (self.fast_reference_input and fast_reconstruct and
                           p.dtype == torch.float32 and s.dtype == torch.float16 and
                           fused_brsa_available())
        reference_input = (fused_reference_input(p, s) if fused_reference else
                           torch.cat((p.to(s.dtype), s), 1))
        control = self.reference(reference_input)
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            fused_control = (self.fast_control_activation and fast_reconstruct and
                             control.dtype == torch.float16 and not return_aux and
                             fused_brsa_available())
            if fused_control:
                logits = None
                q, alpha = fused_control_activation(control)
            else:
                logits = control[:, :1].float()
                q = logits.sigmoid()
                alpha = control[:, 1:].float().softmax(1)
            fused_statistics = (self.fast_statistics and fast_reconstruct and
                                fused_brsa_available())
            if fused_statistics:
                descriptor, rho = fused_statistics_descriptor(
                    p, q, alpha, mask, s, self.tau, self.eps)
            else:
                mu, sigma, rho = self.statistics(p, q, alpha, mask)
                residual = (p - mu) / (sigma + self.eps)
                descriptor = torch.cat((p, mu, sigma, residual, s.float()), 1).to(s.dtype)
        if self.fast_mlp and fast_reconstruct and descriptor.dtype == torch.float16:
            modulation = fused_pointwise_mlp(descriptor, self.modulation[0], self.modulation[2])
        else:
            modulation = self.modulation(descriptor)
        if fast_reconstruct:
            fused_modulated = (self.fast_modulated_reconstruct and
                               self.fast_reconstruct_down and z.is_cuda and
                               u.dtype == torch.float16 and modulation.dtype == torch.float16 and
                               fused_brsa_available())
            if fused_modulated:
                delta_z = fused_modulate_frequency(z, modulation, rho)
            else:
                mod = modulation.reshape(b, 4, 1, 16, h, w).tanh()
                delta_z = (z.to(u.dtype).reshape(b, 4, 8, 16, h, w) * mod *
                           rho.to(u.dtype)[:, None, None])
                delta_z = delta_z.reshape(b, 32, 16, h, w)
            reconstructed = None
        else:
            with torch.autocast(device_type=shallow.device.type, enabled=False):
                mod = modulation.float().reshape(b, 4, 1, 16, h, w).tanh()
                delta_z = z.reshape(b, 4, 8, 16, h, w) * mod * rho[:, None, None]
                delta_z = delta_z.reshape(b, 32, 16, h, w)
                reconstructed = self.idct(delta_z)
        if fast_reconstruct:
            downsampled = self.reconstruct_down(delta_z, semantic.shape[-2:], u.dtype)
        else:
            downsampled = self.down(reconstructed.to(u.dtype))
        delta = self.output(downsampled)
        fused_enhance = (self.fast_enhance and return_recovered_delta and fast_reconstruct and
                         semantic.dtype == torch.float16 and
                         semantic.stride() == delta.stride() and fused_brsa_available())
        if fused_enhance:
            enhanced, recovered_delta = fused_enhance_delta(semantic, delta)
        else:
            enhanced = semantic + delta.to(semantic.dtype)
            recovered_delta = enhanced - semantic if return_recovered_delta else None
        aux = dict(logits=logits, mask=mask) if return_aux else None
        if diagnostics:
            aux = {} if aux is None else aux
            aux['diagnostics'] = {k: v.detach().cpu() for k, v in dict(q=q, rho=rho, delta=delta).items()}
        return ((enhanced, aux, recovered_delta) if return_recovered_delta else
                (enhanced, aux))

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

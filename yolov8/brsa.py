"""Inference implementation of the BRSA module used in the 2026-09 runs."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def projection(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class BRSA(nn.Module):
    def __init__(self, cx, cf, cr=32, block_size=4, groups=4):
        super().__init__()
        self.b = block_size
        self.cr = cr
        self.g = groups
        self.eps = 1e-6
        self.windows = (3, 5)
        self.valid = None

        if cr % groups != 0 or block_size not in (2, 4, 8):
            raise ValueError("invalid BRSA grouping")
        self.d = groups * block_size * block_size
        self.xproj = projection(cx, cr)
        self.sproj = projection(cf, 16)
        self.reference = nn.Sequential(
            nn.Conv2d(self.d + 16, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 1 + len(self.windows), 1),
        )
        self.modulation = nn.Sequential(
            nn.Conv2d(4 * self.d + 16, 32, 1),
            nn.SiLU(),
            nn.Conv2d(32, self.d, 1),
        )
        self.align = nn.Sequential(
            nn.Conv2d(cr, cr, 3, 2, 1, groups=cr, bias=False),
            nn.Conv2d(cr, cf, 1, bias=False),
        )
        self.tau = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))

        x = torch.arange(block_size).float()
        u = x[:, None]
        basis = (
            torch.cos(math.pi * (2 * x + 1) * u / (2 * block_size))
            * math.sqrt(2 / block_size)
        )
        basis[0] /= math.sqrt(2)
        self.register_buffer("A", basis)
        for radius in self.windows:
            kernel = torch.ones(1, 1, radius, radius)
            kernel[0, 0, radius // 2, radius // 2] = 0
            self.register_buffer(f"kernel{radius}", kernel)

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def dct(self, features):
        n, c, height, width = features.shape
        block = self.b
        features = F.pad(
            features,
            (0, (-width) % block, 0, (-height) % block),
        )
        hb = features.shape[-2] // block
        wb = features.shape[-1] // block
        blocks = features.reshape(n, c, hb, block, wb, block).permute(
            0, 1, 2, 4, 3, 5
        )
        transformed = torch.einsum(
            "ui,ncpqij,vj->ncpquv",
            self.A.float(),
            blocks.float(),
            self.A.float(),
        )
        return transformed.permute(0, 1, 4, 5, 2, 3).reshape(
            n, c, block * block, hb, wb
        )

    def idct(self, transformed, shape):
        n, c, _, hb, wb = transformed.shape
        block = self.b
        blocks = transformed.reshape(n, c, block, block, hb, wb).permute(
            0, 1, 4, 5, 2, 3
        )
        features = torch.einsum(
            "ui,ncpquv,vj->ncpqij",
            self.A.float(),
            blocks.float(),
            self.A.float(),
        )
        return features.permute(0, 1, 2, 4, 3, 5).reshape(
            n, c, hb * block, wb * block
        )[..., : shape[0], : shape[1]]

    def statistics(self, power, foreground, alpha, valid_cells):
        weight = valid_cells * (1 - foreground.detach())
        means = []
        seconds = []
        supports = []
        counts = []
        for radius in self.windows:
            kernel = getattr(self, f"kernel{radius}").float()
            denominator = F.conv2d(weight, kernel, padding=radius // 2)
            count = F.conv2d(valid_cells, kernel, padding=radius // 2)
            expanded = kernel.expand(power.shape[1], 1, radius, radius)

            def grouped_conv(value):
                return F.conv2d(
                    value,
                    expanded,
                    padding=radius // 2,
                    groups=power.shape[1],
                )

            means.append(grouped_conv(weight * power) / denominator.clamp_min(self.eps))
            seconds.append(
                grouped_conv(weight * power.square()) / denominator.clamp_min(self.eps)
            )
            supports.append(denominator / count.clamp_min(1))
            counts.append(count)

        mean = sum(alpha[:, i : i + 1] * value for i, value in enumerate(means))
        second = sum(alpha[:, i : i + 1] * value for i, value in enumerate(seconds))
        sigma = (torch.clamp(second - mean.square(), min=0) + self.eps).sqrt()
        support = sum(
            alpha[:, i : i + 1] * value for i, value in enumerate(supports)
        )
        available = (sum(counts) > 0).float()
        rho = support / (support + F.softplus(self.tau))
        return mean, sigma, rho * available

    def forward(self, inputs):
        shallow, semantic = inputs
        projected = self.xproj(shallow)
        semantic_projected = self.sproj(semantic)
        if self.b > 2:
            semantic_projected = F.avg_pool2d(
                semantic_projected,
                self.b // 2,
                self.b // 2,
                ceil_mode=True,
            )

        with torch.autocast(device_type=shallow.device.type, enabled=False):
            transformed = self.dct(projected.float())
            n, c, frequencies, hb, wb = transformed.shape
            power = torch.log1p(
                transformed.reshape(
                    n, self.g, c // self.g, frequencies, hb, wb
                )
                .square()
                .mean(2)
            ).reshape(n, self.d, hb, wb)
            semantic_projected = semantic_projected.float()
            if self.valid is None:
                valid = torch.ones(
                    n,
                    1,
                    shallow.shape[-2] * 4,
                    shallow.shape[-1] * 4,
                    device=shallow.device,
                )
            else:
                valid = self.valid.float()
            cell = self.b * 4
            valid = F.pad(
                valid,
                (0, (-valid.shape[-1]) % cell, 0, (-valid.shape[-2]) % cell),
            )
            valid_cells = (F.avg_pool2d(valid, cell, cell) >= 1 - 1e-7).float()

        logits = self.reference(torch.cat([power, semantic_projected], 1))
        foreground = logits[:, :1].float().sigmoid()
        alpha = logits[:, 1:].float().softmax(1)
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            mean, sigma, rho = self.statistics(power, foreground, alpha, valid_cells)
            residual = (power - mean) / (sigma + self.eps)

        modulation_input = torch.cat(
            [power, mean, sigma, residual, semantic_projected], 1
        )
        modulation_logits = self.modulation(modulation_input)
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            modulation = (rho * torch.tanh(modulation_logits.float())).reshape(
                n, self.g, 1, frequencies, hb, wb
            )
            delta_spectrum = (
                transformed.reshape(n, self.g, c // self.g, frequencies, hb, wb)
                * modulation
            ).reshape_as(transformed)
            reconstructed = self.idct(delta_spectrum, projected.shape[-2:])

        delta = self.align(reconstructed.to(projected.dtype)).to(semantic.dtype)
        if delta.shape != semantic.shape:
            raise RuntimeError(f"shape mismatch: {delta.shape} != {semantic.shape}")
        return semantic + delta

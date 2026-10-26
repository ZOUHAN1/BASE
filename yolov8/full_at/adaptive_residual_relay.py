import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3k2
from .background_referenced_reading import conv_bn_act
from .geometry import masked_softmax


def pack_candidates(x, factor, target_shape=None):
    b, c, h, w = x.shape
    ph, pw = (-h) % factor, (-w) % factor
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    hc, wc = x.shape[-2] // factor, x.shape[-1] // factor
    packed = x.reshape(b, c, hc, factor, wc, factor).permute(0, 3, 5, 1, 2, 4)
    packed = packed.reshape(b, factor * factor, c, hc, wc)
    if target_shape is not None:
        packed = packed[..., :target_shape[0], :target_shape[1]]
    return packed


class AdaptiveResidualRelay(nn.Module):
    def __init__(self, factor, base_channels=512, correction_channels=256, refine=None):
        super().__init__()
        self.factor = factor
        self.query = conv_bn_act(base_channels, 32)
        self.key = nn.Conv2d(correction_channels, 32, 1, bias=False)
        self.score = nn.Sequential(nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.position_bias = nn.Parameter(torch.zeros(factor * factor))
        self.value = nn.Conv2d(correction_channels, base_channels, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.refine = (C3k2(base_channels, base_channels, n=1, c3k=True, e=.5,
                            shortcut=True) if refine is None else refine)
        for mlp in (self.score, self.gate):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def aggregate(self, base, correction, valid):
        q, key = self.query(base), self.key(correction)
        keys = pack_candidates(key, self.factor, base.shape[-2:])
        candidates = pack_candidates(correction, self.factor, base.shape[-2:])
        mask = pack_candidates(valid.float(), self.factor, base.shape[-2:])[:, :, 0].bool()
        b, k, _, h, w = keys.shape
        inputs = torch.cat((q[:, None].expand(-1, k, -1, -1, -1), keys), 2).reshape(b * k, 64, h, w)
        scores = self.score(inputs).reshape(b, k, h, w)
        with torch.autocast(device_type=base.device.type, enabled=False):
            weights = masked_softmax(scores.float() + self.position_bias.float()[None, :, None, None], mask, 1)
            a = (weights[:, :, None] * candidates.float()).sum(1)
            kbar = (weights[:, :, None] * keys.float()).sum(1)
        value = self.value(a.to(base.dtype))
        gate = self.gate(torch.cat((q, kbar.to(q.dtype)), 1)).sigmoid()
        return value, gate, weights

    def forward(self, base, correction, valid, diagnostics=False):
        value, gate, weights = self.aggregate(base, correction, valid)
        result = self.refine(base + gate * value)
        return (result, dict(weights=weights.detach().cpu(), gate=gate.detach().cpu())) if diagnostics else result


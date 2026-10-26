import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3k2
from .background_referenced_reading import conv_bn_act
from .fused_arr import (fused_arr_aggregate, fused_arr_available, fused_arr_gate,
                        fused_arr_relay, fused_arr_score)
from .geometry import masked_softmax


def pack_candidates(x, factor, target_shape=None):
    b, c, h, w = x.shape
    ph, pw = (-h) % factor, (-w) % factor
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    hc, wc = x.shape[-2] // factor, x.shape[-1] // factor
    packed = F.pixel_unshuffle(x, factor).reshape(b, c, factor * factor, hc, wc)
    packed = packed.permute(0, 2, 1, 3, 4)
    if target_shape is not None:
        packed = packed[..., :target_shape[0], :target_shape[1]]
    return packed


class AdaptiveResidualRelay(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.query = conv_bn_act(512, 32)
        self.key = nn.Conv2d(256, 32, 1, bias=False)
        self.score = nn.Sequential(nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.position_bias = nn.Parameter(torch.zeros(factor * factor))
        self.value = nn.Conv2d(256, 512, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.refine = C3k2(512, 512, n=1, c3k=True, e=.5, shortcut=True)
        for mlp in (self.score, self.gate):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        self.fast_aggregate = True
        self.fast_score = True
        self.fast_relay = True
        self.fast_gate = True

    def _apply(self, fn):
        if hasattr(self, '_fast_refine_cache'):
            self._fast_refine_cache = None
        return super()._apply(fn)

    def aggregate(self, base, correction, valid, implicit_valid=False,
                  return_weights=False, precomputed_query=None, precomputed_key=None,
                  defer_value=False):
        q = self.query(base) if precomputed_query is None else precomputed_query
        key = self.key(correction) if precomputed_key is None else precomputed_key
        first, activation, last = self.score
        fused_score = (self.fast_score and not self.training and base.is_cuda and
                       q.dtype == torch.float16 and not torch.is_grad_enabled() and
                       fused_arr_available())
        if fused_score:
            scores = fused_arr_score(q, key, first, last, self.factor)
        else:
            keys = pack_candidates(key, self.factor, base.shape[-2:])
            b, k, _, h, w = keys.shape
            query_score = F.conv2d(q, first.weight[:, :32], first.bias)
            key_score = F.conv2d(keys.reshape(b * k, 32, h, w), first.weight[:, 32:])
            hidden = activation(key_score.reshape(b, k, 32, h, w) + query_score[:, None])
            scores = F.conv2d(hidden.reshape(b * k, 32, h, w), last.weight, last.bias).reshape(b, k, h, w)
        fast_half = (not self.training and base.is_cuda and
                     scores.dtype in (torch.float16, torch.bfloat16))
        fused_aggregate = (self.fast_aggregate and fast_half and
                           scores.dtype == torch.float16 and not torch.is_grad_enabled() and
                           fused_arr_available())
        if fused_aggregate:
            a, kbar, weights = fused_arr_aggregate(
                scores, self.position_bias, correction, key, valid, self.factor,
                implicit_valid=implicit_valid, store_weights=return_weights)
        else:
            if fused_score:
                keys = pack_candidates(key, self.factor, base.shape[-2:])
            candidates = pack_candidates(correction, self.factor, base.shape[-2:])
            mask = pack_candidates(valid.float(), self.factor, base.shape[-2:])[:, :, 0].bool()
            compute_dtype = scores.dtype if fast_half else torch.float32
            with torch.autocast(device_type=base.device.type, enabled=False):
                logits = scores.to(compute_dtype) + self.position_bias.to(compute_dtype)[None, :, None, None]
                weights = masked_softmax(logits, mask, 1, force_float32=not fast_half)
                a = (weights[:, :, None] * candidates.to(compute_dtype)).sum(1)
                kbar = (weights[:, :, None] * keys.to(compute_dtype)).sum(1)
        value = a.to(base.dtype) if defer_value else self.value(a.to(base.dtype))
        fused_gate = (self.fast_gate and not self.training and base.is_cuda and
                      q.dtype == torch.float16 and not torch.is_grad_enabled() and
                      fused_arr_available())
        if fused_gate:
            gate = fused_arr_gate(q, kbar.to(q.dtype), self.gate[0], self.gate[2])
        else:
            gate = self.gate(torch.cat((q, kbar.to(q.dtype)), 1)).sigmoid()
        return value, gate, weights

    def forward(self, base, correction, valid, diagnostics=False, skip_refine=False,
                implicit_valid=False, precomputed_query=None, precomputed_key=None):
        fast_refine = (not skip_refine and not diagnostics and
                       hasattr(self, '_fast_refine_forward') and
                       not self.training and base.is_cuda and
                       not torch.is_grad_enabled())
        value, gate, weights = self.aggregate(
            base, correction, valid, implicit_valid, return_weights=diagnostics,
            precomputed_query=precomputed_query, precomputed_key=precomputed_key,
            defer_value=fast_refine)
        if fast_refine:
            return self._fast_refine_forward(base, value, gate)
        fused_relay = (self.fast_relay and not self.training and base.is_cuda and
                       base.dtype == torch.float16 and not torch.is_grad_enabled() and
                       fused_arr_available())
        relayed = fused_arr_relay(base, value, gate) if fused_relay else base + gate * value
        result = relayed if skip_refine else self.refine(relayed)
        return (result, dict(weights=weights.detach().cpu(), gate=gate.detach().cpu())) if diagnostics else result


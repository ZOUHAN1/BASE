import torch
from torch import nn
import torch.nn.functional as F
from .geometry import canonical_grid, masked_softmax, sample_points
from .fused_sampling import (fused_bilinear_sample, fused_sample_aggregate_control,
                             fused_sampling_available)
from .fused_brr import fused_brr_available, fused_source_merge


def conv_bn_act(c1, c2, kernel=1):
    return nn.Sequential(nn.Conv2d(c1, c2, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(c2), nn.SiLU())


class BackgroundReferencedReading(nn.Module):
    def __init__(self, refine, anchor_stride, source_strides):
        super().__init__()
        self.anchor_stride, self.source_strides = anchor_stride, tuple(source_strides)
        self.query = conv_bn_act(512, 32)
        self.keys = nn.ModuleList([conv_bn_act(512, 32) for _ in source_strides])
        self.values = nn.ModuleList([nn.Conv2d(512, 64, 1, bias=False) for _ in source_strides])
        self.controllers = nn.ModuleList([
            nn.Sequential(conv_bn_act(64, 64, 3), nn.Conv2d(64, 52, 1, bias=True))
            for _ in source_strides])
        self.scale_logits = nn.Conv2d(32, 2, 1) if len(source_strides) == 2 else None
        self.output = nn.Conv2d(64, 512, 1, bias=False)
        self.refine = refine
        support = [(x, y) for y in (-.5, 0, .5) for x in (-.5, 0, .5)]
        reference = [(x, y) for y in (-2, 0, 2) for x in (-2, 0, 2) if x or y]
        self.register_buffer('template', torch.tensor(support + reference).float())
        self.register_buffer('offset_scale', torch.tensor([.25] * 9 + [.5] * 8).view(1, 1, 1, 17, 1))
        for controller in self.controllers:
            nn.init.zeros_(controller[-1].weight)
            nn.init.zeros_(controller[-1].bias)
        if self.scale_logits is not None:
            nn.init.zeros_(self.scale_logits.weight)
            nn.init.zeros_(self.scale_logits.bias)
        self._grid_cache = {}
        self.parallel_inference = True
        self.overlap_query = True
        self.fast_source_merge = True
        self._source_streams = {}
        self.combined_projections = None

    def _apply(self, fn):
        self._grid_cache.clear()
        self._source_streams.clear()
        if hasattr(self, '_fast_refine_cache'):
            self._fast_refine_cache = None
        # Module.to(memory_format=channels_last) incorrectly applies the 4-D
        # format to 5-D buffers as well. Flatten this broadcast-only buffer
        # during the conversion, then restore its checkpoint-compatible shape.
        offset_shape = self.offset_scale.shape
        self.offset_scale = self.offset_scale.reshape(-1)
        try:
            result = super()._apply(fn)
        finally:
            self.offset_scale = self.offset_scale.reshape(offset_shape)
        return result

    def _center_grid(self, h, w, source, source_stride, dtype):
        device = source.device
        key = (h, w, *source.shape[-2:], source_stride, device.type, device.index, dtype)
        center = self._grid_cache.get(key)
        if center is None:
            center = canonical_grid(h, w, *source.shape[-2:], self.anchor_stride,
                                    source_stride, device).to(dtype)
            self._grid_cache[key] = center
        return center

    def forward(self, anchor, sources, masks, diagnostics=False, skip_refine=False,
                implicit_valid=False, precomputed_query=None, precomputed_sources=None):
        b, _, h, w = anchor.shape
        fast_dtype = anchor.dtype in (torch.float16, torch.bfloat16, torch.float32)
        fast_inference = (not self.training and anchor.is_cuda and fast_dtype and
                          not diagnostics and not torch.is_grad_enabled())
        fused = fast_inference and fused_sampling_available()
        compute_dtype = anchor.dtype if fast_dtype else torch.float32

        def prepare_source(index, source, _mask, key_proj, value_proj, _controller, ss):
            center = self._center_grid(h, w, source, ss, compute_dtype)
            center = center[None, :, :, None].expand(b, -1, -1, -1, -1)
            projected = (None if precomputed_sources is None else
                         precomputed_sources[index])
            if projected is not None:
                key, value = F.silu(projected[:, :32]), projected[:, 32:]
            elif self.combined_projections is None:
                key, value = key_proj(source), value_proj(source)
            else:
                projected = self.combined_projections[index](source)
                key, value = F.silu(projected[:, :32]), projected[:, 32:]
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                if fused:
                    center_key = fused_bilinear_sample(key, center)
                else:
                    center_key = sample_points(key, center, force_float32=not fast_dtype)[:, 0]
            return center, value, center_key

        def process_source(index, source, mask, key_proj, value_proj, controller, ss,
                           prepared=None):
            center, value, center_key = (prepare_source(
                index, source, mask, key_proj, value_proj, controller, ss)
                                         if prepared is None else prepared)
            control = controller(torch.cat((q, center_key.to(q.dtype)), 1))
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                ctrl = control.to(compute_dtype)
                if fused:
                    z, has_s = fused_sample_aggregate_control(
                        value, mask, center, self.template, self.offset_scale, ctrl,
                        implicit_valid=implicit_valid, store_valid=len(sources) > 1)
                else:
                    offset = ctrl[:, :34].reshape(b, 17, 2, h, w).permute(0, 3, 4, 1, 2)
                    template = self.template.to(compute_dtype)[None, None, None]
                    offset_scale = self.offset_scale.to(compute_dtype)
                    coords = center + template + offset_scale * offset.tanh()
                    value_and_mask = torch.cat((value, mask.to(value.dtype)), 1)
                    sampled = sample_points(value_and_mask, coords, force_float32=not fast_dtype)
                    vals, sampled_mask = sampled[:, :, :-1], sampled[:, :, -1]
                    valid = sampled_mask >= 1 - 1e-6
                    ws = masked_softmax(ctrl[:, 34:43], valid[:, :9], 1, force_float32=not fast_dtype)
                    wb = masked_softmax(ctrl[:, 43:51], valid[:, 9:], 1, force_float32=not fast_dtype)
                    support = (vals[:, :9] * ws[:, :, None]).sum(1)
                    reference = (vals[:, 9:] * wb[:, :, None]).sum(1)
                    has_s = valid[:, :9].any(1, keepdim=True)
                    has_b = valid[:, 9:].any(1, keepdim=True)
                    contrast = (support - reference) * has_b
                    gate = ctrl[:, 51:52].tanh()
                    z = (support + gate * contrast) * has_s
                detail = ({k: v.detach().cpu() for k, v in
                           dict(coords=coords, support_weights=ws, reference_weights=wb,
                                support_valid=has_s, reference_valid=has_b, gate=gate).items()}
                          if diagnostics else None)
            return z, has_s, detail

        arguments = [(index, *items) for index, items in enumerate(zip(
            sources, masks, self.keys, self.values, self.controllers, self.source_strides))]
        parallel = (self.parallel_inference and len(arguments) > 1 and fast_inference)
        overlap_query = self.overlap_query and fused and precomputed_query is None
        if overlap_query:
            device = anchor.device
            key = (device.type, device.index)
            streams = self._source_streams.get(key)
            if streams is None:
                streams = tuple(torch.cuda.Stream(device=device) for _ in arguments)
                self._source_streams[key] = streams
            current = torch.cuda.current_stream(device)
            prepared_sources = []
            for stream, args in zip(streams, arguments):
                stream.wait_stream(current)
                args[1].record_stream(stream)
                args[2].record_stream(stream)
                with torch.cuda.stream(stream):
                    prepared_sources.append(prepare_source(*args))
            q = self.query(anchor)
            results = []
            for stream, args, prepared in zip(streams, arguments, prepared_sources):
                stream.wait_stream(current)
                q.record_stream(stream)
                with torch.cuda.stream(stream):
                    results.append(process_source(*args, prepared=prepared))
            for stream, result in zip(streams, results):
                current.wait_stream(stream)
                result[0].record_stream(current)
                if result[1] is not None:
                    result[1].record_stream(current)
        else:
            q = self.query(anchor) if precomputed_query is None else precomputed_query
        if parallel and not overlap_query:
            device = anchor.device
            key = (device.type, device.index)
            streams = self._source_streams.get(key)
            if streams is None:
                streams = tuple(torch.cuda.Stream(device=device) for _ in arguments)
                self._source_streams[key] = streams
            current = torch.cuda.current_stream(device)
            results = []
            for stream, args in zip(streams, arguments):
                stream.wait_stream(current)
                q.record_stream(stream)
                args[1].record_stream(stream)
                args[2].record_stream(stream)
                with torch.cuda.stream(stream):
                    results.append(process_source(*args))
            for stream, result in zip(streams, results):
                current.wait_stream(stream)
                result[0].record_stream(current)
                result[1].record_stream(current)
        elif not overlap_query:
            results = [process_source(*args) for args in arguments]
        outputs = [result[0] for result in results]
        source_valid = [result[1] for result in results]
        details = [result[2] for result in results] if diagnostics else []
        if len(outputs) == 1:
            weights = source_valid[0].to(compute_dtype) if diagnostics else None
            merged = outputs[0]
        else:
            logits = self.scale_logits(q)
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                if (self.fast_source_merge and fused and logits.dtype == torch.float16 and
                        fused_brr_available()):
                    merged = fused_source_merge(
                        outputs[0], outputs[1], logits, source_valid[0], source_valid[1])
                    weights = None
                else:
                    weights = masked_softmax(
                        logits.to(compute_dtype), torch.cat(source_valid, 1), 1,
                        force_float32=not fast_dtype)
                    merged = outputs[0] * weights[:, :1] + outputs[1] * weights[:, 1:]
        fast_refine = (not skip_refine and not diagnostics and
                       hasattr(self, '_fast_refine_forward') and
                       not self.training and anchor.is_cuda and
                       not torch.is_grad_enabled())
        if fast_refine:
            out = self._fast_refine_forward(merged.to(anchor.dtype), anchor)
        else:
            semantic = self.output(merged.to(anchor.dtype))
            out = semantic if skip_refine else self.refine(torch.cat((semantic, anchor), 1))
        return (out, dict(sources=details, scale_weights=weights.detach().cpu())) if diagnostics else out

    def fuse_source_projections_(self):
        """Combine each inference-only key/value projection into one pointwise convolution."""
        if self.combined_projections is not None:
            return self
        combined = []
        for key_projection, value_projection in zip(self.keys, self.values):
            key_conv = key_projection[0]
            if not isinstance(key_conv, nn.Conv2d) or not isinstance(value_projection, nn.Conv2d):
                raise RuntimeError('Source projection fusion requires fused Conv2d modules')
            projection = nn.Conv2d(512, 96, 1, bias=True).to(
                device=key_conv.weight.device, dtype=key_conv.weight.dtype)
            with torch.no_grad():
                projection.weight.copy_(torch.cat((key_conv.weight, value_projection.weight), 0))
                key_bias = (key_conv.bias if key_conv.bias is not None else
                            torch.zeros(32, device=key_conv.weight.device, dtype=key_conv.weight.dtype))
                value_bias = (value_projection.bias if value_projection.bias is not None else
                              torch.zeros(64, device=key_conv.weight.device, dtype=key_conv.weight.dtype))
                projection.bias.copy_(torch.cat((key_bias, value_bias)))
            combined.append(projection)
        self.combined_projections = nn.ModuleList(combined)
        self.keys = nn.ModuleList([nn.Identity() for _ in self.keys])
        self.values = nn.ModuleList([nn.Identity() for _ in self.values])
        return self


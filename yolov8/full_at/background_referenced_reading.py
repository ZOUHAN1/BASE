import torch
from torch import nn
from .geometry import canonical_grid, masked_softmax, sample_points


def conv_bn_act(c1, c2, kernel=1):
    return nn.Sequential(nn.Conv2d(c1, c2, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(c2), nn.SiLU())


class BackgroundReferencedReading(nn.Module):
    def __init__(self, refine, anchor_stride, source_strides, anchor_channels=512,
                 source_channels=None, semantic_channels=512):
        super().__init__()
        self.anchor_stride, self.source_strides = anchor_stride, tuple(source_strides)
        if source_channels is None:
            source_channels = [512] * len(source_strides)
        if len(source_channels) != len(source_strides):
            raise ValueError('source_channels and source_strides must have the same length')
        self.query = conv_bn_act(anchor_channels, 32)
        self.keys = nn.ModuleList([conv_bn_act(channels, 32) for channels in source_channels])
        self.values = nn.ModuleList([
            nn.Conv2d(channels, 64, 1, bias=False) for channels in source_channels
        ])
        self.controllers = nn.ModuleList([
            nn.Sequential(conv_bn_act(64, 64, 3), nn.Conv2d(64, 52, 1, bias=True))
            for _ in source_strides])
        self.scale_logits = nn.Conv2d(32, 2, 1) if len(source_strides) == 2 else None
        self.output = nn.Conv2d(64, semantic_channels, 1, bias=False)
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

    def forward(self, anchor, sources, masks, diagnostics=False):
        q = self.query(anchor)
        b, _, h, w = anchor.shape
        outputs, source_valid, details = [], [], []
        for source, mask, key_proj, value_proj, controller, ss in zip(
                sources, masks, self.keys, self.values, self.controllers, self.source_strides):
            center = canonical_grid(h, w, *source.shape[-2:], self.anchor_stride, ss, source.device)
            center = center[None, :, :, None].expand(b, -1, -1, -1, -1)
            key, value = key_proj(source), value_proj(source)
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                center_key = sample_points(key, center)[:, 0]
            control = controller(torch.cat((q, center_key.to(q.dtype)), 1))
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                ctrl = control.float()
                offset = ctrl[:, :34].reshape(b, 17, 2, h, w).permute(0, 3, 4, 1, 2)
                coords = center + self.template.float()[None, None, None] + self.offset_scale.float() * offset.tanh()
                vals = sample_points(value, coords)
                valid = sample_points(mask.float(), coords)[:, :, 0] >= 1 - 1e-6
                ws = masked_softmax(ctrl[:, 34:43], valid[:, :9], 1)
                wb = masked_softmax(ctrl[:, 43:51], valid[:, 9:], 1)
                support = (vals[:, :9] * ws[:, :, None]).sum(1)
                reference = (vals[:, 9:] * wb[:, :, None]).sum(1)
                has_s, has_b = valid[:, :9].any(1, keepdim=True), valid[:, 9:].any(1, keepdim=True)
                contrast = (support - reference) * has_b
                gate = ctrl[:, 51:52].tanh()
                z = (support + gate * contrast) * has_s
                outputs.append(z)
                source_valid.append(has_s)
                if diagnostics:
                    details.append({k: v.detach().cpu() for k, v in
                                    dict(coords=coords, support_weights=ws, reference_weights=wb,
                                         support_valid=has_s, reference_valid=has_b, gate=gate).items()})
        logits = self.scale_logits(q) if self.scale_logits is not None else q[:, :1] * 0
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            weights = masked_softmax(logits, torch.cat(source_valid, 1), 1)
            merged = (torch.stack(outputs, 1) * weights[:, :, None]).sum(1)
        semantic = self.output(merged.to(anchor.dtype))
        out = self.refine(torch.cat((semantic, anchor), 1))
        return (out, dict(sources=details, scale_weights=weights.detach().cpu())) if diagnostics else out


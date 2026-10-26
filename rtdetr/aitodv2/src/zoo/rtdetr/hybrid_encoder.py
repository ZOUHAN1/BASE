'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class BRSAModule(nn.Module):
    def __init__(self,
                 shallow_channels=256,
                 hidden_dim=256,
                 cr=32,
                 block_size=4,
                 groups=4,
                 semantic_channels=16,
                 windows=[3, 5],
                 lambda_fg=0.1,
                 residual_scale=0.01,
                 enable_enhance=True,
                 detach_aux_inputs=False):
        super().__init__()
        assert block_size == 4, 'BRSA protocol fixes block_size=4.'
        assert groups > 0 and shallow_channels % groups == 0
        self.block_size = block_size
        self.groups = groups
        self.cr = cr
        self.windows = windows
        self.lambda_fg = lambda_fg
        self.residual_scale = residual_scale
        self.enable_enhance = enable_enhance
        self.detach_aux_inputs = detach_aux_inputs

        coeff_channels = groups * block_size * block_size
        self.semantic_proj = nn.Conv2d(hidden_dim, semantic_channels, 1)
        self.lbr = nn.Conv2d(coeff_channels + semantic_channels, 1 + len(windows), 1)
        self.rsm = nn.Sequential(
            nn.Conv2d(coeff_channels * 4 + semantic_channels, cr, 1),
            nn.SiLU(),
            nn.Conv2d(cr, coeff_channels, 1),
        )
        self.dw_down = nn.Conv2d(shallow_channels, shallow_channels, 3, 2, 1,
                                 groups=shallow_channels, bias=False)
        self.out_proj = nn.Conv2d(shallow_channels, hidden_dim, 1, bias=False)

        nn.init.zeros_(self.rsm[-1].weight)
        nn.init.zeros_(self.rsm[-1].bias)

        dct = self._build_dct(block_size)
        self.register_buffer('dct_mat', dct, persistent=False)

    @staticmethod
    def _build_dct(n):
        mat = torch.empty(n, n)
        for k in range(n):
            scale = (1.0 / n) ** 0.5 if k == 0 else (2.0 / n) ** 0.5
            for i in range(n):
                mat[k, i] = scale * torch.cos(torch.tensor(torch.pi * (2 * i + 1) * k / (2 * n)))
        return mat

    def _dct_blocks(self, x):
        b, c, h, w = x.shape
        bs = self.block_size
        h_crop, w_crop = h - h % bs, w - w % bs
        x = x[:, :, :h_crop, :w_crop].float()
        x = x.view(b, self.groups, c // self.groups, h_crop, w_crop).mean(2)
        x = x.view(b, self.groups, h_crop // bs, bs, w_crop // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct = self.dct_mat.to(dtype=x.dtype, device=x.device)
        coeff = torch.einsum('uv,bgijvw,wx->bgijux', dct, x, dct.t())
        return coeff.reshape(b, self.groups * bs * bs, h_crop // bs, w_crop // bs), (h_crop, w_crop)

    def _idct_blocks(self, coeff, spatial_size, channels):
        b, _, hb, wb = coeff.shape
        bs = self.block_size
        coeff = coeff.view(b, self.groups, bs, bs, hb, wb).permute(0, 1, 4, 5, 2, 3).contiguous()
        dct = self.dct_mat.to(dtype=coeff.dtype, device=coeff.device)
        blocks = torch.einsum('uv,bgijvw,wx->bgijux', dct.t(), coeff, dct)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = blocks.view(b, self.groups, hb * bs, wb * bs)
        x = x.repeat_interleave(channels // self.groups, dim=1)
        h, w = spatial_size
        return x[:, :channels, :h, :w]

    def _local_reference(self, p, q, valid_mask):
        refs, vars_, supports = [], [], []
        weight = valid_mask * (1. - q.detach().sigmoid())
        for window in self.windows:
            pad = window // 2
            kernel = torch.ones(1, 1, window, window, dtype=p.dtype, device=p.device)
            kernel[:, :, pad, pad] = 0.
            denom = F.conv2d(weight, kernel, padding=pad).clamp_min(1e-4)
            mean = F.conv2d(p * weight, kernel.expand(p.shape[1], 1, window, window),
                            padding=pad, groups=p.shape[1]) / denom
            mean2 = F.conv2d(p.square() * weight, kernel.expand(p.shape[1], 1, window, window),
                             padding=pad, groups=p.shape[1]) / denom
            refs.append(mean)
            vars_.append((mean2 - mean.square()).clamp_min(0.).sqrt())
            supports.append((denom / float(window * window - 1)).clamp(0., 1.))
        return refs, vars_, supports

    def _occupancy_loss(self, logits, targets):
        if targets is None:
            return None
        b, _, h, w = logits.shape
        target_map = torch.zeros((b, 1, h, w), dtype=logits.dtype, device=logits.device)
        yy = (torch.arange(h, device=logits.device, dtype=logits.dtype) + 0.5) / h
        xx = (torch.arange(w, device=logits.device, dtype=logits.dtype) + 0.5) / w
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        for bi, tgt in enumerate(targets):
            boxes = tgt.get('boxes')
            if boxes is None or boxes.numel() == 0:
                continue
            cx, cy, bw, bh = boxes.unbind(-1)
            x1 = (cx - bw * 0.5).clamp(0., 1.)[:, None, None]
            y1 = (cy - bh * 0.5).clamp(0., 1.)[:, None, None]
            x2 = (cx + bw * 0.5).clamp(0., 1.)[:, None, None]
            y2 = (cy + bh * 0.5).clamp(0., 1.)[:, None, None]
            occ = ((gx[None] >= x1) & (gx[None] <= x2) & (gy[None] >= y1) & (gy[None] <= y2)).any(0)
            target_map[bi, 0] = occ.to(logits.dtype)
        return self.lambda_fg * F.binary_cross_entropy_with_logits(logits, target_map)

    def forward(self, shallow, feat, targets=None):
        brsa_shallow = shallow.detach() if self.detach_aux_inputs else shallow
        brsa_feat = feat.detach() if self.detach_aux_inputs else feat
        p, spatial_size = self._dct_blocks(brsa_shallow)
        s = self.semantic_proj(brsa_feat)
        s = F.adaptive_avg_pool2d(s, p.shape[-2:])
        q_alpha = self.lbr(torch.cat([p.to(s.dtype), s], dim=1))
        q = q_alpha[:, :1].float()
        alpha = q_alpha[:, 1:].float().softmax(1)
        valid = torch.ones_like(q)

        refs, sigmas, supports = self._local_reference(p.float(), q, valid)
        mu = sum(alpha[:, i:i + 1] * refs[i] for i in range(len(refs)))
        sigma = sum(alpha[:, i:i + 1] * sigmas[i] for i in range(len(sigmas)))
        rho = sum(alpha[:, i:i + 1] * supports[i] for i in range(len(supports)))
        residual = p.float() - mu
        h = self.rsm(torch.cat([p.float(), mu, sigma, residual, s.float()], dim=1))
        aux_loss = self._occupancy_loss(q, targets)
        if not self.enable_enhance or self.residual_scale == 0:
            return feat, aux_loss

        delta_coeff = rho * torch.tanh(h) * p.float()
        delta = self._idct_blocks(delta_coeff, spatial_size, brsa_shallow.shape[1])
        delta = self.out_proj(self.dw_down(delta.to(feat.dtype)))
        if delta.shape[-2:] != feat.shape[-2:]:
            delta = F.interpolate(delta, size=feat.shape[-2:], mode='nearest')
        return feat + self.residual_scale * delta, aux_loss


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_brsa=False,
                 brsa=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.use_brsa = use_brsa
        self.brsa_aux_loss = None

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self.brsa = BRSAModule(hidden_dim=hidden_dim, **(brsa or {})) if use_brsa else None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                # persistent=False keeps it out of state_dict, so released
                # checkpoints still load with strict=True
                self.register_buffer(f'pos_embed{idx}', pos_embed, persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, targets=None):
        shallow = None
        if len(feats) == len(self.in_channels) + 1:
            shallow, feats = feats[0], feats[1:]
        assert len(feats) == len(self.in_channels)
        self.brsa_aux_loss = None
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        if self.brsa is not None:
            assert shallow is not None, 'PResNet.expose_c2 must be enabled when HybridEncoder.use_brsa=True.'
            inner_outs[0], self.brsa_aux_loss = self.brsa(shallow, inner_outs[0], targets)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs

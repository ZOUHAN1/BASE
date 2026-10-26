import torch
import torch.nn.functional as F


def valid_cells(pixel_mask, stride):
    return -F.max_pool2d(-pixel_mask.float(), stride, stride) >= 1 - 1e-6


def masked_softmax(logits, valid, dim):
    valid = valid.bool()
    masked = logits.float().masked_fill(~valid, -1e30)
    weights = torch.softmax(masked, dim=dim) * valid
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def canonical_grid(h, w, source_h, source_w, anchor_stride, source_stride, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    x = (x.float() + .5) * anchor_stride / source_stride - .5
    y = (y.float() + .5) * anchor_stride / source_stride - .5
    return torch.stack((x, y), -1)


def normalize_grid(coords, h, w):
    return 2 * (coords + .5) / coords.new_tensor([w, h]) - 1


def sample_points(values, coords):
    # Stack all points in one grid_sample call: B,H,W,K,2 -> B,H,W*K,2.
    b, h, w, k, _ = coords.shape
    grid = normalize_grid(coords, *values.shape[-2:]).reshape(b, h, w * k, 2)
    sampled = F.grid_sample(values.float(), grid.float(), mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    return sampled.reshape(b, values.shape[1], h, w, k).permute(0, 4, 1, 2, 3)


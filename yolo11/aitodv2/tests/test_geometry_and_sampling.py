import pytest
import torch
from models.geometry import canonical_grid, masked_softmax, sample_points, valid_cells
from models.background_referenced_reading import BackgroundReferencedReading
from models.fused_sampling import (fused_bilinear_sample, fused_sample_aggregate,
                                   fused_sampling_available)
from models.fused_brr import fused_brr_available, fused_source_merge
from torch import nn
from ultralytics.nn.modules import C3k2


def test_masked_softmax_empty_and_partial_gradient():
    logits = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[[1], [0], [1], [0]], [[0], [0], [0], [0]]]).expand_as(logits)
    result = masked_softmax(logits, mask, 1)
    assert torch.allclose(result[0].sum(0), torch.ones(3))
    assert result[1].count_nonzero() == 0 and torch.isfinite(result).all()
    result.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_strict_min_pool():
    pixels = torch.ones(1, 1, 32, 32)
    pixels[:, :, 0, 0] = 0
    valid = valid_cells(pixels, 16)
    assert valid.tolist() == [[[[False, True], [True, True]]]]


def test_coordinate_ramp_and_offset_units():
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
    value = torch.stack((x, y))[None].float()
    center = canonical_grid(8, 8, 8, 8, 8, 16, 'cpu')
    coord = center[None, :, :, None]
    sample = sample_points(value, coord)[0, 0]
    assert torch.allclose(sample[:, 2:7, 2:7], center[2:7, 2:7].permute(2, 0, 1))
    shifted = sample_points(value, coord + torch.tensor([.25, -.25]))[0, 0]
    assert torch.allclose(shifted[:, 3:7, 3:7] - sample[:, 3:7, 3:7],
                          torch.tensor([.25, -.25])[:, None, None].expand(2, 4, 4), atol=1e-6)


def test_impulse_half_pixel():
    value = torch.zeros(1, 1, 4, 4)
    value[0, 0, 1, 1] = 1
    coords = torch.tensor([[[[[1., 1.], [1.5, 1.], [1.5, 1.5]]]]])
    assert torch.allclose(sample_points(value, coords).flatten(), torch.tensor([1., .5, .25]))


@pytest.mark.parametrize('sources', [[32], [16, 32]])
def test_brr_empty_and_gate_gradients(sources):
    module = BackgroundReferencedReading(C3k2(1024, 256 if len(sources) == 2 else 512, n=1, c3k=True),
                                         8 if len(sources) == 2 else 16, sources).eval()
    anchor = torch.randn(1, 512, 16, 16)
    values = [torch.randn(1, 512, 16 // (s // module.anchor_stride), 16 // (s // module.anchor_stride)) for s in sources]
    masks = [torch.zeros_like(v[:, :1], dtype=torch.bool) for v in values]
    out, diag = module(anchor, values, masks, True)
    assert torch.isfinite(out).all() and diag['scale_weights'].count_nonzero() == 0
    masks = [torch.ones_like(v[:, :1], dtype=torch.bool) for v in values]
    for controller in module.controllers:
        controller[-1].bias.data[51] = .2
    out, diag = module(anchor, values, masks, True)
    out.square().mean().backward()
    for controller in module.controllers:
        grads = controller[-1].weight.grad
        assert torch.isfinite(grads).all()
        assert grads[:34].abs().sum() > 0
        assert grads[34:51].abs().sum() > 0
        assert grads[51:].abs().sum() > 0


def test_brr_reference_empty_disables_contrast():
    module = BackgroundReferencedReading(nn.Identity(), 16, [32]).eval()
    # A one-cell valid island makes the support nonempty and the outer reference empty.
    source = torch.randn(1, 512, 5, 5)
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, 2, 2] = 1
    anchor = torch.randn(1, 512, 10, 10)
    first, diag = module(anchor, [source], [mask], True)
    module.controllers[0][-1].bias.data[51] = 1
    second, _ = module(anchor, [source], [mask], True)
    no_reference = ~diag['sources'][0]['reference_valid']
    assert torch.allclose(first[:, :512] * no_reference, second[:, :512] * no_reference)


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_sampling_available(),
                    reason='CUDA and Triton are required')
@pytest.mark.parametrize('channels_last', [False, True])
def test_fused_sampling_matches_float32_reference(channels_last):
    torch.manual_seed(11)
    values = torch.randn(2, 64, 7, 6, device='cuda', dtype=torch.float16)
    mask = torch.ones(2, 1, 7, 6, device='cuda', dtype=torch.bool)
    coords = torch.empty(2, 9, 8, 17, 2, device='cuda', dtype=torch.float16)
    coords[..., 0].uniform_(0, 5)
    coords[..., 1].uniform_(0, 6)
    control = torch.randn(2, 52, 9, 8, device='cuda', dtype=torch.float16)
    if channels_last:
        values = values.contiguous(memory_format=torch.channels_last)
        mask = mask.contiguous(memory_format=torch.channels_last)
        control = control.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        center = fused_bilinear_sample(values[:, :32], coords[:, :, :, :1])
        center_reference = sample_points(values[:, :32].float(), coords[:, :, :, :1].float())[:, 0]
        actual, support_valid = fused_sample_aggregate(values, mask, coords, control)
        sampled = sample_points(values.float(), coords.float())
        support_weights = torch.softmax(control[:, 34:43].float(), 1)
        reference_weights = torch.softmax(control[:, 43:51].float(), 1)
        support = (sampled[:, :9] * support_weights[:, :, None]).sum(1)
        reference = (sampled[:, 9:] * reference_weights[:, :, None]).sum(1)
        expected = support + control[:, 51:52].float().tanh() * (support - reference)
    assert support_valid.all()
    assert actual.is_contiguous(memory_format=(
        torch.channels_last if channels_last else torch.contiguous_format))
    assert torch.allclose(center.float(), center_reference, atol=1e-3, rtol=1e-3)
    assert torch.allclose(actual.float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_brr_available(),
                    reason='CUDA and Triton are required')
def test_fused_brr_source_merge_is_within_one_fp16_ulp():
    torch.manual_seed(23)
    first = torch.randn(1, 64, 9, 7, device='cuda', dtype=torch.float16)
    second = torch.randn_like(first)
    logits = torch.randn(1, 2, 9, 7, device='cuda', dtype=torch.float16)
    valid = torch.rand(1, 2, 9, 7, device='cuda') > .2
    weights = masked_softmax(logits, valid, 1, force_float32=False)
    expected = first * weights[:, :1] + second * weights[:, 1:]
    actual = fused_source_merge(first, second, logits, valid[:, :1], valid[:, 1:])
    assert (actual.float() - expected.float()).abs().max() <= 2 ** -9

import pytest
import torch
from models.geometry import canonical_grid, masked_softmax, sample_points, valid_cells
from models.background_referenced_reading import BackgroundReferencedReading
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

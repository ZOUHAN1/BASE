import pytest
import torch
from models.adaptive_residual_relay import AdaptiveResidualRelay, pack_candidates


@pytest.mark.parametrize('factor', [2, 4])
def test_pack_row_major(factor):
    x = torch.arange(factor * factor * 6).reshape(1, 1, 2 * factor, 3 * factor).float()
    packed = pack_candidates(x, factor)
    for iy in range(2):
        for ix in range(3):
            expected = x[0, 0, iy * factor:(iy + 1) * factor, ix * factor:(ix + 1) * factor].flatten()
            assert torch.equal(packed[0, :, 0, iy, ix], expected)


@pytest.mark.parametrize('factor', [2, 4])
def test_arr_zero_constant_and_gradients(factor):
    module = AdaptiveResidualRelay(factor).eval()
    base = torch.randn(1, 512, 3, 4)
    zero = torch.zeros(1, 256, 3 * factor, 4 * factor)
    valid = torch.ones_like(zero[:, :1], dtype=torch.bool)
    assert torch.equal(module(base, zero, valid), module.refine(base))
    constant = torch.ones_like(zero) * 2
    value, gate, weight = module.aggregate(base, constant, valid)
    assert torch.allclose(value, module.value(torch.ones_like(base[:, :256]) * 2), atol=1e-6)
    assert torch.all(gate == .5)
    assert torch.allclose(weight.sum(1), torch.ones(1, 3, 4))
    delta = torch.randn_like(zero, requires_grad=True)
    module.score[-1].weight.data.normal_(0, .1)
    module.gate[-1].weight.data.normal_(0, .1)
    result = module(base, delta, valid)
    result.square().mean().backward()
    assert delta.grad.abs().sum() > 0
    for parameter in (module.score[0].weight, module.score[-1].weight, module.gate[0].weight,
                      module.gate[-1].weight, module.key.weight, module.value.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    empty = module(base, delta.detach(), valid * False)
    assert torch.allclose(empty, module.refine(base), atol=1e-6)


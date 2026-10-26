import torch
import torch.nn.functional as F
import pytest
from models.brsa_original import OriginalBRSA
from models.fused_brsa import (fused_brsa_available, fused_control_activation,
                               fused_modulate_frequency, fused_reconstruct_down,
                               fused_reference_input,
                               fused_silu_avg_pool2d)


def test_dct_idct_and_dc():
    module = OriginalBRSA()
    x = torch.randn(2, 32, 24, 16)
    assert (module.idct(module.dct(x)) - x).abs().max() < 1e-5
    z = module.dct(torch.ones(1, 32, 8, 12))
    assert torch.allclose(z[:, :, 0], torch.full_like(z[:, :, 0], 4.))
    assert z[:, :, 1:].abs().max() < 1e-5
    assert 'dct_matrix' not in dict(module.named_parameters())


def test_combined_reconstruct_down_matches_two_stage_operator():
    torch.manual_seed(7)
    module = OriginalBRSA().eval()
    z = torch.randn(1, 32, 16, 3, 4)
    reconstructed = module.idct(z)
    expected = module.down(reconstructed)
    actual = module.reconstruct_down(z, expected.shape[-2:], z.dtype)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=3e-5, rtol=1e-5)
    assert '_reconstruct_down_cache' not in module.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_brsa_available(),
                    reason='CUDA and Triton are required')
def test_fused_reconstruct_down_matches_cudnn():
    torch.manual_seed(17)
    module = OriginalBRSA().cuda().half().eval()
    z = torch.randn(1, 32, 16, 3, 4, device='cuda', dtype=torch.float16)
    kernel = module.combined_reconstruct_down_kernel(torch.float16)
    expected = F.conv_transpose2d(z.reshape(1, 512, 3, 4), kernel,
                                  stride=2, groups=32)[..., :6, :8]
    actual = fused_reconstruct_down(z, kernel, (6, 8))
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_brsa_available(),
                    reason='CUDA and Triton are required')
def test_fused_frequency_modulation_matches_pytorch():
    torch.manual_seed(29)
    z = torch.randn(1, 32, 16, 3, 4, device='cuda', dtype=torch.float32)
    modulation = torch.randn(1, 64, 3, 4, device='cuda', dtype=torch.float16)
    rho = torch.rand(1, 1, 3, 4, device='cuda', dtype=torch.float32)
    expected = (z.half().reshape(1, 4, 8, 16, 3, 4) *
                modulation.reshape(1, 4, 1, 16, 3, 4).tanh() *
                rho.half()[:, None, None]).reshape(1, 32, 16, 3, 4)
    actual = fused_modulate_frequency(z, modulation, rho)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_brsa_available(),
                    reason='CUDA and Triton are required')
def test_fused_silu_avg_pool2d_matches_pytorch():
    torch.manual_seed(37)
    x = torch.randn(2, 16, 18, 14, device='cuda', dtype=torch.float16)
    expected = F.avg_pool2d(F.silu(x), 2, 2)
    actual = fused_silu_avg_pool2d(x)
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_brsa_available(),
                    reason='CUDA and Triton are required')
def test_fused_control_activation_matches_pytorch():
    torch.manual_seed(41)
    control = torch.randn(2, 3, 11, 9, device='cuda', dtype=torch.float16)
    expected = (control[:, :1].float().sigmoid(), control[:, 1:].float().softmax(1))
    actual = fused_control_activation(control)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not fused_brsa_available(),
                    reason='CUDA and Triton are required')
def test_fused_reference_input_matches_pytorch():
    torch.manual_seed(47)
    frequency = torch.randn(2, 64, 11, 9, device='cuda', dtype=torch.float32)
    semantic = torch.randn(2, 16, 11, 9, device='cuda', dtype=torch.float16)
    expected = torch.cat((frequency.half(), semantic), 1)
    actual = fused_reference_input(frequency, semantic)
    assert torch.equal(actual, expected)


def test_brsa_identity_mask_empty_and_first_update():
    module = OriginalBRSA().eval()
    shallow = torch.randn(1, 256, 32, 32, requires_grad=True)
    semantic = torch.randn(1, 256, 16, 16, requires_grad=True)
    mask = torch.ones(1, 1, 128, 128)
    out, aux = module(shallow, semantic, mask, True)
    assert torch.equal(out, semantic)
    out.square().mean().backward()
    assert module.modulation[-1].weight.grad.abs().sum() > 0
    assert semantic.grad is not None
    empty, details = module(shallow, semantic, mask * 0, True, True)
    assert torch.equal(empty, semantic)
    assert details['diagnostics']['rho'].count_nonzero() == 0


def test_neighbor_statistics_second_moment_and_empty():
    module = OriginalBRSA()
    p = torch.ones(1, 64, 5, 5) * 2
    q = torch.ones(1, 1, 5, 5) * .2
    alpha = torch.ones(1, 2, 5, 5) * .5
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mu, sigma, rho = module.statistics(p, q, alpha, mask)
    assert torch.allclose(mu, p, atol=1e-5)
    assert sigma.max() < .005 and (rho > 0).all()
    mu, sigma, rho = module.statistics(p, q, alpha, mask * False)
    assert mu.count_nonzero() == rho.count_nonzero() == 0


def test_occupancy_strict_intersection():
    boxes = torch.tensor([[.25, .25, .5, .5], [.75, .75, .1, .1]])
    target = OriginalBRSA.occupancy_target(boxes, torch.tensor([0, 0]), (1, 1, 2, 2), (32, 32))
    assert target.tolist() == [[[[True, False], [False, True]]]]


def test_balanced_occupancy_empty_and_nonempty():
    module = OriginalBRSA()
    logits = torch.randn(1, 1, 2, 2, requires_grad=True)
    batch = dict(img=torch.zeros(1, 3, 32, 32), bboxes=torch.tensor([[.25, .25, .5, .5]]),
                 batch_idx=torch.tensor([0]))
    aux = dict(logits=logits, mask=torch.ones_like(logits, dtype=torch.bool))
    result = module.occupancy_loss(aux, batch)
    target = torch.tensor([[[[1., 0.], [0., 0.]]]])
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    assert torch.allclose(result, (raw[0, 0, 0, 0] + raw[target == 0].mean()) / 2)
    aux['mask'] *= False
    zero = module.occupancy_loss(aux, batch)
    assert zero == 0
    zero.backward()
    assert logits.grad is not None


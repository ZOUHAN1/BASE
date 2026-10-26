import torch
import torch.nn.functional as F
from models.brsa_original import OriginalBRSA


def test_dct_idct_and_dc():
    module = OriginalBRSA()
    x = torch.randn(2, 32, 24, 16)
    assert (module.idct(module.dct(x)) - x).abs().max() < 1e-5
    z = module.dct(torch.ones(1, 32, 8, 12))
    assert torch.allclose(z[:, :, 0], torch.full_like(z[:, :, 0], 4.))
    assert z[:, :, 1:].abs().max() < 1e-5
    assert 'dct_matrix' not in dict(module.named_parameters())


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


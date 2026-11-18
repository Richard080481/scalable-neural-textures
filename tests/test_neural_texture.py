import torch

from neural_texture.models import NeuralTexture


def test_uv_corner_convention_is_u_horizontal_v_vertical():
    sampler = NeuralTexture(channels=1, height=2, width=2, align_corners=True)
    with torch.no_grad():
        # Layout is top-left=1, top-right=2, bottom-left=3, bottom-right=4.
        sampler.texture.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))
    uv = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]])
    sampled = sampler(uv)
    assert torch.equal(sampled, sampler.texture)


def test_bilinear_center_sample_and_texture_gradient():
    sampler = NeuralTexture(channels=1, height=2, width=2, align_corners=True)
    with torch.no_grad():
        sampler.texture.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))
    value = sampler(torch.tensor([[[[0.5, 0.5]]]]))
    assert torch.allclose(value, torch.tensor([[[[2.5]]]]))
    value.sum().backward()
    assert sampler.texture.grad is not None
    assert torch.allclose(sampler.texture.grad, torch.full_like(sampler.texture, 0.25))


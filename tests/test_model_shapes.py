import torch

from neural_texture.models import NeuralTextureModel


def test_model_output_shape_and_gradients_reach_both_components():
    model = NeuralTextureModel(4, 8, 8, renderer_width=8, renderer_depth=2)
    uv = torch.rand(2, 6, 7, 2)
    mask = torch.ones(2, 1, 6, 7)
    output = model(uv, mask)
    assert output.shape == (2, 3, 6, 7)
    output.mean().backward()
    assert model.neural_texture.texture.grad is not None
    assert model.neural_texture.texture.grad.abs().sum() > 0
    renderer_grads = [parameter.grad for parameter in model.renderer.parameters()]
    assert all(gradient is not None for gradient in renderer_grads)
    assert sum(gradient.abs().sum() for gradient in renderer_grads) > 0


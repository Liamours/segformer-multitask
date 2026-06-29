import torch

from src.backbones import MIT_B0, MIT_B2, MixVisionTransformer


def test_mit_b0_feature_pyramid_shapes():
    model = MixVisionTransformer(MIT_B0)
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert len(outputs) == 4
    assert tuple(feature.shape[1] for feature in outputs) == MIT_B0.stage_channels
    assert outputs[0].shape[2:] == (16, 16)
    assert outputs[1].shape[2:] == (8, 8)
    assert outputs[2].shape[2:] == (4, 4)
    assert outputs[3].shape[2:] == (2, 2)


def test_mit_b2_feature_pyramid_shapes():
    model = MixVisionTransformer(MIT_B2)
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert len(outputs) == 4
    assert tuple(feature.shape[1] for feature in outputs) == MIT_B2.stage_channels

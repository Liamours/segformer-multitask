import torch

from src.backbones import MIT_B0
from src.decoders import SegFormerDecoder


def test_segformer_decoder_output_shape():
    decoder = SegFormerDecoder(MIT_B0.stage_channels, embedding_dim=128)
    features = (
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 160, 4, 4),
        torch.randn(2, 256, 2, 2),
    )
    output = decoder(features)
    assert output.shape == (2, 128, 16, 16)

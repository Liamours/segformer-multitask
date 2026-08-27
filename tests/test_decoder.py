import torch

from src.backbones import MIT_B0
from src.decoders import SegFormerDecoder, SegFormerFusionHead, SegFormerProjectionTrunk


def _b0_features():
    return (
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 160, 4, 4),
        torch.randn(2, 256, 2, 2),
    )


def test_segformer_decoder_output_shape():
    decoder = SegFormerDecoder(MIT_B0.stage_channels, embedding_dim=128)
    output = decoder(_b0_features())
    assert output.shape == (2, 128, 16, 16)


def test_projection_trunk_output_shape():
    trunk = SegFormerProjectionTrunk(MIT_B0.stage_channels, embedding_dim=128)
    output = trunk(_b0_features())
    assert output.shape == (2, 128 * 4, 16, 16)
    assert trunk.output_dim == 128 * 4


def test_fusion_head_output_shape():
    trunk = SegFormerProjectionTrunk(MIT_B0.stage_channels, embedding_dim=128)
    fuse = SegFormerFusionHead(trunk.output_dim, output_dim=64)
    output = fuse(trunk(_b0_features()))
    assert output.shape == (2, 64, 16, 16)


def test_trunk_and_fusion_head_match_decoder_for_shared_weights():
    decoder = SegFormerDecoder(MIT_B0.stage_channels, embedding_dim=128)
    trunk = SegFormerProjectionTrunk(MIT_B0.stage_channels, embedding_dim=128)
    fuse = SegFormerFusionHead(trunk.output_dim, output_dim=decoder.output_dim, dropout=0.0)
    decoder.dropout.p = 0.0

    # Copy the decoder's own weights into the split-out trunk/fuse pair so the two paths compute
    # the identical projection+fuse chain, only through separate module objects.
    trunk.projections.load_state_dict(decoder.projections.state_dict())
    fuse.fuse.load_state_dict(decoder.fuse.state_dict())

    decoder.eval()
    trunk.eval()
    fuse.eval()

    features = _b0_features()
    expected = decoder(features)
    actual = fuse(trunk(features))
    assert torch.allclose(actual, expected, atol=1e-6)

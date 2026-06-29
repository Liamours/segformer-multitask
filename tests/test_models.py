import torch

from src.backbones import MIT_B0, MixVisionTransformer
from src.decoders import SegFormerDecoder
from src.heads import SegmentationHead
from src.models import DualDecoderSegFormer, DualHeadSegFormer, SingleTaskSegFormer


def _build_parts():
    backbone = MixVisionTransformer(MIT_B0)
    decoder = SegFormerDecoder(MIT_B0.stage_channels)
    head = SegmentationHead(decoder.output_dim, 4)
    return backbone, decoder, head


def test_single_task_forward():
    backbone, decoder, head = _build_parts()
    model = SingleTaskSegFormer(backbone, decoder, head)
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 64, 64)


def test_dual_head_forward():
    backbone, decoder, head = _build_parts()
    model = DualHeadSegFormer(backbone, decoder, head, SegmentationHead(decoder.output_dim, 4))
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["task_a_logits"].shape == (2, 4, 64, 64)
    assert outputs["task_b_logits"].shape == (2, 4, 64, 64)


def test_dual_decoder_forward():
    backbone, decoder, head = _build_parts()
    model = DualDecoderSegFormer(
        backbone,
        decoder,
        SegFormerDecoder(MIT_B0.stage_channels),
        head,
        SegmentationHead(decoder.output_dim, 4),
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["task_a_logits"].shape == (2, 4, 64, 64)
    assert outputs["task_b_logits"].shape == (2, 4, 64, 64)

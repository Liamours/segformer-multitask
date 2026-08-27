import torch

from src.backbones import MIT_B0, MixVisionTransformer
from src.decoders import SegFormerDecoder, SegFormerFusionHead, SegFormerProjectionTrunk
from src.heads import SegmentationHead
from src.models import DualDecoderSegFormer, DualFuseSegFormer, DualHeadSegFormer, SingleTaskSegFormer
from src.train import build_model
from src.types import TaskClassCounts


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


def test_task_specific_class_counts():
    model = build_model(
        variant="mit_b1",
        task_mode="dual_head",
        num_classes=TaskClassCounts(task_a=3, task_b=5),
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["task_a_logits"].shape == (2, 3, 64, 64)
    assert outputs["task_b_logits"].shape == (2, 5, 64, 64)


def test_dual_fuse_forward():
    backbone = MixVisionTransformer(MIT_B0)
    trunk = SegFormerProjectionTrunk(MIT_B0.stage_channels)
    fuse_a = SegFormerFusionHead(trunk.output_dim)
    fuse_b = SegFormerFusionHead(trunk.output_dim)
    model = DualFuseSegFormer(
        backbone,
        trunk,
        fuse_a,
        fuse_b,
        SegmentationHead(fuse_a.output_dim, 4),
        SegmentationHead(fuse_b.output_dim, 4),
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["task_a_logits"].shape == (2, 4, 64, 64)
    assert outputs["task_b_logits"].shape == (2, 4, 64, 64)


def test_dual_fuse_shares_trunk_but_not_fuse_or_head_weights():
    model = build_model(
        variant="mit_b0",
        task_mode="dual_fuse",
        num_classes=TaskClassCounts(task_a=3, task_b=5),
    )
    assert model.fuse_a is not model.fuse_b
    assert model.head_a is not model.head_b

    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    outputs["task_a_logits"].sum().backward()

    # Only fuse_a/head_a should receive gradients from a task_a-only loss; the shared trunk sits
    # upstream of both tasks and must, while the sibling task_b branch must not, or the "shared
    # trunk, split fuse" claim behind dual_fuse is not actually true.
    assert all(param.grad is not None for param in model.trunk.parameters())
    assert all(param.grad is not None for param in model.fuse_a.parameters())
    assert all(param.grad is None for param in model.fuse_b.parameters())
    assert all(param.grad is None for param in model.head_b.parameters())


def test_dual_fuse_task_specific_class_counts():
    model = build_model(
        variant="mit_b1",
        task_mode="dual_fuse",
        num_classes=TaskClassCounts(task_a=3, task_b=5),
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["task_a_logits"].shape == (2, 3, 64, 64)
    assert outputs["task_b_logits"].shape == (2, 5, 64, 64)

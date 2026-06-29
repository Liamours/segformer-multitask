import torch

from src.heads import SegmentationHead


def test_segmentation_head_output_size():
    head = SegmentationHead(in_channels=128, num_classes=3)
    features = torch.randn(2, 128, 16, 16)
    logits = head(features, output_size=(64, 64))
    assert logits.shape == (2, 3, 64, 64)

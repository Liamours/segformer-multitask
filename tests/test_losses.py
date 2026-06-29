import torch

from src.losses import segmentation_loss


def test_segmentation_loss_runs():
    logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    loss = segmentation_loss(logits, target)
    assert loss.ndim == 0

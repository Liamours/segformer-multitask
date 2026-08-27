import torch

from src.losses import segmentation_loss, soft_dice_loss


def test_segmentation_loss_runs():
    logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    loss = segmentation_loss(logits, target)
    assert loss.ndim == 0


def test_ce_only_segmentation_loss_runs():
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    loss = segmentation_loss(logits, target, name="ce")
    assert loss.ndim == 0


def test_soft_dice_loss_runs():
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    loss = soft_dice_loss(logits, target)
    assert loss.ndim == 0
    assert loss >= 0

import torch

from src.metrics import segmentation_scores


def test_segmentation_scores_include_accuracy_iou_and_dice():
    logits = torch.tensor(
        [
            [
                [[10.0, 0.0], [0.0, 10.0]],
                [[0.0, 10.0], [10.0, 0.0]],
            ]
        ]
    )
    target = torch.tensor([[[0, 1], [1, 0]]])

    scores = segmentation_scores(logits, target, num_classes=2)

    assert scores["pixel_accuracy"].item() == 1.0
    assert scores["mean_iou"].item() == 1.0
    assert scores["mean_dice"].item() == 1.0
    assert scores["foreground_mean_iou"].item() == 1.0
    assert scores["foreground_mean_dice"].item() == 1.0


def test_segmentation_scores_foreground_mean_excludes_background():
    logits = torch.tensor(
        [
            [
                [[10.0, 10.0], [10.0, 0.0]],
                [[0.0, 0.0], [0.0, 10.0]],
            ]
        ]
    )
    target = torch.tensor([[[0, 0], [1, 1]]])

    scores = segmentation_scores(logits, target, num_classes=2)

    assert scores["class_0_dice"].item() > scores["class_1_dice"].item()
    assert scores["foreground_mean_dice"].item() == scores["class_1_dice"].item()

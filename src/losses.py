import torch
import torch.nn.functional as F


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


def multitask_segmentation_losses(
    task_a_logits: torch.Tensor,
    task_a_target: torch.Tensor,
    task_b_logits: torch.Tensor,
    task_b_target: torch.Tensor,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    return {
        "task_a": segmentation_loss(task_a_logits, task_a_target, ignore_index=ignore_index),
        "task_b": segmentation_loss(task_b_logits, task_b_target, ignore_index=ignore_index),
    }

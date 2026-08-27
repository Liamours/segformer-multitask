import torch
import torch.nn.functional as F


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    name: str = "dice_ce",
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long(), ignore_index=ignore_index)
    if name == "ce":
        return ce
    if name != "dice_ce":
        raise ValueError(f"Unsupported loss={name}.")
    dice = soft_dice_loss(logits, target, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dice


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    probabilities = torch.softmax(logits, dim=1)
    safe_target = target.long().clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    valid_mask = valid.unsqueeze(1)
    probabilities = probabilities * valid_mask
    one_hot = one_hot * valid_mask
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def multitask_segmentation_losses(
    task_a_logits: torch.Tensor,
    task_a_target: torch.Tensor,
    task_b_logits: torch.Tensor,
    task_b_target: torch.Tensor,
    ignore_index: int = 255,
    name: str = "dice_ce",
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    return {
        "task_a": segmentation_loss(
            task_a_logits,
            task_a_target,
            ignore_index=ignore_index,
            name=name,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        ),
        "task_b": segmentation_loss(
            task_b_logits,
            task_b_target,
            ignore_index=ignore_index,
            name=name,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        ),
    }

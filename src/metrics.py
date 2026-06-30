import torch


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    correct = (prediction[valid] == target[valid]).float().mean()
    return correct


def segmentation_scores(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    prediction = logits.argmax(dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=logits.device)
        return {"pixel_accuracy": zero, "mean_iou": zero, "mean_dice": zero}

    pixel_acc = (prediction[valid] == target[valid]).float().mean()
    ious = []
    dices = []
    for class_index in range(num_classes):
        pred_class = (prediction == class_index) & valid
        target_class = (target == class_index) & valid
        union = pred_class | target_class
        if union.sum() == 0:
            continue
        intersection = pred_class & target_class
        ious.append(intersection.float().sum() / union.float().sum().clamp_min(1.0))
        denominator = pred_class.float().sum() + target_class.float().sum()
        dices.append((2.0 * intersection.float().sum()) / denominator.clamp_min(1.0))

    if not ious:
        zero = torch.tensor(0.0, device=logits.device)
        mean_iou = zero
        mean_dice = zero
    else:
        mean_iou = torch.stack(ious).mean()
        mean_dice = torch.stack(dices).mean()

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
    }

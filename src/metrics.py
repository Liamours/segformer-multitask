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
        scores = {
            "pixel_accuracy": zero,
            "mean_iou": zero,
            "mean_dice": zero,
            "foreground_mean_iou": zero,
            "foreground_mean_dice": zero,
        }
        for class_index in range(num_classes):
            scores[f"class_{class_index}_iou"] = zero
            scores[f"class_{class_index}_dice"] = zero
        return scores

    pixel_acc = (prediction[valid] == target[valid]).float().mean()
    ious = []
    dices = []
    foreground_ious = []
    foreground_dices = []
    scores: dict[str, torch.Tensor] = {"pixel_accuracy": pixel_acc}
    for class_index in range(num_classes):
        pred_class = (prediction == class_index) & valid
        target_class = (target == class_index) & valid
        union = pred_class | target_class
        if union.sum() == 0:
            zero = torch.tensor(0.0, device=logits.device)
            scores[f"class_{class_index}_iou"] = zero
            scores[f"class_{class_index}_dice"] = zero
            continue
        intersection = pred_class & target_class
        class_iou = intersection.float().sum() / union.float().sum().clamp_min(1.0)
        denominator = pred_class.float().sum() + target_class.float().sum()
        class_dice = (2.0 * intersection.float().sum()) / denominator.clamp_min(1.0)
        scores[f"class_{class_index}_iou"] = class_iou
        scores[f"class_{class_index}_dice"] = class_dice
        ious.append(class_iou)
        dices.append(class_dice)
        if class_index != 0:
            foreground_ious.append(class_iou)
            foreground_dices.append(class_dice)

    if not ious:
        zero = torch.tensor(0.0, device=logits.device)
        mean_iou = zero
        mean_dice = zero
    else:
        mean_iou = torch.stack(ious).mean()
        mean_dice = torch.stack(dices).mean()

    scores["mean_iou"] = mean_iou
    scores["mean_dice"] = mean_dice
    if foreground_ious:
        scores["foreground_mean_iou"] = torch.stack(foreground_ious).mean()
        scores["foreground_mean_dice"] = torch.stack(foreground_dices).mean()
    else:
        zero = torch.tensor(0.0, device=logits.device)
        scores["foreground_mean_iou"] = zero
        scores["foreground_mean_dice"] = zero
    return scores

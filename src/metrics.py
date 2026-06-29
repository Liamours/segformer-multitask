import torch


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    correct = (prediction[valid] == target[valid]).float().mean()
    return correct

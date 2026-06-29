import torch
import torch.nn.functional as F
from torch import nn


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        logits = self.classifier(x)
        if output_size is not None and logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

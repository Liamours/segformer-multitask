import torch
import torch.nn.functional as F
from torch import nn

from .types import FeaturePyramid


class SegFormerDecoder(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        embedding_dim: int = 256,
        output_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embedding_dim, kernel_size=1)
            for channels in in_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: FeaturePyramid) -> torch.Tensor:
        if len(features) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} feature maps, got {len(features)}.")

        target_size = features[0].shape[2:]
        projected = []
        for feature, projection in zip(features, self.projections, strict=True):
            x = projection(feature)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)

        fused = self.fuse(torch.cat(projected, dim=1))
        return self.dropout(fused)

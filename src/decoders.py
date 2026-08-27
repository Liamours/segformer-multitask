import torch
import torch.nn.functional as F
from torch import nn

from .types import FeaturePyramid


def _project_pyramid(features: FeaturePyramid, projections: nn.ModuleList) -> torch.Tensor:
    if len(features) != len(projections):
        raise ValueError(f"Expected {len(projections)} feature maps, got {len(features)}.")

    target_size = features[0].shape[2:]
    projected = []
    for feature, projection in zip(features, projections, strict=True):
        x = projection(feature)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        projected.append(x)
    return torch.cat(projected, dim=1)


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
        fused = self.fuse(_project_pyramid(features, self.projections))
        return self.dropout(fused)


class SegFormerProjectionTrunk(nn.Module):
    """The shared per-scale projection stage of a SegFormer decoder, stopping short of fuse.

    Same projection-then-concat computation as `SegFormerDecoder`, split out so one shared
    instance can feed two independent `SegFormerFusionHead`s (mid fission: shared projections,
    task-specific fuse and head). Does not wrap `SegFormerDecoder` itself, an existing
    `SegFormerDecoder` instance always owns an unused `fuse`/`dropout` for the three trained
    single_task/dual_head/dual_decoder checkpoints; those checkpoints' state_dict keys must stay
    exactly as they are, so this is a separate class rather than a change to that one.
    """

    def __init__(self, in_channels: tuple[int, int, int, int], embedding_dim: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim * len(in_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embedding_dim, kernel_size=1)
            for channels in in_channels
        )

    def forward(self, features: FeaturePyramid) -> torch.Tensor:
        return _project_pyramid(features, self.projections)


class SegFormerFusionHead(nn.Module):
    """The task-specific fuse stage of a SegFormer decoder: `SegFormerDecoder`'s `fuse`+`dropout`,
    split out so two independent instances can each own their own weights while reading from one
    shared `SegFormerProjectionTrunk`."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim or in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fuse(projected))

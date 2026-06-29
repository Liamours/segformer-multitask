from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from segformer_multitask.types import FeaturePyramid


@dataclass(frozen=True)
class MiTSpec:
    name: str
    embed_dims: int
    num_stages: int
    num_layers: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    patch_sizes: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]
    sr_ratios: tuple[int, int, int, int]
    mlp_ratio: int
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float

    @property
    def stage_channels(self) -> tuple[int, int, int, int]:
        return tuple(self.embed_dims * head for head in self.num_heads)


MIT_B0 = MiTSpec(
    name="mit_b0",
    embed_dims=32,
    num_stages=4,
    num_layers=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3),
    strides=(4, 2, 2, 2),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratio=4,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
)

MIT_B2 = MiTSpec(
    name="mit_b2",
    embed_dims=64,
    num_stages=4,
    num_layers=(3, 4, 6, 3),
    num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3),
    strides=(4, 2, 2, 2),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratio=4,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
)


class PyramidBackbone(nn.Module, ABC):
    @property
    @abstractmethod
    def out_channels(self) -> tuple[int, int, int, int]:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        pass


class MixVisionTransformer(PyramidBackbone):
    def __init__(self, spec: MiTSpec, in_channels: int = 3) -> None:
        super().__init__()
        self.spec = spec
        self.in_channels = in_channels

    @property
    def out_channels(self) -> tuple[int, int, int, int]:
        return self.spec.stage_channels

    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        raise NotImplementedError(
            "MixVisionTransformer is scaffolded but not implemented yet. "
            "Next step: port the MiT backbone logic using mmsegmentation as the reference."
        )


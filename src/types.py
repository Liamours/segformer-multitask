from collections.abc import Sequence
from dataclasses import dataclass

import torch

FeaturePyramid = Sequence[torch.Tensor]
LossDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class TaskClassCounts:
    task_a: int
    task_b: int

from collections.abc import Sequence

import torch

FeaturePyramid = Sequence[torch.Tensor]
LossDict = dict[str, torch.Tensor]

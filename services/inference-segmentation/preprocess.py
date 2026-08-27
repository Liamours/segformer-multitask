from __future__ import annotations

from dataclasses import dataclass

import torch

from pair_loader import PairedImageSample
from transforms import InferenceTransformConfig, deterministic_paired_transform


@dataclass(frozen=True)
class PreprocessedPair:
    case_id: str
    views: tuple[str, str]
    image: torch.Tensor
    original_paths: dict[str, str]


def preprocess_pair(sample: PairedImageSample, config: InferenceTransformConfig) -> PreprocessedPair:
    canvas = deterministic_paired_transform(sample.anterior_path, sample.posterior_path, config)
    image = canvas.unsqueeze(0)
    return PreprocessedPair(
        case_id=sample.case_id,
        views=sample.views,
        image=image,
        original_paths={
            "anterior": str(sample.anterior_path),
            "posterior": str(sample.posterior_path),
        },
    )

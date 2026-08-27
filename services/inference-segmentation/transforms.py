from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class InferenceTransformConfig:
    image_size: tuple[int, int] = (256, 1024)
    input_channels: int = 3
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None


def resolve_mean_std(
    input_channels: int,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if input_channels not in {1, 3}:
        raise ValueError("input_channels must be 1 or 3.")
    if mean is None:
        mean = (0.485, 0.456, 0.406) if input_channels == 3 else (0.5,)
    if std is None:
        std = (0.229, 0.224, 0.225) if input_channels == 3 else (0.5,)
    mean_tuple = tuple(float(value) for value in mean)
    std_tuple = tuple(float(value) for value in std)
    if len(mean_tuple) != input_channels or len(std_tuple) != input_channels:
        raise ValueError("mean/std length must match input_channels.")
    if any(value <= 0 for value in std_tuple):
        raise ValueError("std values must be positive.")
    return mean_tuple, std_tuple


def load_image(path: str | Path, input_channels: int) -> Image.Image:
    image = Image.open(path)
    if input_channels == 1:
        return image.convert("L")
    if input_channels == 3:
        return image.convert("RGB")
    raise ValueError("input_channels must be 1 or 3.")


def resize_image(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    height, width = image_size
    return image.resize((width, height), Image.Resampling.BILINEAR)


def image_to_tensor(image: Image.Image, input_channels: int) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if input_channels == 1:
        array = array[None, :, :]
    else:
        array = array.transpose(2, 0, 1)
    return torch.from_numpy(array)


def normalize_tensor(
    tensor: torch.Tensor,
    input_channels: int,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> torch.Tensor:
    mean_tuple, std_tuple = resolve_mean_std(input_channels, mean, std)
    mean_tensor = torch.tensor(mean_tuple, dtype=tensor.dtype).view(input_channels, 1, 1)
    std_tensor = torch.tensor(std_tuple, dtype=tensor.dtype).view(input_channels, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def deterministic_transform(path: str | Path, config: InferenceTransformConfig) -> torch.Tensor:
    image = load_image(path, config.input_channels)
    image = resize_image(image, config.image_size)
    tensor = image_to_tensor(image, config.input_channels)
    return normalize_tensor(tensor, config.input_channels, config.mean, config.std)


def deterministic_paired_transform(
    anterior_path: str | Path,
    posterior_path: str | Path,
    config: InferenceTransformConfig,
) -> torch.Tensor:
    """Anterior/posterior side by side into one canvas, matching src/datasets.py training-time loading."""
    height, width = config.image_size
    if width % 2 != 0:
        raise ValueError("image_size width must be even for paired-view inference.")
    view_size = (height, width // 2)
    views = []
    for path in (anterior_path, posterior_path):
        image = load_image(path, config.input_channels)
        image = resize_image(image, view_size)
        tensor = image_to_tensor(image, config.input_channels)
        tensor = normalize_tensor(tensor, config.input_channels, config.mean, config.std)
        views.append(tensor)
    return torch.cat(views, dim=2)

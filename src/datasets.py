from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .types import TaskClassCounts


@dataclass(frozen=True)
class SegmentationSamplePaths:
    image: Path
    posterior_image: Path | None = None
    mask: Path | None = None
    posterior_mask: Path | None = None
    task_a_mask: Path | None = None
    posterior_task_a_mask: Path | None = None
    task_b_mask: Path | None = None
    posterior_task_b_mask: Path | None = None


class DummySegmentationDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        image_size: tuple[int, int] = (64, 64),
        num_classes: int | TaskClassCounts = 4,
        multitask: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.multitask = multitask
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand((3, *self.image_size), generator=generator)
        if isinstance(self.num_classes, TaskClassCounts):
            task_a_classes = self.num_classes.task_a
            task_b_classes = self.num_classes.task_b
            mask = torch.randint(0, task_a_classes, self.image_size, generator=generator)
        else:
            task_a_classes = self.num_classes
            task_b_classes = self.num_classes
            mask = torch.randint(0, self.num_classes, self.image_size, generator=generator)

        if self.multitask:
            task_b_mask = torch.randint(0, task_b_classes, self.image_size, generator=generator)
            return {"image": image, "task_a_mask": mask, "task_b_mask": task_b_mask}

        return {"image": image, "mask": mask}


class SingleTaskSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSamplePaths],
        image_size: tuple[int, int],
        normalize_mean: tuple[float, ...] | None = None,
        normalize_std: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        if not self.samples:
            raise ValueError("SingleTaskSegmentationDataset requires at least one sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        if sample.mask is None:
            raise ValueError("Single-task sample is missing mask path.")
        if sample.posterior_image is not None:
            if sample.posterior_mask is None:
                raise ValueError("Paired single-task sample is missing posterior mask path.")
            return {
                "image": load_paired_image_tensor(
                    sample.image,
                    sample.posterior_image,
                    self.image_size,
                    self.normalize_mean,
                    self.normalize_std,
                ),
                "mask": load_paired_mask_tensor(sample.mask, sample.posterior_mask, self.image_size),
            }
        return {
            "image": load_image_tensor(sample.image, self.image_size, self.normalize_mean, self.normalize_std),
            "mask": load_mask_tensor(sample.mask, self.image_size),
        }


class MultiTaskSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSamplePaths],
        image_size: tuple[int, int],
        normalize_mean: tuple[float, ...] | None = None,
        normalize_std: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        if not self.samples:
            raise ValueError("MultiTaskSegmentationDataset requires at least one sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        if sample.task_a_mask is None or sample.task_b_mask is None:
            raise ValueError("Multitask sample is missing one or more task mask paths.")
        if sample.posterior_image is not None:
            if sample.posterior_task_a_mask is None or sample.posterior_task_b_mask is None:
                raise ValueError("Paired multitask sample is missing one or more posterior task mask paths.")
            return {
                "image": load_paired_image_tensor(
                    sample.image,
                    sample.posterior_image,
                    self.image_size,
                    self.normalize_mean,
                    self.normalize_std,
                ),
                "task_a_mask": load_paired_mask_tensor(sample.task_a_mask, sample.posterior_task_a_mask, self.image_size),
                "task_b_mask": load_paired_mask_tensor(sample.task_b_mask, sample.posterior_task_b_mask, self.image_size),
            }
        return {
            "image": load_image_tensor(sample.image, self.image_size, self.normalize_mean, self.normalize_std),
            "task_a_mask": load_mask_tensor(sample.task_a_mask, self.image_size),
            "task_b_mask": load_mask_tensor(sample.task_b_mask, self.image_size),
        }


def build_folder_samples(
    root_dir: str | Path,
    split: str | Path | None,
    image_dir: str,
    mask_dir: str,
    task_a_mask_dir: str,
    task_b_mask_dir: str,
    image_suffix: str,
    mask_suffix: str,
    multitask: bool,
    paired_views: bool = False,
    anterior_suffix: str = "_0000",
    posterior_suffix: str = "_0001",
) -> list[SegmentationSamplePaths]:
    root = Path(root_dir)
    image_root = root / image_dir
    if split is None:
        sample_ids = sorted(path.stem for path in image_root.glob(f"*{image_suffix}"))
    else:
        split_path = Path(split)
        if not split_path.is_absolute():
            split_path = root / split_path
        sample_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    samples: list[SegmentationSamplePaths] = []
    for sample_id in sample_ids:
        if paired_views:
            samples.append(
                _build_paired_sample(
                    root=root,
                    sample_id=sample_id,
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    task_a_mask_dir=task_a_mask_dir,
                    task_b_mask_dir=task_b_mask_dir,
                    image_suffix=image_suffix,
                    mask_suffix=mask_suffix,
                    multitask=multitask,
                    anterior_suffix=anterior_suffix,
                    posterior_suffix=posterior_suffix,
                )
            )
            continue
        image_path = image_root / f"{sample_id}{image_suffix}"
        _require_path(image_path, "image")
        if multitask:
            task_a_mask_path = root / task_a_mask_dir / f"{sample_id}{mask_suffix}"
            task_b_mask_path = root / task_b_mask_dir / f"{sample_id}{mask_suffix}"
            _require_path(task_a_mask_path, "task_a_mask")
            _require_path(task_b_mask_path, "task_b_mask")
            samples.append(
                SegmentationSamplePaths(
                    image=image_path,
                    task_a_mask=task_a_mask_path,
                    task_b_mask=task_b_mask_path,
                )
            )
        else:
            mask_path = root / mask_dir / f"{sample_id}{mask_suffix}"
            _require_path(mask_path, "mask")
            samples.append(SegmentationSamplePaths(image=image_path, mask=mask_path))

    if not samples:
        raise ValueError(f"No samples found under {root}.")
    return samples


def _build_paired_sample(
    root: Path,
    sample_id: str,
    image_dir: str,
    mask_dir: str,
    task_a_mask_dir: str,
    task_b_mask_dir: str,
    image_suffix: str,
    mask_suffix: str,
    multitask: bool,
    anterior_suffix: str,
    posterior_suffix: str,
) -> SegmentationSamplePaths:
    anterior_id = f"{sample_id}{anterior_suffix}"
    posterior_id = f"{sample_id}{posterior_suffix}"
    anterior_image = root / image_dir / f"{anterior_id}{image_suffix}"
    posterior_image = root / image_dir / f"{posterior_id}{image_suffix}"
    _require_path(anterior_image, "anterior image")
    _require_path(posterior_image, "posterior image")

    if multitask:
        anterior_task_a = root / task_a_mask_dir / f"{anterior_id}{mask_suffix}"
        posterior_task_a = root / task_a_mask_dir / f"{posterior_id}{mask_suffix}"
        anterior_task_b = root / task_b_mask_dir / f"{anterior_id}{mask_suffix}"
        posterior_task_b = root / task_b_mask_dir / f"{posterior_id}{mask_suffix}"
        _require_path(anterior_task_a, "anterior task_a_mask")
        _require_path(posterior_task_a, "posterior task_a_mask")
        _require_path(anterior_task_b, "anterior task_b_mask")
        _require_path(posterior_task_b, "posterior task_b_mask")
        return SegmentationSamplePaths(
            image=anterior_image,
            posterior_image=posterior_image,
            task_a_mask=anterior_task_a,
            posterior_task_a_mask=posterior_task_a,
            task_b_mask=anterior_task_b,
            posterior_task_b_mask=posterior_task_b,
        )

    anterior_mask = root / mask_dir / f"{anterior_id}{mask_suffix}"
    posterior_mask = root / mask_dir / f"{posterior_id}{mask_suffix}"
    _require_path(anterior_mask, "anterior mask")
    _require_path(posterior_mask, "posterior mask")
    return SegmentationSamplePaths(
        image=anterior_image,
        posterior_image=posterior_image,
        mask=anterior_mask,
        posterior_mask=posterior_mask,
    )


def load_image_tensor(
    path: Path,
    image_size: tuple[int, int],
    normalize_mean: tuple[float, ...] | None = None,
    normalize_std: tuple[float, ...] | None = None,
) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size[1], image_size[0]), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return normalize_image_tensor(tensor, normalize_mean, normalize_std)


def load_paired_image_tensor(
    anterior_path: Path,
    posterior_path: Path,
    image_size: tuple[int, int],
    normalize_mean: tuple[float, ...] | None = None,
    normalize_std: tuple[float, ...] | None = None,
) -> torch.Tensor:
    height, width = image_size
    if width % 2 != 0:
        raise ValueError("Paired image_size width must be even.")
    view_size = (height, width // 2)
    anterior = load_image_tensor(anterior_path, view_size, normalize_mean, normalize_std)
    posterior = load_image_tensor(posterior_path, view_size, normalize_mean, normalize_std)
    return torch.cat((anterior, posterior), dim=2).contiguous()


def normalize_image_tensor(
    tensor: torch.Tensor,
    normalize_mean: tuple[float, ...] | None,
    normalize_std: tuple[float, ...] | None,
) -> torch.Tensor:
    if normalize_mean is None or normalize_std is None:
        return tensor
    mean = torch.tensor(normalize_mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(normalize_std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def load_mask_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    mask = Image.open(path)
    mask = mask.resize((image_size[1], image_size[0]), resample=Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.int64)
    return torch.from_numpy(array).contiguous()


def load_paired_mask_tensor(anterior_path: Path, posterior_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    if width % 2 != 0:
        raise ValueError("Paired image_size width must be even.")
    view_size = (height, width // 2)
    anterior = load_mask_tensor(anterior_path, view_size)
    posterior = load_mask_tensor(posterior_path, view_size)
    return torch.cat((anterior, posterior), dim=1).contiguous()


def _require_path(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} file: {path}")

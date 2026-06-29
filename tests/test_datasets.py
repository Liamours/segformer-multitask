import numpy as np
from PIL import Image

from src.datasets import MultiTaskSegmentationDataset, SingleTaskSegmentationDataset, build_folder_samples


def _write_rgb(path, value: int) -> None:
    array = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_mask(path, value: int) -> None:
    array = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_single_task_folder_dataset_contract(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    (root / "masks").mkdir()
    (root / "train.txt").write_text("sample_a\n", encoding="utf-8")
    (root / "val.txt").write_text("sample_a\n", encoding="utf-8")
    _write_rgb(root / "images" / "sample_a.png", 64)
    _write_mask(root / "masks" / "sample_a.png", 2)

    samples = build_folder_samples(root, "train.txt", "images", "masks", "task_a_masks", "task_b_masks", ".png", ".png", False)
    dataset = SingleTaskSegmentationDataset(samples, (16, 16))
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 16)
    assert sample["mask"].shape == (16, 16)


def test_multitask_folder_dataset_contract(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    (root / "task_a_masks").mkdir()
    (root / "task_b_masks").mkdir()
    (root / "train.txt").write_text("sample_a\n", encoding="utf-8")
    _write_rgb(root / "images" / "sample_a.png", 32)
    _write_mask(root / "task_a_masks" / "sample_a.png", 1)
    _write_mask(root / "task_b_masks" / "sample_a.png", 3)

    samples = build_folder_samples(root, "train.txt", "images", "masks", "task_a_masks", "task_b_masks", ".png", ".png", True)
    dataset = MultiTaskSegmentationDataset(samples, (16, 16))
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 16)
    assert sample["task_a_mask"].shape == (16, 16)
    assert sample["task_b_mask"].shape == (16, 16)

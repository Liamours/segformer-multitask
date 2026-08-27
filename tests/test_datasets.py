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


def test_paired_single_task_folder_dataset_contract(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    (root / "masks").mkdir()
    (root / "train.txt").write_text("case_a\n", encoding="utf-8")
    _write_rgb(root / "images" / "case_a_0000.png", 64)
    _write_rgb(root / "images" / "case_a_0001.png", 96)
    _write_mask(root / "masks" / "case_a_0000.png", 1)
    _write_mask(root / "masks" / "case_a_0001.png", 2)

    samples = build_folder_samples(
        root,
        "train.txt",
        "images",
        "masks",
        "task_a_masks",
        "task_b_masks",
        ".png",
        ".png",
        False,
        paired_views=True,
    )
    dataset = SingleTaskSegmentationDataset(samples, (16, 32))
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 32)
    assert sample["mask"].shape == (16, 32)
    assert sample["mask"][:, :16].unique().item() == 1
    assert sample["mask"][:, 16:].unique().item() == 2


def test_paired_multitask_folder_dataset_contract(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    (root / "task_a_masks").mkdir()
    (root / "task_b_masks").mkdir()
    (root / "train.txt").write_text("case_a\n", encoding="utf-8")
    _write_rgb(root / "images" / "case_a_0000.png", 64)
    _write_rgb(root / "images" / "case_a_0001.png", 96)
    _write_mask(root / "task_a_masks" / "case_a_0000.png", 1)
    _write_mask(root / "task_a_masks" / "case_a_0001.png", 2)
    _write_mask(root / "task_b_masks" / "case_a_0000.png", 3)
    _write_mask(root / "task_b_masks" / "case_a_0001.png", 4)

    samples = build_folder_samples(
        root,
        "train.txt",
        "images",
        "masks",
        "task_a_masks",
        "task_b_masks",
        ".png",
        ".png",
        True,
        paired_views=True,
    )
    dataset = MultiTaskSegmentationDataset(samples, (16, 32))
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 32)
    assert sample["task_a_mask"][:, :16].unique().item() == 1
    assert sample["task_a_mask"][:, 16:].unique().item() == 2
    assert sample["task_b_mask"][:, :16].unique().item() == 3
    assert sample["task_b_mask"][:, 16:].unique().item() == 4


def test_folder_dataset_applies_normalization(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    (root / "masks").mkdir()
    (root / "train.txt").write_text("sample_a\n", encoding="utf-8")
    _write_rgb(root / "images" / "sample_a.png", 128)
    _write_mask(root / "masks" / "sample_a.png", 1)

    samples = build_folder_samples(root, "train.txt", "images", "masks", "task_a_masks", "task_b_masks", ".png", ".png", False)
    dataset = SingleTaskSegmentationDataset(samples, (8, 8), normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    sample = dataset[0]

    assert sample["image"].mean().item() > 0.0
    assert sample["image"].mean().item() < 0.01

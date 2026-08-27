import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.configs import ConfigHandler
from src.train import build_model_from_config

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "inference-segmentation"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from batch_infer import run_batch_inference  # noqa: E402
from component_cleanup import remove_small_components  # noqa: E402
from main import run_segmentation  # noqa: E402
from pair_loader import build_pair, infer_case_id  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
from preprocess import preprocess_pair  # noqa: E402
from transforms import InferenceTransformConfig  # noqa: E402


def _write_gray(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    array = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size)
    Image.fromarray(array, mode="L").save(path)


def test_pair_loader_validates_case_and_preserves_order(tmp_path: Path) -> None:
    anterior = tmp_path / "bs80k_0001_0000.png"
    posterior = tmp_path / "bs80k_0001_0001.png"
    _write_gray(anterior)
    _write_gray(posterior)

    assert infer_case_id(anterior) == "bs80k_0001"
    sample = build_pair(anterior, posterior)

    assert sample.case_id == "bs80k_0001"
    assert sample.views == ("anterior", "posterior")
    assert sample.anterior_path == anterior
    assert sample.posterior_path == posterior


def test_pair_loader_rejects_mismatched_case(tmp_path: Path) -> None:
    anterior = tmp_path / "bs80k_0001_0000.png"
    posterior = tmp_path / "bs80k_0002_0001.png"
    _write_gray(anterior)
    _write_gray(posterior)

    with pytest.raises(ValueError, match="case mismatch"):
        build_pair(anterior, posterior)


def test_preprocess_pair_returns_paired_3_channel_tensor(tmp_path: Path) -> None:
    anterior = tmp_path / "bs80k_0001_0000.png"
    posterior = tmp_path / "bs80k_0001_0001.png"
    _write_gray(anterior)
    _write_gray(posterior)

    sample = build_pair(anterior, posterior)
    pair = preprocess_pair(sample, InferenceTransformConfig(image_size=(12, 20), input_channels=3))

    assert pair.image.shape == (1, 3, 12, 20)
    assert pair.views == ("anterior", "posterior")
    assert torch.isfinite(pair.image).all()


def test_postprocess_splits_tasks_and_keeps_overlap_allowed() -> None:
    lesion_logits = torch.zeros(1, 3, 8, 16)
    bone_logits = torch.zeros(1, 13, 8, 16)
    lesion_logits[:, 2, 1:3, 1:3] = 10.0
    lesion_logits[:, 2, 1:3, 9:11] = 10.0
    bone_logits[:, 4, 1:3, 1:3] = 10.0
    bone_logits[:, 4, 1:3, 9:11] = 10.0

    result = postprocess_logits({"lesion": lesion_logits, "bone": bone_logits})

    assert result.masks["lesion"]["anterior"].shape == (8, 8)
    assert result.masks["bone"]["posterior"].shape == (8, 8)
    assert result.lesion_malignant["anterior"].sum() == 4
    assert result.bone_color["anterior"].shape == (8, 8, 3)
    assert result.metadata["overlap_allowed"] is True


def test_component_cleanup_removes_only_small_components() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 2
    mask[4:7, 4:7] = 2

    cleaned = remove_small_components(mask, min_pixels=3, target_labels=[2])

    assert cleaned[1, 1] == 0
    assert cleaned[5, 5] == 2


def test_run_segmentation_loads_checkpoint_and_writes_outputs(tmp_path: Path) -> None:
    config = ConfigHandler.from_dict(
        {
            "model": {
                "variant": "mit_b0",
                "task_mode": "dual_head",
                "num_classes": 3,
                "task_a_classes": 3,
                "task_b_classes": 13,
                "decoder_dim": 16,
            },
            "data": {"image_size": [32, 32]},
        }
    )
    model = build_model_from_config(config.model)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, checkpoint)

    anterior = tmp_path / "bs80k_0001_0000.png"
    posterior = tmp_path / "bs80k_0001_0001.png"
    _write_gray(anterior, size=(32, 32))
    _write_gray(posterior, size=(32, 32))
    output_dir = tmp_path / "out"

    result = run_segmentation(
        checkpoint_path=checkpoint,
        anterior_path=anterior,
        posterior_path=posterior,
        output_dir=output_dir,
        image_size=(32, 32),
        input_channels=3,
    )

    assert set(result.masks) == {"lesion", "bone"}
    assert result.masks["lesion"]["anterior"].shape == (32, 16)
    assert (output_dir / "lesion_anterior.png").is_file()
    assert (output_dir / "bone_color_posterior.png").is_file()
    assert (output_dir / "metadata.json").is_file()

    summary_path = output_dir / "summary.csv"
    assert summary_path.is_file()
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert {row["view"] for row in rows} == {"anterior", "posterior"}
    anterior_row = next(row for row in rows if row["view"] == "anterior")
    lesion_px = int(anterior_row["lesion_background_px"]) + int(anterior_row["lesion_benign_px"]) + int(anterior_row["lesion_malignant_px"])
    assert lesion_px == 32 * 16


def test_run_batch_inference_writes_per_case_and_combined_results(tmp_path: Path) -> None:
    config = ConfigHandler.from_dict(
        {
            "model": {
                "variant": "mit_b0",
                "task_mode": "single_task",
                "num_classes": 3,
                "decoder_dim": 16,
            },
            "data": {"image_size": [32, 32]},
        }
    )
    model = build_model_from_config(config.model)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, checkpoint)

    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    for case_id in ("case_a", "case_b"):
        _write_gray(root / "images" / f"{case_id}_0000.png", size=(32, 32))
        _write_gray(root / "images" / f"{case_id}_0001.png", size=(32, 32))
    (root / "val.txt").write_text("case_a\ncase_b\n", encoding="utf-8")

    output_dir = tmp_path / "batch_out"
    combined_path = run_batch_inference(
        checkpoint_path=checkpoint,
        root_dir=root,
        split_file="val.txt",
        output_dir=output_dir,
        image_size=(32, 32),
    )

    assert combined_path == output_dir / "all_cases_summary.csv"
    assert (output_dir / "case_a" / "lesion_anterior.png").is_file()
    assert (output_dir / "case_b" / "metadata.json").is_file()

    rows = list(csv.DictReader(combined_path.open(encoding="utf-8")))
    assert {row["case_id"] for row in rows} == {"case_a", "case_b"}
    assert len(rows) == 4

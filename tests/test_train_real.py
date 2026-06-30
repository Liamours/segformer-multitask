import numpy as np
from PIL import Image

from src.configs import ConfigHandler
from src.evaluate import evaluate_checkpoint
from src.train import Trainer
from src.utils import list_num_workers_options, recommend_num_workers


def _write_rgb(path, value: int) -> None:
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path, value: int) -> None:
    Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(path)


def _make_single_task_dataset(root) -> None:
    (root / "images").mkdir()
    (root / "masks").mkdir()
    (root / "train.txt").write_text("a\nb\n", encoding="utf-8")
    (root / "val.txt").write_text("b\n", encoding="utf-8")
    _write_rgb(root / "images" / "a.png", 32)
    _write_rgb(root / "images" / "b.png", 64)
    _write_mask(root / "masks" / "a.png", 1)
    _write_mask(root / "masks" / "b.png", 2)


def _make_multitask_dataset(root) -> None:
    (root / "images").mkdir()
    (root / "task_a_masks").mkdir()
    (root / "task_b_masks").mkdir()
    (root / "train.txt").write_text("a\nb\n", encoding="utf-8")
    (root / "val.txt").write_text("b\n", encoding="utf-8")
    _write_rgb(root / "images" / "a.png", 32)
    _write_rgb(root / "images" / "b.png", 64)
    _write_mask(root / "task_a_masks" / "a.png", 1)
    _write_mask(root / "task_a_masks" / "b.png", 2)
    _write_mask(root / "task_b_masks" / "a.png", 3)
    _write_mask(root / "task_b_masks" / "b.png", 4)


def test_trainer_runs_on_real_single_task_folder_dataset(tmp_path):
    _make_single_task_dataset(tmp_path)
    config = ConfigHandler.from_dict(
        {
            "model": {"variant": "mit_b0", "task_mode": "single_task", "num_classes": 5},
            "data": {
                "dataset_name": "folder",
                "root_dir": str(tmp_path),
                "train_split": "train.txt",
                "val_split": "val.txt",
                "batch_size": 1,
                "eval_batch_size": 1,
                "image_size": [32, 32],
            },
            "logging": {"output_dir": str(tmp_path / "logs"), "save_checkpoints": True},
            "run": {"epochs": 1, "max_train_batches": 1, "max_eval_batches": 1},
        }
    )
    summary = Trainer(config).fit()

    assert "train" in summary
    assert "val" in summary
    assert "mean_iou" in summary["val"]
    assert "mean_dice" in summary["val"]
    assert (tmp_path / "logs" / "checkpoints" / "latest.pt").exists()
    eval_metrics = evaluate_checkpoint(tmp_path / "logs" / "checkpoints" / "latest.pt", max_batches=1)
    assert eval_metrics["loss"] >= 0.0
    assert "mean_iou" in eval_metrics


def test_trainer_runs_on_real_dual_decoder_folder_dataset(tmp_path):
    _make_multitask_dataset(tmp_path)
    config = ConfigHandler.from_dict(
        {
            "model": {
                "variant": "mit_b1",
                "task_mode": "dual_decoder",
                "task_a_classes": 5,
                "task_b_classes": 6,
            },
            "data": {
                "dataset_name": "folder",
                "root_dir": str(tmp_path),
                "train_split": "train.txt",
                "val_split": "val.txt",
                "batch_size": 1,
                "eval_batch_size": 1,
                "image_size": [32, 32],
            },
            "logging": {"output_dir": str(tmp_path / "logs_dual_decoder"), "save_checkpoints": True},
            "run": {"epochs": 1, "max_train_batches": 1, "max_eval_batches": 1},
        }
    )
    summary = Trainer(config).fit()

    assert summary["val"]["loss"] >= 0.0
    assert "task_a_pixel_accuracy" in summary["val"]
    assert "task_b_pixel_accuracy" in summary["val"]
    assert "task_a_mean_iou" in summary["val"]
    assert "task_b_mean_iou" in summary["val"]
    eval_metrics = evaluate_checkpoint(tmp_path / "logs_dual_decoder" / "checkpoints" / "latest.pt", max_batches=1)
    assert eval_metrics["task_a_loss"] >= 0.0
    assert eval_metrics["task_b_loss"] >= 0.0


def test_trainer_runs_on_real_dual_head_folder_dataset(tmp_path):
    _make_multitask_dataset(tmp_path)
    config = ConfigHandler.from_dict(
        {
            "model": {
                "variant": "mit_b2",
                "task_mode": "dual_head",
                "task_a_classes": 5,
                "task_b_classes": 6,
            },
            "data": {
                "dataset_name": "folder",
                "root_dir": str(tmp_path),
                "train_split": "train.txt",
                "val_split": "val.txt",
                "batch_size": 1,
                "eval_batch_size": 1,
                "image_size": [32, 32],
            },
            "logging": {"output_dir": str(tmp_path / "logs_dual_head"), "save_checkpoints": True},
            "run": {"epochs": 1, "max_train_batches": 1, "max_eval_batches": 1},
        }
    )
    summary = Trainer(config).fit()

    assert summary["train"]["loss"] >= 0.0
    assert summary["val"]["pixel_accuracy"] >= 0.0
    assert "task_a_pixel_accuracy" in summary["val"]
    assert "task_b_pixel_accuracy" in summary["val"]
    assert "task_a_mean_dice" in summary["val"]
    assert "task_b_mean_dice" in summary["val"]
    assert (tmp_path / "logs_dual_head" / "checkpoints" / "latest.pt").exists()
    eval_metrics = evaluate_checkpoint(tmp_path / "logs_dual_head" / "checkpoints" / "latest.pt", max_batches=1)
    assert eval_metrics["task_a_loss"] >= 0.0
    assert eval_metrics["task_b_loss"] >= 0.0


def test_checkpoint_selection_supports_loss_min_mode(tmp_path):
    _make_single_task_dataset(tmp_path)
    config = ConfigHandler.from_dict(
        {
            "model": {"variant": "mit_b0", "task_mode": "single_task", "num_classes": 5},
            "data": {
                "dataset_name": "folder",
                "root_dir": str(tmp_path),
                "train_split": "train.txt",
                "val_split": "val.txt",
                "batch_size": 1,
                "eval_batch_size": 1,
                "image_size": [32, 32],
            },
            "logging": {
                "output_dir": str(tmp_path / "logs_loss_mode"),
                "save_checkpoints": True,
                "checkpoint_metric": "loss",
                "checkpoint_mode": "min",
            },
            "run": {"epochs": 1, "max_train_batches": 1, "max_eval_batches": 1},
        }
    )
    summary = Trainer(config).fit()

    assert summary["best_checkpoint_path"] is not None
    assert (tmp_path / "logs_loss_mode" / "checkpoints" / "best.pt").exists()


def test_num_workers_policy_outputs_candidates():
    options = list_num_workers_options(cpu_count=8)
    recommended = recommend_num_workers(batch_size=4, cpu_count=8)

    assert options == [0, 1, 2, 4, 6]
    assert recommended == 6

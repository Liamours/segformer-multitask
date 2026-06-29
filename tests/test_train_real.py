import numpy as np
from PIL import Image

from src.configs import ConfigHandler
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
    assert (tmp_path / "logs" / "checkpoints" / "latest.pt").exists()


def test_trainer_runs_on_real_multitask_folder_dataset(tmp_path):
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
            "run": {"epochs": 1, "max_train_batches": 1, "max_eval_batches": 1},
        }
    )
    summary = Trainer(config).fit()

    assert summary["val"]["loss"] >= 0.0


def test_num_workers_policy_outputs_candidates():
    options = list_num_workers_options(cpu_count=8)
    recommended = recommend_num_workers(batch_size=4, cpu_count=8)

    assert options == [0, 1, 2, 4, 6]
    assert recommended == 6

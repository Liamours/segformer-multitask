import json
import subprocess

import numpy as np
from PIL import Image


REPO_ROOT = "C:\\Users\\lulay\\Desktop\\segformer-multitask\\repo\\segformer-multitask"


def test_train_entrypoint_accepts_config_path(tmp_path):
    config = {
        "data": {
            "dataset_name": "dummy",
            "image_size": [32, 32],
            "num_samples": 4,
        },
        "run": {
            "epochs": 1,
            "max_train_batches": 1,
            "max_eval_batches": 1,
        },
        "logging": {
            "output_dir": str(tmp_path / "logs"),
            "save_checkpoints": False,
            "write_jsonl": False,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        ["uv", "run", "segformer-multitask-train", "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "[summary]" in result.stdout


def _write_rgb(path, value: int) -> None:
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path, value: int) -> None:
    Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(path)


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


def test_train_and_evaluate_cli_for_dual_head_folder_dataset(tmp_path):
    _make_multitask_dataset(tmp_path)
    config = {
        "model": {
            "variant": "mit_b0",
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
        "logging": {
            "output_dir": str(tmp_path / "logs"),
            "save_checkpoints": True,
            "write_jsonl": False,
        },
        "run": {"epochs": 1, "max_train_batches": 1, "max_eval_batches": 1},
    }
    config_path = tmp_path / "dual_head.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    subprocess.run(
        ["uv", "run", "segformer-multitask-train", "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    checkpoint_path = tmp_path / "logs" / "checkpoints" / "latest.pt"
    result = subprocess.run(
        [
            "uv",
            "run",
            "segformer-multitask-evaluate",
            "--checkpoint",
            str(checkpoint_path),
            "--max-batches",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "task_a_pixel_accuracy" in result.stdout
    assert "task_b_pixel_accuracy" in result.stdout

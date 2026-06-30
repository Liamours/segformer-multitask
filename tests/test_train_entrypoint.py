import json
import subprocess
import sys


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
        [sys.executable, "-m", "src.train", "--config", str(config_path)],
        cwd=tmp_path.parent if (tmp_path.parent / "src").exists() else "C:\\Users\\lulay\\Desktop\\segformer-multitask\\repo\\segformer-multitask",
        capture_output=True,
        text=True,
        check=True,
    )

    assert "[summary]" in result.stdout

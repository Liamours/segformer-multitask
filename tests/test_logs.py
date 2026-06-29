import json

from src.configs import ConfigHandler
from src.logs import LogHandler, StepLog


def test_log_handler_writes_artifacts(tmp_path):
    config = ConfigHandler.from_dict(
        {
            "logging": {
                "output_dir": str(tmp_path),
                "run_name": "unit",
                "write_jsonl": True,
            }
        }
    )
    logger = LogHandler(config)
    logger.log_step(
        StepLog(
            epoch=1,
            step=1,
            split="train",
            loss=1.0,
            pixel_accuracy=0.5,
            learning_rate_backbone=1e-5,
            learning_rate_head=1e-4,
        )
    )
    logger.log_epoch(1, "val", {"loss": 1.0, "pixel_accuracy": 0.5})
    logger.write_summary({"train": {"loss": 1.2, "pixel_accuracy": 0.4}, "val": {"loss": 1.0, "pixel_accuracy": 0.5}, "best_metric": 0.5})

    assert (tmp_path / "unit_config.json").exists()
    assert (tmp_path / "unit_metrics.jsonl").exists()
    summary = json.loads((tmp_path / "unit_summary.json").read_text(encoding="utf-8"))
    assert summary["best_metric"] == 0.5

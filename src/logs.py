import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .configs import ExperimentConfig


@dataclass(frozen=True)
class StepLog:
    epoch: int
    step: int
    split: str
    loss: float
    pixel_accuracy: float
    learning_rate_backbone: float
    learning_rate_head: float
    class_0_dice: float | None = None
    class_1_dice: float | None = None
    class_2_dice: float | None = None
    seconds_per_step: float | None = None


class LogHandler:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.logging.output_dir) if config.logging.output_dir else None
        self.run_name = config.logging.run_name
        self.history: list[dict[str, Any]] = []
        self.metrics_path: Path | None = None
        self.summary_path: Path | None = None
        self.preflight_path: Path | None = None
        self.checkpoint_dir: Path | None = None

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"{self.run_name}_config.json").write_text(
                json.dumps(config.to_dict(), indent=2),
                encoding="utf-8",
            )
            if config.logging.write_jsonl:
                self.metrics_path = self.output_dir / f"{self.run_name}_metrics.jsonl"
            self.summary_path = self.output_dir / f"{self.run_name}_summary.json"
            self.preflight_path = self.output_dir / f"{self.run_name}_preflight.json"
            if config.logging.save_checkpoints:
                self.checkpoint_dir = self.output_dir / "checkpoints"
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_message(self, message: str) -> None:
        tqdm.write(message)

    def log_worker_options(self, options: list[int], recommended: int) -> None:
        self.log_message(
            f"[workers] options={options} recommended={recommended} "
            f"train={self.config.data.train_num_workers} val={self.config.data.val_num_workers}"
        )

    def log_step(self, step_log: StepLog) -> None:
        payload = asdict(step_log)
        self.history.append(payload)
        if self.metrics_path is not None:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    def log_epoch(self, epoch: int, split: str, metrics: dict[str, float]) -> None:
        payload = {"epoch": epoch, "split": split, **metrics}
        self.history.append(payload)
        metric_text = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        self.log_message(
            f"[{split}-epoch] epoch={epoch} {metric_text}"
        )
        if self.metrics_path is not None:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        if self.summary_path is not None:
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log_message(
            f"[summary] best_metric={summary.get('best_metric', 0.0):.4f} "
            f"final_train_loss={summary['train']['loss']:.4f} "
            f"final_val_loss={summary['val']['loss']:.4f}"
        )

    def write_preflight(self, report: dict[str, Any]) -> None:
        if self.preflight_path is not None:
            self.preflight_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.log_message(
            "[preflight] "
            f"train_batches={report['effective_train_batches']}/{report['total_train_batches']} "
            f"val_batches={report['effective_val_batches']}/{report['total_val_batches']} "
            f"epochs={report['epochs']} expected_train_steps={report['expected_train_steps']}"
        )

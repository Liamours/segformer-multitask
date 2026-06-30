import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from .backbones import SUPPORTED_MIT_SPECS
from .defaults import DEFAULTS
from .types import TaskClassCounts


@dataclass(frozen=True)
class ModelConfig:
    variant: str
    task_mode: str
    num_classes: int
    task_a_classes: int
    task_b_classes: int
    decoder_dim: int = 256
    decoder_dropout: float = 0.1


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    root_dir: str | None
    train_split: str | None
    val_split: str | None
    image_dir: str
    mask_dir: str
    task_a_mask_dir: str
    task_b_mask_dir: str
    image_suffix: str
    mask_suffix: str
    image_size: tuple[int, int]
    batch_size: int
    eval_batch_size: int
    train_num_workers: int
    val_num_workers: int
    pin_memory: bool
    persistent_workers: bool
    shuffle: bool = True
    drop_last: bool = False
    num_samples: int | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate_backbone: float
    learning_rate_head: float
    betas: tuple[float, float]
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    warmup_iters: int
    warmup_ratio: float
    poly_power: float
    min_lr: float


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str | None = None
    run_name: str = "default"
    log_every_n_steps: int = 10
    write_jsonl: bool = True
    save_checkpoints: bool = True
    checkpoint_metric: str = "pixel_accuracy"
    checkpoint_mode: str = "max"


@dataclass(frozen=True)
class RunConfig:
    seed: int
    epochs: int
    device: str = "cpu"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    run: RunConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def task_class_counts(self) -> int | TaskClassCounts:
        if self.model.task_mode == "single_task":
            return self.model.num_classes
        return TaskClassCounts(self.model.task_a_classes, self.model.task_b_classes)


class ConfigHandler:
    @staticmethod
    def default() -> ExperimentConfig:
        return ExperimentConfig(
            model=ModelConfig(
                variant=DEFAULTS.variant_name,
                task_mode=DEFAULTS.task_mode,
                num_classes=DEFAULTS.num_classes,
                task_a_classes=DEFAULTS.task_a_classes,
                task_b_classes=DEFAULTS.task_b_classes,
            ),
            data=DataConfig(
                dataset_name="dummy",
                root_dir=None,
                train_split=None,
                val_split=None,
                image_dir="images",
                mask_dir="masks",
                task_a_mask_dir="task_a_masks",
                task_b_mask_dir="task_b_masks",
                image_suffix=".png",
                mask_suffix=".png",
                image_size=DEFAULTS.image_size,
                batch_size=DEFAULTS.batch_size,
                eval_batch_size=DEFAULTS.eval_batch_size,
                train_num_workers=DEFAULTS.train_num_workers,
                val_num_workers=DEFAULTS.val_num_workers,
                pin_memory=DEFAULTS.pin_memory,
                persistent_workers=DEFAULTS.persistent_workers,
                num_samples=max(DEFAULTS.batch_size, 8),
            ),
            optimizer=OptimizerConfig(
                name=DEFAULTS.optimizer_name,
                learning_rate_backbone=DEFAULTS.learning_rate_backbone,
                learning_rate_head=DEFAULTS.learning_rate_head,
                betas=DEFAULTS.betas,
                weight_decay=DEFAULTS.weight_decay,
            ),
            scheduler=SchedulerConfig(
                name=DEFAULTS.scheduler_name,
                warmup_iters=DEFAULTS.warmup_iters,
                warmup_ratio=DEFAULTS.warmup_ratio,
                poly_power=DEFAULTS.poly_power,
                min_lr=DEFAULTS.min_lr,
            ),
            logging=LoggingConfig(),
            run=RunConfig(
                seed=DEFAULTS.seed,
                epochs=DEFAULTS.epochs,
                max_train_batches=DEFAULTS.max_train_batches,
                max_eval_batches=DEFAULTS.max_eval_batches,
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        config = cls.default()
        for section_name in ("model", "data", "optimizer", "scheduler", "logging", "run"):
            if section_name not in payload:
                continue
            current = getattr(config, section_name)
            updated = cls._merge_dataclass(current, payload[section_name])
            config = replace(config, **{section_name: updated})
        cls.validate(config)
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def to_json(config: ExperimentConfig, path: str | Path) -> None:
        Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def validate(config: ExperimentConfig) -> None:
        valid_task_modes = {"single_task", "dual_head", "dual_decoder"}
        valid_datasets = {"dummy", "folder"}
        valid_checkpoint_metrics = {"loss", "pixel_accuracy"}
        if config.model.variant not in SUPPORTED_MIT_SPECS:
            raise ValueError(f"Unsupported variant={config.model.variant}.")
        if config.model.task_mode not in valid_task_modes:
            raise ValueError(f"Unsupported task_mode={config.model.task_mode}.")
        if config.data.dataset_name not in valid_datasets:
            raise ValueError(f"Unsupported dataset_name={config.data.dataset_name}.")
        if config.data.dataset_name == "folder" and not config.data.root_dir:
            raise ValueError("data.root_dir is required for dataset_name='folder'.")
        if config.model.task_mode == "single_task" and config.model.num_classes <= 0:
            raise ValueError("model.num_classes must be positive for single_task.")
        if config.model.task_mode != "single_task":
            if config.model.task_a_classes <= 0 or config.model.task_b_classes <= 0:
                raise ValueError("task_a_classes and task_b_classes must be positive for multitask modes.")
        if config.model.decoder_dim <= 0:
            raise ValueError("model.decoder_dim must be positive.")
        if config.model.decoder_dropout < 0.0 or config.model.decoder_dropout >= 1.0:
            raise ValueError("model.decoder_dropout must be in [0.0, 1.0).")
        if config.data.image_size[0] <= 0 or config.data.image_size[1] <= 0:
            raise ValueError("data.image_size must contain positive values.")
        if config.data.batch_size <= 0 or config.data.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive.")
        if config.data.train_num_workers < 0 or config.data.val_num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if config.run.epochs <= 0:
            raise ValueError("run.epochs must be positive.")
        if config.logging.checkpoint_mode not in {"min", "max"}:
            raise ValueError("logging.checkpoint_mode must be 'min' or 'max'.")
        if config.logging.checkpoint_metric not in valid_checkpoint_metrics:
            raise ValueError("logging.checkpoint_metric must be one of: loss, pixel_accuracy.")
        if config.logging.log_every_n_steps <= 0:
            raise ValueError("logging.log_every_n_steps must be positive.")

    @staticmethod
    def _merge_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
        valid_names = {field.name for field in fields(instance)}
        unknown = set(payload) - valid_names
        if unknown:
            unknown_names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown config keys for {type(instance).__name__}: {unknown_names}.")

        normalized = {}
        for key, value in payload.items():
            current = getattr(instance, key)
            if isinstance(current, tuple) and isinstance(value, list):
                normalized[key] = tuple(value)
            else:
                normalized[key] = value
        return replace(instance, **normalized)

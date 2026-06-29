from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .backbones import MixVisionTransformer, get_mit_spec
from .configs import ConfigHandler, DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from .datasets import (
    DummySegmentationDataset,
    MultiTaskSegmentationDataset,
    SingleTaskSegmentationDataset,
    build_folder_samples,
)
from .decoders import SegFormerDecoder
from .defaults import DEFAULTS
from .heads import SegmentationHead
from .logs import LogHandler, StepLog
from .losses import multitask_segmentation_losses, segmentation_loss
from .metrics import pixel_accuracy
from .models import DualDecoderSegFormer, DualHeadSegFormer, SingleTaskSegFormer
from .types import TaskClassCounts
from .utils import list_num_workers_options, recommend_num_workers, set_seed
from .weighting import FixedLossWeighting


def build_model_from_config(config: ModelConfig) -> nn.Module:
    spec = get_mit_spec(config.variant)
    decoder_kwargs = {
        "in_channels": spec.stage_channels,
        "embedding_dim": config.decoder_dim,
        "dropout": config.decoder_dropout,
    }
    backbone = MixVisionTransformer(spec)

    if config.task_mode == "single_task":
        decoder = SegFormerDecoder(**decoder_kwargs)
        return SingleTaskSegFormer(backbone, decoder, SegmentationHead(decoder.output_dim, config.num_classes))

    if config.task_mode == "dual_head":
        decoder = SegFormerDecoder(**decoder_kwargs)
        return DualHeadSegFormer(
            backbone,
            decoder,
            SegmentationHead(decoder.output_dim, config.task_a_classes),
            SegmentationHead(decoder.output_dim, config.task_b_classes),
        )

    if config.task_mode == "dual_decoder":
        decoder_a = SegFormerDecoder(**decoder_kwargs)
        decoder_b = SegFormerDecoder(**decoder_kwargs)
        return DualDecoderSegFormer(
            backbone,
            decoder_a,
            decoder_b,
            SegmentationHead(decoder_a.output_dim, config.task_a_classes),
            SegmentationHead(decoder_b.output_dim, config.task_b_classes),
        )

    raise ValueError(f"Unsupported task_mode={config.task_mode}.")


def build_model(
    variant: str = "mit_b0",
    task_mode: str = "single_task",
    num_classes: int | TaskClassCounts = 4,
) -> nn.Module:
    if isinstance(num_classes, TaskClassCounts):
        config = ModelConfig(
            variant=variant,
            task_mode=task_mode,
            num_classes=num_classes.task_a,
            task_a_classes=num_classes.task_a,
            task_b_classes=num_classes.task_b,
        )
    else:
        config = ModelConfig(
            variant=variant,
            task_mode=task_mode,
            num_classes=num_classes,
            task_a_classes=num_classes,
            task_b_classes=num_classes,
        )
    return build_model_from_config(config)


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> AdamW:
    if config.name != "adamw":
        raise ValueError(f"Unsupported optimizer={config.name}.")

    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    return AdamW(
        [
            {"params": backbone_params, "lr": config.learning_rate_backbone},
            {"params": head_params, "lr": config.learning_rate_head},
        ],
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: AdamW, total_steps: int, config: SchedulerConfig) -> LambdaLR:
    if config.name != "poly":
        raise ValueError(f"Unsupported scheduler={config.name}.")

    warmup_iters = max(1, config.warmup_iters)
    total_steps = max(total_steps, warmup_iters)

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if current_step <= warmup_iters:
            progress = current_step / warmup_iters
            return config.warmup_ratio + progress * (1.0 - config.warmup_ratio)
        progress = (current_step - warmup_iters) / max(1, total_steps - warmup_iters)
        return max(config.min_lr, (1.0 - progress) ** config.poly_power)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_datasets(
    config: DataConfig,
    task_mode: str,
    task_class_counts: int | TaskClassCounts,
) -> tuple[Dataset, Dataset]:
    multitask = task_mode != "single_task"

    if config.dataset_name == "dummy":
        length = config.num_samples or max(config.batch_size * 2, 8)
        train_dataset = DummySegmentationDataset(
            length=length,
            image_size=config.image_size,
            num_classes=task_class_counts,
            multitask=multitask,
            seed=42,
        )
        val_dataset = DummySegmentationDataset(
            length=max(config.eval_batch_size * 2, 4),
            image_size=config.image_size,
            num_classes=task_class_counts,
            multitask=multitask,
            seed=1042,
        )
        return train_dataset, val_dataset

    train_samples = build_folder_samples(
        root_dir=config.root_dir,
        split=config.train_split,
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        task_a_mask_dir=config.task_a_mask_dir,
        task_b_mask_dir=config.task_b_mask_dir,
        image_suffix=config.image_suffix,
        mask_suffix=config.mask_suffix,
        multitask=multitask,
    )
    val_samples = build_folder_samples(
        root_dir=config.root_dir,
        split=config.val_split,
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        task_a_mask_dir=config.task_a_mask_dir,
        task_b_mask_dir=config.task_b_mask_dir,
        image_suffix=config.image_suffix,
        mask_suffix=config.mask_suffix,
        multitask=multitask,
    )
    dataset_type = MultiTaskSegmentationDataset if multitask else SingleTaskSegmentationDataset
    return dataset_type(train_samples, config.image_size), dataset_type(val_samples, config.image_size)


def build_dataloaders(config: DataConfig, task_mode: str, task_class_counts: int | TaskClassCounts) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(config, task_mode, task_class_counts)
    loader_kwargs = {
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.train_num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        num_workers=config.train_num_workers,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.val_num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.val_num_workers > 0,
    )
    return train_loader, val_loader


class Trainer:
    def __init__(self, config: ExperimentConfig, logger: LogHandler | None = None) -> None:
        self.config = config
        self.device = torch.device(config.run.device)
        self.logger = logger or LogHandler(config)
        self.task_class_counts = config.task_class_counts()
        self.model = build_model_from_config(config.model).to(self.device)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.train_loader, self.val_loader = build_dataloaders(config.data, config.model.task_mode, self.task_class_counts)
        total_steps = max(1, len(self.train_loader) * config.run.epochs)
        self.scheduler = build_scheduler(self.optimizer, total_steps, config.scheduler)
        self.weighting = FixedLossWeighting({"task_a": 1.0, "task_b": 1.0})
        self.best_metric = float("-inf") if config.logging.checkpoint_mode == "max" else float("inf")
        self.best_checkpoint_path: Path | None = None
        self.worker_options = list_num_workers_options()
        self.recommended_workers = recommend_num_workers(config.data.batch_size)
        self.logger.log_worker_options(self.worker_options, self.recommended_workers)

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _compute_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._move_batch(batch)
        outputs = self.model(batch["image"])
        if self.config.model.task_mode == "single_task":
            loss = segmentation_loss(outputs["logits"], batch["mask"])
            accuracy = pixel_accuracy(outputs["logits"], batch["mask"])
            return loss, accuracy

        losses = multitask_segmentation_losses(
            outputs["task_a_logits"],
            batch["task_a_mask"],
            outputs["task_b_logits"],
            batch["task_b_mask"],
        )
        loss = self.weighting.reduce(losses)
        accuracy = pixel_accuracy(outputs["task_a_logits"], batch["task_a_mask"])
        return loss, accuracy

    def _run_epoch(self, loader: DataLoader, epoch: int, split: str, max_batches: int | None) -> dict[str, float]:
        is_train = split == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            with torch.set_grad_enabled(is_train):
                loss, accuracy = self._compute_step(batch)
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

            loss_value = float(loss.detach().cpu().item())
            accuracy_value = float(accuracy.detach().cpu().item())
            total_loss += loss_value
            total_accuracy += accuracy_value
            num_batches += 1

            if is_train and batch_index % self.config.logging.log_every_n_steps == 0:
                self.logger.log_step(
                    StepLog(
                        epoch=epoch,
                        step=batch_index,
                        split=split,
                        loss=loss_value,
                        pixel_accuracy=accuracy_value,
                        learning_rate_backbone=float(self.optimizer.param_groups[0]["lr"]),
                        learning_rate_head=float(self.optimizer.param_groups[1]["lr"]),
                    )
                )

        if num_batches == 0:
            raise ValueError(f"No batches were processed for split={split}.")

        metrics = {
            "loss": total_loss / num_batches,
            "pixel_accuracy": total_accuracy / num_batches,
        }
        self.logger.log_epoch(epoch, split, metrics)
        return metrics

    def _is_better(self, metric: float) -> bool:
        if self.config.logging.checkpoint_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def _save_checkpoint(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float], is_best: bool) -> None:
        if self.logger.checkpoint_dir is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        latest_path = self.logger.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = self.logger.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path

    def fit(self) -> dict[str, object]:
        set_seed(self.config.run.seed)
        final_train_metrics = {"loss": 0.0, "pixel_accuracy": 0.0}
        final_val_metrics = {"loss": 0.0, "pixel_accuracy": 0.0}

        for epoch in range(1, self.config.run.epochs + 1):
            final_train_metrics = self._run_epoch(
                self.train_loader,
                epoch=epoch,
                split="train",
                max_batches=self.config.run.max_train_batches,
            )
            final_val_metrics = self._run_epoch(
                self.val_loader,
                epoch=epoch,
                split="val",
                max_batches=self.config.run.max_eval_batches,
            )
            metric_name = self.config.logging.checkpoint_metric
            metric_value = float(final_val_metrics[metric_name])
            is_best = self._is_better(metric_value)
            if is_best:
                self.best_metric = metric_value
            self._save_checkpoint(epoch, final_train_metrics, final_val_metrics, is_best)

        summary = {
            "train": final_train_metrics,
            "val": final_val_metrics,
            "best_metric": self.best_metric,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "num_workers_options": self.worker_options,
            "recommended_num_workers": self.recommended_workers,
        }
        self.logger.write_summary(summary)
        return summary


def run_smoke_test(
    variant: str = DEFAULTS.variant_name,
    task_mode: str = DEFAULTS.task_mode,
    num_classes: int | TaskClassCounts = DEFAULTS.num_classes,
    image_size: tuple[int, int] = DEFAULTS.image_size,
    steps: int = 1,
) -> dict[str, float]:
    if isinstance(num_classes, TaskClassCounts):
        task_a_classes = num_classes.task_a
        task_b_classes = num_classes.task_b
        base_num_classes = num_classes.task_a
    else:
        task_a_classes = num_classes
        task_b_classes = num_classes
        base_num_classes = num_classes

    config = ConfigHandler.from_dict(
        {
            "model": {
                "variant": variant,
                "task_mode": task_mode,
                "num_classes": base_num_classes,
                "task_a_classes": task_a_classes,
                "task_b_classes": task_b_classes,
            },
            "data": {
                "dataset_name": "dummy",
                "image_size": list(image_size),
                "num_samples": max(steps * DEFAULTS.batch_size, 4),
            },
            "run": {
                "epochs": 1,
                "max_train_batches": steps,
                "max_eval_batches": 1,
            },
            "logging": {
                "write_jsonl": False,
                "save_checkpoints": False,
                "log_every_n_steps": 1,
            },
        }
    )
    summary = Trainer(config).fit()
    return summary["val"]


def load_config(path: str | None = None) -> ExperimentConfig:
    if path is None:
        return ConfigHandler.default()
    return ConfigHandler.from_json(path)


def main() -> None:
    config = load_config()
    result = Trainer(config).fit()
    print(asdict(DEFAULTS))
    print(result)


if __name__ == "__main__":
    main()

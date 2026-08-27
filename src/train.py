import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .backbones import HuggingFacePretrainedMiT, MixVisionTransformer, get_mit_spec
from .configs import ConfigHandler, DataConfig, ExperimentConfig, LossConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from .datasets import (
    DummySegmentationDataset,
    MultiTaskSegmentationDataset,
    SingleTaskSegmentationDataset,
    build_folder_samples,
)
from .decoders import SegFormerDecoder, SegFormerFusionHead, SegFormerProjectionTrunk
from .defaults import DEFAULTS
from .heads import SegmentationHead
from .logs import LogHandler, StepLog
from .losses import multitask_segmentation_losses, segmentation_loss
from .metrics import segmentation_scores
from .models import DualDecoderSegFormer, DualFuseSegFormer, DualHeadSegFormer, SingleTaskSegFormer
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

    pretrained_report: dict[str, object] | None = None
    if config.pretrained_hf_name is not None:
        backbone: nn.Module = HuggingFacePretrainedMiT(config.pretrained_hf_name, spec)
        pretrained_report = _build_hf_pretrained_report(backbone)
    else:
        backbone = MixVisionTransformer(spec)

    if config.task_mode == "single_task":
        decoder = SegFormerDecoder(**decoder_kwargs)
        model = SingleTaskSegFormer(backbone, decoder, SegmentationHead(decoder.output_dim, config.num_classes))
    elif config.task_mode == "dual_head":
        decoder = SegFormerDecoder(**decoder_kwargs)
        model = DualHeadSegFormer(
            backbone,
            decoder,
            SegmentationHead(decoder.output_dim, config.task_a_classes),
            SegmentationHead(decoder.output_dim, config.task_b_classes),
        )
    elif config.task_mode == "dual_decoder":
        decoder_a = SegFormerDecoder(**decoder_kwargs)
        decoder_b = SegFormerDecoder(**decoder_kwargs)
        model = DualDecoderSegFormer(
            backbone,
            decoder_a,
            decoder_b,
            SegmentationHead(decoder_a.output_dim, config.task_a_classes),
            SegmentationHead(decoder_b.output_dim, config.task_b_classes),
        )
    elif config.task_mode == "dual_fuse":
        trunk = SegFormerProjectionTrunk(in_channels=spec.stage_channels, embedding_dim=config.decoder_dim)
        fuse_a = SegFormerFusionHead(trunk.output_dim, output_dim=config.decoder_dim, dropout=config.decoder_dropout)
        fuse_b = SegFormerFusionHead(trunk.output_dim, output_dim=config.decoder_dim, dropout=config.decoder_dropout)
        model = DualFuseSegFormer(
            backbone,
            trunk,
            fuse_a,
            fuse_b,
            SegmentationHead(fuse_a.output_dim, config.task_a_classes),
            SegmentationHead(fuse_b.output_dim, config.task_b_classes),
        )
    else:
        raise ValueError(f"Unsupported task_mode={config.task_mode}.")

    _apply_model_loading_policy(model, config, pretrained_report)
    return model


def _build_hf_pretrained_report(backbone: HuggingFacePretrainedMiT) -> dict[str, object]:
    missing_keys = sorted(backbone.loading_info.get("missing_keys", []))
    unexpected_keys = sorted(backbone.loading_info.get("unexpected_keys", []))
    mismatched_keys = sorted(backbone.loading_info.get("mismatched_keys", []))
    model_tensors = len(backbone.model.state_dict())
    loaded_tensors = model_tensors - len(missing_keys) - len(mismatched_keys)
    weight = backbone.patch_embedding_weight().detach().float().cpu()
    return {
        "checkpoint_source": backbone.hf_name,
        "model_tensors": model_tensors,
        "loaded_tensors": loaded_tensors,
        "missing_tensors": len(missing_keys),
        "unexpected_tensors": len(unexpected_keys),
        "mismatched_tensors": len(mismatched_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
        "stage1_patch_embed_weight_name": "stages.0.patch_embeddings.proj.weight",
        "stage1_patch_embed_mean": float(weight.mean().item()),
        "stage1_patch_embed_std": float(weight.std().item()),
        "stage1_patch_embed_slice_3x3": weight[0, 0, :3, :3].tolist(),
    }


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Pretrained checkpoint must be a state_dict or a dict containing one.")
    for key in ("state_dict", "model_state_dict", "model", "backbone"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not find tensor state_dict in pretrained checkpoint.")


def _strip_prefixes(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    return cleaned


def load_backbone_pretrained(model: nn.Module, checkpoint_path: str | Path) -> dict[str, object]:
    if not hasattr(model, "backbone"):
        raise ValueError("Model does not expose a backbone attribute.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_prefixes(
        state_dict,
        (
            "module.backbone.",
            "model.backbone.",
            "backbone.",
            "module.",
        ),
    )
    incompatible = model.backbone.load_state_dict(state_dict, strict=True)
    return {
        "path": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _apply_model_loading_policy(
    model: nn.Module,
    config: ModelConfig,
    pretrained_report: dict[str, object] | None = None,
) -> None:
    if config.pretrained_path is not None:
        pretrained_report = load_backbone_pretrained(model, config.pretrained_path)
    if config.freeze_backbone:
        model.freeze_backbone()
    setattr(model, "pretrained_report", pretrained_report)


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


def _segmentation_loss_from_config(logits: torch.Tensor, target: torch.Tensor, config: LossConfig) -> torch.Tensor:
    return segmentation_loss(
        logits,
        target,
        ignore_index=config.ignore_index,
        name=config.name,
        ce_weight=config.ce_weight,
        dice_weight=config.dice_weight,
    )


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
        paired_views=config.paired_views,
        anterior_suffix=config.anterior_suffix,
        posterior_suffix=config.posterior_suffix,
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
        paired_views=config.paired_views,
        anterior_suffix=config.anterior_suffix,
        posterior_suffix=config.posterior_suffix,
    )
    dataset_type = MultiTaskSegmentationDataset if multitask else SingleTaskSegmentationDataset
    return (
        dataset_type(train_samples, config.image_size, config.normalize_mean, config.normalize_std),
        dataset_type(val_samples, config.image_size, config.normalize_mean, config.normalize_std),
    )


def build_eval_dataset(
    config: DataConfig,
    task_mode: str,
    task_class_counts: int | TaskClassCounts,
    split_name: str,
) -> Dataset:
    multitask = task_mode != "single_task"
    if split_name == "val":
        split = config.val_split
    elif split_name == "test":
        split = config.test_split
        if split is None:
            raise ValueError("data.test_split is required for split='test'.")
    else:
        raise ValueError("split must be 'val' or 'test'.")

    if config.dataset_name == "dummy":
        return DummySegmentationDataset(
            length=max(config.eval_batch_size * 2, 4),
            image_size=config.image_size,
            num_classes=task_class_counts,
            multitask=multitask,
            seed=1042 if split_name == "val" else 2042,
        )

    samples = build_folder_samples(
        root_dir=config.root_dir,
        split=split,
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        task_a_mask_dir=config.task_a_mask_dir,
        task_b_mask_dir=config.task_b_mask_dir,
        image_suffix=config.image_suffix,
        mask_suffix=config.mask_suffix,
        multitask=multitask,
        paired_views=config.paired_views,
        anterior_suffix=config.anterior_suffix,
        posterior_suffix=config.posterior_suffix,
    )
    dataset_type = MultiTaskSegmentationDataset if multitask else SingleTaskSegmentationDataset
    return dataset_type(samples, config.image_size, config.normalize_mean, config.normalize_std)


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
        self._configure_runtime()
        self.logger = logger or LogHandler(config)
        self._mask_values_verified = False
        self.task_class_counts = config.task_class_counts()
        set_seed(config.run.seed)
        self.model = build_model_from_config(config.model).to(self.device)
        if self._use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self._log_optimizer_groups()
        self.train_loader, self.val_loader = build_dataloaders(config.data, config.model.task_mode, self.task_class_counts)
        optimizer_steps_per_epoch = math.ceil(len(self.train_loader) / config.run.gradient_accumulation_steps)
        total_steps = max(1, optimizer_steps_per_epoch * config.run.epochs)
        self.scheduler = build_scheduler(self.optimizer, total_steps, config.scheduler)
        self.weighting = FixedLossWeighting(
            {"task_a": config.loss.task_a_weight, "task_b": config.loss.task_b_weight}
        )
        self.best_metric = float("-inf") if config.logging.checkpoint_mode == "max" else float("inf")
        self.best_checkpoint_path: Path | None = None
        self.start_epoch = 1
        self.worker_options = list_num_workers_options()
        self.recommended_workers = recommend_num_workers(config.data.batch_size)
        self.logger.log_worker_options(self.worker_options, self.recommended_workers)
        if config.run.resume_checkpoint is not None:
            self.load_training_state(config.run.resume_checkpoint)
        self.preflight_report = self._build_preflight_report()
        self.logger.write_preflight(self.preflight_report)

    def _configure_runtime(self) -> None:
        cuda_device = self.config.run.device.startswith("cuda") and torch.cuda.is_available()
        self._use_amp = cuda_device and self.config.run.amp
        self._use_channels_last = cuda_device and self.config.run.channels_last
        if cuda_device:
            torch.backends.cuda.matmul.allow_tf32 = self.config.run.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.config.run.allow_tf32
            torch.backends.cudnn.benchmark = self.config.run.cudnn_benchmark
        if self.config.run.amp_dtype == "bf16":
            if cuda_device and torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
            else:
                self._amp_dtype = torch.float16
        else:
            self._amp_dtype = torch.float16
        self._grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp and self._amp_dtype == torch.float16)

    def _autocast(self):
        return torch.amp.autocast(
            device_type="cuda",
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        )

    def _log_optimizer_groups(self) -> None:
        for group_index, group in enumerate(self.optimizer.param_groups):
            tensors = list(group["params"])
            parameter_count = sum(parameter.numel() for parameter in tensors)
            self.logger.log_message(
                f"[optimizer-group] group={group_index} lr={group['lr']:.8f} "
                f"tensors={len(tensors)} params={parameter_count}"
            )

    def load_training_state(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
        best_path = checkpoint.get("best_checkpoint_path")
        self.best_checkpoint_path = Path(best_path) if best_path else self.best_checkpoint_path
        if self.start_epoch > self.config.run.epochs:
            raise ValueError(
                f"resume checkpoint epoch={checkpoint['epoch']} is already >= configured epochs={self.config.run.epochs}."
            )

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {key: value.to(self.device, non_blocking=self.config.data.pin_memory) for key, value in batch.items()}
        if self._use_channels_last:
            moved["image"] = moved["image"].contiguous(memory_format=torch.channels_last)
        return moved

    def _build_preflight_report(self) -> dict[str, object]:
        total_train_batches = len(self.train_loader)
        total_val_batches = len(self.val_loader)
        effective_train_batches = min(total_train_batches, self.config.run.max_train_batches or total_train_batches)
        effective_val_batches = min(total_val_batches, self.config.run.max_eval_batches or total_val_batches)
        limited_train = effective_train_batches != total_train_batches
        limited_val = effective_val_batches != total_val_batches
        if (limited_train or limited_val) and not self.config.run.allow_limited_batches:
            raise ValueError("Internal error: limited batches reached trainer without explicit permission.")
        return {
            "dataset_name": self.config.data.dataset_name,
            "root_dir": self.config.data.root_dir,
            "task_mode": self.config.model.task_mode,
            "variant": self.config.model.variant,
            "epochs": self.config.run.epochs,
            "batch_size": self.config.data.batch_size,
            "eval_batch_size": self.config.data.eval_batch_size,
            "total_train_batches": total_train_batches,
            "total_val_batches": total_val_batches,
            "effective_train_batches": effective_train_batches,
            "effective_val_batches": effective_val_batches,
            "max_train_batches": self.config.run.max_train_batches,
            "max_eval_batches": self.config.run.max_eval_batches,
            "allow_limited_batches": self.config.run.allow_limited_batches,
            "expected_train_steps": effective_train_batches * self.config.run.epochs,
            "paired_views": self.config.data.paired_views,
            "image_size": self.config.data.image_size,
            "device": self.config.run.device,
            "amp": self.config.run.amp,
            "amp_dtype": self.config.run.amp_dtype,
            "effective_amp_dtype": str(self._amp_dtype).replace("torch.", ""),
            "channels_last": self.config.run.channels_last,
            "allow_tf32": self.config.run.allow_tf32,
            "cudnn_benchmark": self.config.run.cudnn_benchmark,
            "gradient_accumulation_steps": self.config.run.gradient_accumulation_steps,
            "optimizer_steps_per_epoch": math.ceil(effective_train_batches / self.config.run.gradient_accumulation_steps),
            "learning_rate_backbone": self.config.optimizer.learning_rate_backbone,
            "learning_rate_head": self.config.optimizer.learning_rate_head,
            "pretrained_path": self.config.model.pretrained_path,
            "pretrained_hf_name": self.config.model.pretrained_hf_name,
            "pretrained_report": getattr(self.model, "pretrained_report", None),
            "freeze_backbone": self.config.model.freeze_backbone,
            "checkpoint_metric": self.config.logging.checkpoint_metric,
            "checkpoint_mode": self.config.logging.checkpoint_mode,
            "early_stopping_patience": self.config.run.early_stopping_patience,
            "early_stopping_metric": self.config.run.early_stopping_metric,
        }

    def _compute_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = self._move_batch(batch)
        self._verify_mask_values(batch)
        with self._autocast():
            outputs = self.model(batch["image"])
            return self._compute_loss_and_metrics(outputs, batch)

    def _verify_mask_values(self, batch: dict[str, torch.Tensor]) -> None:
        if self._mask_values_verified:
            return
        if self.config.model.task_mode == "single_task":
            masks = {"mask": (batch["mask"], self.config.model.num_classes)}
        else:
            masks = {
                "task_a_mask": (batch["task_a_mask"], self.config.model.task_a_classes),
                "task_b_mask": (batch["task_b_mask"], self.config.model.task_b_classes),
            }
        for name, (mask, num_classes) in masks.items():
            unique_values = [int(value) for value in torch.unique(mask).detach().cpu().tolist()]
            self.logger.log_message(f"[mask-unique] {name}={unique_values}")
            invalid_values = sorted(set(unique_values) - set(range(num_classes)))
            if invalid_values:
                raise ValueError(
                    f"{name} contains invalid labels {invalid_values}; expected only 0..{num_classes - 1}."
                )
        self._mask_values_verified = True

    def _compute_loss_and_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.model.task_mode == "single_task":
            loss = _segmentation_loss_from_config(outputs["logits"], batch["mask"], self.config.loss)
            metrics = segmentation_scores(outputs["logits"], batch["mask"], self.config.model.num_classes)
            metrics["loss"] = loss.detach()
            return loss, metrics

        losses = multitask_segmentation_losses(
            outputs["task_a_logits"],
            batch["task_a_mask"],
            outputs["task_b_logits"],
            batch["task_b_mask"],
            ignore_index=self.config.loss.ignore_index,
            name=self.config.loss.name,
            ce_weight=self.config.loss.ce_weight,
            dice_weight=self.config.loss.dice_weight,
        )
        loss = self.weighting.reduce(losses)
        task_a_metrics = segmentation_scores(
            outputs["task_a_logits"],
            batch["task_a_mask"],
            self.config.model.task_a_classes,
        )
        task_b_metrics = segmentation_scores(
            outputs["task_b_logits"],
            batch["task_b_mask"],
            self.config.model.task_b_classes,
        )
        metrics = {
            "loss": loss.detach(),
            "task_a_loss": losses["task_a"].detach(),
            "task_b_loss": losses["task_b"].detach(),
        }
        for key, value in task_a_metrics.items():
            metrics[f"task_a_{key}"] = value
        for key, value in task_b_metrics.items():
            metrics[f"task_b_{key}"] = value
        metrics["pixel_accuracy"] = (task_a_metrics["pixel_accuracy"] + task_b_metrics["pixel_accuracy"]) / 2.0
        metrics["mean_iou"] = (task_a_metrics["mean_iou"] + task_b_metrics["mean_iou"]) / 2.0
        metrics["mean_dice"] = (task_a_metrics["mean_dice"] + task_b_metrics["mean_dice"]) / 2.0
        metrics["foreground_mean_iou"] = (task_a_metrics["foreground_mean_iou"] + task_b_metrics["foreground_mean_iou"]) / 2.0
        metrics["foreground_mean_dice"] = (task_a_metrics["foreground_mean_dice"] + task_b_metrics["foreground_mean_dice"]) / 2.0
        return loss, metrics

    def _run_epoch(self, loader: DataLoader, epoch: int, split: str, max_batches: int | None) -> dict[str, float]:
        is_train = split == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        metric_totals: dict[str, float] = {}
        num_batches = 0
        effective_batches = min(len(loader), max_batches or len(loader))
        accumulation_steps = self.config.run.gradient_accumulation_steps
        if is_train:
            self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            enumerate(loader, start=1),
            total=effective_batches,
            desc=f"epoch {epoch} {'train' if is_train else 'val'}",
            leave=False,
        )
        for batch_index, batch in progress:
            if max_batches is not None and batch_index > max_batches:
                break

            if is_train and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            step_started = time.perf_counter()
            with torch.set_grad_enabled(is_train):
                loss, batch_metrics = self._compute_step(batch)
                if is_train:
                    scaled_loss = loss / accumulation_steps
                    if self._grad_scaler.is_enabled():
                        self._grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    should_step = batch_index % accumulation_steps == 0 or batch_index == effective_batches
                    if should_step:
                        if self._grad_scaler.is_enabled():
                            self._grad_scaler.step(self.optimizer)
                            self._grad_scaler.update()
                        else:
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

            if is_train and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            seconds_per_step = time.perf_counter() - step_started

            batch_metric_values = {
                key: float(value.detach().cpu().item())
                for key, value in batch_metrics.items()
            }
            if is_train:
                batch_metric_values["seconds_per_step"] = seconds_per_step
            for key, value in batch_metric_values.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value
            num_batches += 1

            progress.set_postfix(
                loss=f"{batch_metric_values['loss']:.4f}",
                acc=f"{batch_metric_values['pixel_accuracy']:.4f}",
            )

            if is_train and batch_index % self.config.logging.log_every_n_steps == 0:
                self.logger.log_step(
                    StepLog(
                        epoch=epoch,
                        step=batch_index,
                        split=split,
                        loss=batch_metric_values["loss"],
                        pixel_accuracy=batch_metric_values["pixel_accuracy"],
                        learning_rate_backbone=float(self.optimizer.param_groups[0]["lr"]),
                        learning_rate_head=float(self.optimizer.param_groups[1]["lr"]),
                        class_0_dice=batch_metric_values.get("class_0_dice"),
                        class_1_dice=batch_metric_values.get("class_1_dice"),
                        class_2_dice=batch_metric_values.get("class_2_dice"),
                        seconds_per_step=batch_metric_values["seconds_per_step"],
                    )
                )

        progress.close()

        if num_batches == 0:
            raise ValueError(f"No batches were processed for split={split}.")

        metrics = {key: value / num_batches for key, value in metric_totals.items()}
        self.logger.log_epoch(epoch, split, metrics)
        return metrics

    def _is_better(self, metric: float) -> bool:
        if self.config.logging.checkpoint_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def _early_stopping_is_better(self, metric: float, best_metric: float) -> bool:
        min_delta = self.config.run.early_stopping_min_delta
        if self.config.run.early_stopping_mode == "max":
            return metric > best_metric + min_delta
        return metric < best_metric - min_delta

    def _save_checkpoint(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float], is_best: bool) -> None:
        if self.logger.checkpoint_dir is None:
            return

        best_path = self.logger.checkpoint_dir / "best.pt" if is_best else self.best_checkpoint_path
        if is_best:
            self.best_checkpoint_path = best_path
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_metric": self.best_metric,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
        }
        latest_path = self.logger.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        if self.config.logging.save_epoch_checkpoints:
            torch.save(checkpoint, self.logger.checkpoint_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)

    def fit(self) -> dict[str, object]:
        final_train_metrics = {"loss": 0.0, "pixel_accuracy": 0.0}
        final_val_metrics = {"loss": 0.0, "pixel_accuracy": 0.0}
        early_metric_name = self.config.run.early_stopping_metric or self.config.logging.checkpoint_metric
        early_best = float("-inf") if self.config.run.early_stopping_mode == "max" else float("inf")
        early_bad_epochs = 0
        stopped_epoch: int | None = None

        epoch_progress = tqdm(
            range(self.start_epoch, self.config.run.epochs + 1),
            total=self.config.run.epochs,
            initial=self.start_epoch - 1,
            desc="epochs",
        )
        for epoch in epoch_progress:
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
            epoch_progress.set_postfix(
                train_loss=f"{final_train_metrics['loss']:.4f}",
                val_loss=f"{final_val_metrics['loss']:.4f}",
                best=f"{self.best_metric:.4f}",
            )
            if self.config.run.early_stopping_patience is not None:
                early_metric_value = float(final_val_metrics[early_metric_name])
                if self._early_stopping_is_better(early_metric_value, early_best):
                    early_best = early_metric_value
                    early_bad_epochs = 0
                else:
                    early_bad_epochs += 1
                if early_bad_epochs >= self.config.run.early_stopping_patience:
                    stopped_epoch = epoch
                    self.logger.log_message(
                        f"[early-stopping] epoch={epoch} metric={early_metric_name} "
                        f"bad_epochs={early_bad_epochs}"
                    )
                    break

        epoch_progress.close()

        summary = {
            "train": final_train_metrics,
            "val": final_val_metrics,
            "best_metric": self.best_metric,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "start_epoch": self.start_epoch,
            "completed_epochs": stopped_epoch or self.config.run.epochs,
            "early_stopped_epoch": stopped_epoch,
            "num_workers_options": self.worker_options,
            "recommended_num_workers": self.recommended_workers,
            "preflight": self.preflight_report,
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
                "allow_limited_batches": True,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate SegFormer multitask models.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume optimizer/scheduler/model state from checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.resume_checkpoint is not None:
        payload = config.to_dict()
        payload["run"]["resume_checkpoint"] = args.resume_checkpoint
        config = ConfigHandler.from_dict(payload)
    result = Trainer(config).fit()
    print(result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.configs import ConfigHandler, ExperimentConfig  # noqa: E402
from src.train import build_model_from_config  # noqa: E402


@dataclass(frozen=True)
class LoadedSegFormer:
    model: nn.Module
    config: ExperimentConfig
    device: torch.device
    task_names: tuple[str, ...]

    def output_to_tasks(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.config.model.task_mode == "single_task":
            return {self.task_names[0]: outputs["logits"]}
        return {
            self.task_names[0]: outputs["task_a_logits"],
            self.task_names[1]: outputs["task_b_logits"],
        }


def load_segformer_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    task_names: tuple[str, ...] | None = None,
) -> LoadedSegFormer:
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    config = _load_config(checkpoint, config_path)
    model = build_model_from_config(config.model).to(target_device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    names = task_names or _default_task_names(config)
    return LoadedSegFormer(model=model, config=config, device=target_device, task_names=names)


def _load_config(checkpoint: Any, config_path: str | Path | None) -> ExperimentConfig:
    if config_path is not None:
        return ConfigHandler.from_json(config_path)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        return ConfigHandler.from_dict(checkpoint["config"])
    raise ValueError("Checkpoint must contain a saved config unless config_path is provided.")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise ValueError("Checkpoint does not contain a valid model state dict.")


def _default_task_names(config: ExperimentConfig) -> tuple[str, ...]:
    if config.model.task_mode == "single_task":
        return ("lesion",)
    return ("lesion", "bone")


def load_config_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .configs import ConfigHandler, ExperimentConfig
from .train import Trainer


def load_checkpoint_config(checkpoint_path: str | Path) -> ExperimentConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a config: {checkpoint_path}")
    return ConfigHandler.from_dict(checkpoint["config"])


def evaluate_config(
    config: ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    trainer = Trainer(config)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    return trainer._run_epoch(
        trainer.val_loader,
        epoch=0,
        split="eval",
        max_batches=max_batches if max_batches is not None else config.run.max_eval_batches,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    config = ConfigHandler.from_json(config_path) if config_path is not None else load_checkpoint_config(checkpoint_path)
    return evaluate_config(config, checkpoint_path, max_batches=max_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SegFormer multitask checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint saved by src.train.")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON. Defaults to checkpoint config.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for evaluation batches.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(args.checkpoint, args.config, args.max_batches)
    payload: dict[str, Any] = {"checkpoint": args.checkpoint, "metrics": metrics}
    if args.output is not None:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

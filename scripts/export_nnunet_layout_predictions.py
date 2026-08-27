"""Export SegFormer test predictions as per-view PNGs in nnU-Net's evaluation layout.

The paired canvas is split back into anterior (left half) and posterior (right half)
so the parent project's `nnunetv2/evaluation/evaluate_multitask_predictions.py`
can score SegFormer under the same per-case protocol as the nnU-Net models.

Layout written: <output>/<task>/<view>/<case_id>.png
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import ConfigHandler
from src.train import Trainer, build_eval_dataset

VIEWS = ("anterior", "posterior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint saved by src.train.")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--output", required=True, help="Root folder for the prediction tree.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run under the training autocast dtype instead of fp32. Changes small-class predictions.",
    )
    return parser.parse_args()


def _task_logit_keys(task_mode: str) -> dict[str, str]:
    if task_mode == "single_task":
        return {"lesion": "logits"}
    return {"lesion": "task_a_logits", "bone": "task_b_logits"}


def main() -> None:
    args = parse_args()
    config = ConfigHandler.from_json(args.config)
    trainer = Trainer(config)
    checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()

    dataset = build_eval_dataset(
        config.data, config.model.task_mode, config.task_class_counts(), args.split
    )
    case_ids = [
        line.strip()
        for line in (Path(config.data.root_dir) / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(case_ids) != len(dataset):
        raise ValueError(f"Split list has {len(case_ids)} cases, dataset has {len(dataset)}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    logit_keys = _task_logit_keys(config.model.task_mode)
    output_root = Path(args.output)
    for task in logit_keys:
        for view in VIEWS:
            (output_root / task / view).mkdir(parents=True, exist_ok=True)

    written = 0
    autocast = trainer._autocast() if args.amp else nullcontext()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.split} inference"):
            image = batch["image"].to(trainer.device, non_blocking=True)
            with autocast:
                outputs = trainer.model(image)
            batch_cases = case_ids[written : written + image.shape[0]]
            for task, key in logit_keys.items():
                prediction = outputs[key].argmax(dim=1).to(torch.uint8).cpu().numpy()
                half = prediction.shape[-1] // 2
                for offset, case_id in enumerate(batch_cases):
                    canvas = prediction[offset]
                    for view, view_mask in zip(VIEWS, (canvas[:, :half], canvas[:, half:])):
                        Image.fromarray(np.ascontiguousarray(view_mask), mode="L").save(
                            output_root / task / view / f"{case_id}.png"
                        )
            written += image.shape[0]

    print(f"wrote {written} cases x {len(VIEWS)} views x {len(logit_keys)} task(s) to {output_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from infer import run_inference
from model import load_segformer_checkpoint
from pair_loader import build_pair
from palette import BONE_LABELS
from postprocess import PostprocessResult, postprocess_logits
from preprocess import preprocess_pair
from transforms import InferenceTransformConfig


def run_segmentation(
    checkpoint_path: str | Path,
    anterior_path: str | Path,
    posterior_path: str | Path,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str = "cpu",
    image_size: tuple[int, int] = (256, 1024),
    input_channels: int = 3,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PostprocessResult:
    sample = build_pair(anterior_path, posterior_path)
    transform_config = InferenceTransformConfig(
        image_size=image_size,
        input_channels=input_channels,
        mean=mean,
        std=std,
    )
    pair = preprocess_pair(sample, transform_config)
    loaded = load_segformer_checkpoint(checkpoint_path, config_path=config_path, device=device)
    logits = run_inference(loaded, pair)
    result = postprocess_logits(logits, views=pair.views)
    result.metadata.update(
        {
            "case_id": pair.case_id,
            "original_paths": pair.original_paths,
            "checkpoint_path": str(checkpoint_path),
            "task_mode": loaded.config.model.task_mode,
            "variant": loaded.config.model.variant,
            "image_size": list(image_size),
            "input_channels": input_channels,
        }
    )
    if output_dir is not None:
        save_result(result, output_dir)
    return result


def save_result(result: PostprocessResult, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for task, view_masks in result.masks.items():
        for view, mask in view_masks.items():
            _save_mask(root / f"{task}_{view}.png", mask)
    for view, mask in result.lesion_malignant.items():
        _save_mask(root / f"lesion_malignant_{view}.png", mask * 255)
    for view, color in result.bone_color.items():
        Image.fromarray(color, mode="RGB").save(root / f"bone_color_{view}.png")
    write_csv_summary(result, root / "summary.csv")
    (root / "metadata.json").write_text(json.dumps(_json_safe(result.metadata), indent=2), encoding="utf-8")


def _mask_class_pixel_counts(mask: np.ndarray, num_classes: int) -> list[int]:
    counts = np.bincount(np.asarray(mask).ravel(), minlength=num_classes)
    return counts[:num_classes].tolist()


def write_csv_summary(result: PostprocessResult, path: Path) -> None:
    lesion_masks = result.masks.get("lesion", {})
    bone_masks = result.masks.get("bone", {})
    views = result.metadata.get("views") or sorted(set(lesion_masks) | set(bone_masks))
    fieldnames = [
        "case_id",
        "view",
        "lesion_background_px",
        "lesion_benign_px",
        "lesion_malignant_px",
        "bone_background_px",
        "bone_foreground_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for view in views:
            row = {"case_id": result.metadata.get("case_id", ""), "view": view}
            if view in lesion_masks:
                background, benign, malignant = _mask_class_pixel_counts(lesion_masks[view], 3)
                row["lesion_background_px"] = background
                row["lesion_benign_px"] = benign
                row["lesion_malignant_px"] = malignant
            if view in bone_masks:
                counts = _mask_class_pixel_counts(bone_masks[view], len(BONE_LABELS))
                row["bone_background_px"] = counts[0]
                row["bone_foreground_px"] = sum(counts[1:])
            writer.writerow(row)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(path)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer multitask paired-view inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to SegFormer checkpoint.")
    parser.add_argument("--config", default=None, help="Optional JSON config if checkpoint does not contain one.")
    parser.add_argument("--anterior", required=True, help="Anterior image path.")
    parser.add_argument("--posterior", required=True, help="Posterior image path.")
    parser.add_argument("--output-dir", required=True, help="Directory for masks, overlays, and metadata.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0.")
    parser.add_argument("--height", type=int, default=256, help="Model input height.")
    parser.add_argument("--width", type=int, default=1024, help="Model input width.")
    parser.add_argument("--input-channels", type=int, default=3, choices=(1, 3), help="SegFormer input channels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_segmentation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        anterior_path=args.anterior,
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        device=args.device,
        image_size=(args.height, args.width),
        input_channels=args.input_channels,
    )
    print(asdict(result.metadata) if hasattr(result.metadata, "__dataclass_fields__") else result.metadata)


if __name__ == "__main__":
    main()

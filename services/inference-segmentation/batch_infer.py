from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from infer import run_inference
from main import save_result
from model import load_segformer_checkpoint
from pair_loader import build_pair
from postprocess import postprocess_logits
from preprocess import preprocess_pair
from transforms import InferenceTransformConfig


def read_case_ids(root_dir: Path, split_file: str) -> list[str]:
    path = root_dir / split_file
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_batch_inference(
    checkpoint_path: str | Path,
    root_dir: str | Path,
    split_file: str,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    device: str = "cuda",
    image_size: tuple[int, int] = (1024, 512),
    input_channels: int = 3,
    image_dir: str = "images",
    image_suffix: str = ".png",
    anterior_suffix: str = "_0000",
    posterior_suffix: str = "_0001",
) -> Path:
    root = Path(root_dir)
    case_ids = read_case_ids(root, split_file)
    loaded = load_segformer_checkpoint(checkpoint_path, config_path=config_path, device=device)
    transform_config = InferenceTransformConfig(image_size=image_size, input_channels=input_channels)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for case_id in tqdm(case_ids, desc="batch inference"):
        anterior_path = root / image_dir / f"{case_id}{anterior_suffix}{image_suffix}"
        posterior_path = root / image_dir / f"{case_id}{posterior_suffix}{image_suffix}"
        sample = build_pair(anterior_path, posterior_path)
        pair = preprocess_pair(sample, transform_config)
        logits = run_inference(loaded, pair)
        result = postprocess_logits(logits, views=pair.views)
        result.metadata.update(
            {
                "case_id": pair.case_id,
                "original_paths": pair.original_paths,
                "checkpoint_path": str(checkpoint_path),
                "task_mode": loaded.config.model.task_mode,
                "variant": loaded.config.model.variant,
            }
        )
        save_result(result, out_root / case_id)

    combined_path = out_root / "all_cases_summary.csv"
    _combine_case_csvs(out_root, case_ids, combined_path)
    return combined_path


def _combine_case_csvs(out_root: Path, case_ids: list[str], combined_path: Path) -> None:
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for case_id in case_ids:
        case_csv = out_root / case_id / "summary.csv"
        with case_csv.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)

    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer inference across a full dataset split, saving real per-case results.")
    parser.add_argument("--checkpoint", required=True, help="Path to SegFormer checkpoint.")
    parser.add_argument("--config", default=None, help="Optional JSON config if checkpoint does not contain one.")
    parser.add_argument("--root-dir", required=True, help="Dataset root, e.g. data/bs80k_lesion.")
    parser.add_argument("--split", required=True, choices=("val", "test"), help="Which split file to run.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-case masks, CSVs, and metadata.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0.")
    parser.add_argument("--height", type=int, default=1024, help="Model input height.")
    parser.add_argument("--width", type=int, default=512, help="Model input width.")
    parser.add_argument("--input-channels", type=int, default=3, choices=(1, 3), help="SegFormer input channels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined_path = run_batch_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        root_dir=args.root_dir,
        split_file=f"{args.split}.txt",
        output_dir=args.output_dir,
        device=args.device,
        image_size=(args.height, args.width),
        input_channels=args.input_channels,
    )
    print(f"wrote combined results to {combined_path}")


if __name__ == "__main__":
    main()

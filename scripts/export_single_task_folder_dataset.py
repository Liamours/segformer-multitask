"""Export a lesion-only nnU-Net raw dataset into the SegFormer single-task folder-dataset
contract (images/, masks/, train/val/test.txt).

Source layout (nnU-Net raw, e.g. Dataset261_BS80KLesionOnly):
  imagesTr/<case>_0000.png, <case>_0001.png
  labelsTr/lesion/<case>_0000.png, <case>_0001.png
  split_seed42.csv: id,case_id,split

Output layout (consumed by repo/segformer_multitask/src/datasets.py, single_task + paired_views):
  images/<case>_0000.png, <case>_0001.png
  masks/<case>_0000.png, <case>_0001.png   (lesion: 0 background, 1 benign, 2 malignant)
  train.txt, val.txt, test.txt              (one case id per line, e.g. bs80k_0001)
  export_manifest.csv, export_summary.json

Sibling of export_multitask_folder_dataset.py, dropping the task_b (bone) copy and renaming
task_a_masks -> masks per the single_task contract build_folder_samples() actually reads
(src/datasets.py, multitask=False branch).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from tqdm import tqdm

VIEW_SUFFIXES = ("_0000", "_0001")


def read_split(split_csv: Path) -> dict[str, str]:
    with split_csv.open("r", encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row["split"] for row in csv.DictReader(handle)}


def export(source_root: Path, output_root: Path) -> dict:
    case_to_split = read_split(source_root / "split_seed42.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    for directory in (images_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_cases: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    manifest_rows = []

    for case_id, split in tqdm(sorted(case_to_split.items()), desc="Exporting cases", unit="case"):
        split_cases[split].append(case_id)
        for suffix in VIEW_SUFFIXES:
            filename = f"{case_id}{suffix}.png"
            src_image = source_root / "imagesTr" / filename
            src_mask = source_root / "labelsTr" / "lesion" / filename
            for src in (src_image, src_mask):
                if not src.is_file():
                    raise FileNotFoundError(f"Missing source file: {src}")
            shutil.copy2(src_image, images_dir / filename)
            shutil.copy2(src_mask, masks_dir / filename)
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "view": "anterior" if suffix == "_0000" else "posterior",
                    "image": str((images_dir / filename).relative_to(output_root)),
                    "mask": str((masks_dir / filename).relative_to(output_root)),
                }
            )

    for split, cases in split_cases.items():
        split_file = output_root / f"{split}.txt"
        split_file.write_text("\n".join(sorted(cases)) + "\n", encoding="utf-8")

    with (output_root / "export_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "split", "view", "image", "mask"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source": str(source_root),
        "export": str(output_root),
        "train_cases": len(split_cases["train"]),
        "val_cases": len(split_cases["val"]),
        "test_cases": len(split_cases["test"]),
        "images": len(manifest_rows),
        "masks": len(manifest_rows),
        "task": "lesion: 0 background, 1 benign, 2 malignant",
    }
    with (output_root / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\rifqi\research-wbbs-multitask_uq\dataset\nnunet\nnUNet_raw\Dataset261_BS80KLesionOnly"),
        help="nnU-Net raw dataset root containing imagesTr/, labelsTr/lesion/, split_seed42.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\rifqi\research-wbbs-multitask_uq\dataset\segformer\bs80k_lesion"),
        help="Destination folder-dataset root",
    )
    args = parser.parse_args()

    summary = export(args.source_root, args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

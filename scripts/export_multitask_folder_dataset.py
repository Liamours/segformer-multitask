"""Export the paired-view lesion+bone nnU-Net raw dataset into the SegFormer multitask
folder-dataset contract (images/, task_a_masks/, task_b_masks/, train/val/test.txt).

Source layout (nnU-Net raw, e.g. Dataset260_BS80KLesionBoneMT):
  imagesTr/<case>_0000.png, <case>_0001.png
  labelsTr/lesion/<case>_0000.png, <case>_0001.png
  labelsTr/bone/<case>_0000.png, <case>_0001.png
  split_seed42.csv: id,case_id,split

Output layout (consumed by repo/segformer_multitask/src/datasets.py, multitask + paired_views):
  images/<case>_0000.png, <case>_0001.png
  task_a_masks/<case>_0000.png, <case>_0001.png   (lesion: 0 background, 1 benign, 2 malignant)
  task_b_masks/<case>_0000.png, <case>_0001.png   (bone: 0 background, 1..12 regions)
  train.txt, val.txt, test.txt                     (one case id per line, e.g. bs80k_0001)
  export_manifest.csv, export_summary.json
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
    task_a_dir = output_root / "task_a_masks"
    task_b_dir = output_root / "task_b_masks"
    for directory in (images_dir, task_a_dir, task_b_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_cases: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    manifest_rows = []

    for case_id, split in tqdm(sorted(case_to_split.items()), desc="Exporting cases", unit="case"):
        split_cases[split].append(case_id)
        for suffix in VIEW_SUFFIXES:
            filename = f"{case_id}{suffix}.png"
            src_image = source_root / "imagesTr" / filename
            src_lesion = source_root / "labelsTr" / "lesion" / filename
            src_bone = source_root / "labelsTr" / "bone" / filename
            for src in (src_image, src_lesion, src_bone):
                if not src.is_file():
                    raise FileNotFoundError(f"Missing source file: {src}")
            shutil.copy2(src_image, images_dir / filename)
            shutil.copy2(src_lesion, task_a_dir / filename)
            shutil.copy2(src_bone, task_b_dir / filename)
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "view": "anterior" if suffix == "_0000" else "posterior",
                    "image": str((images_dir / filename).relative_to(output_root)),
                    "task_a_mask": str((task_a_dir / filename).relative_to(output_root)),
                    "task_b_mask": str((task_b_dir / filename).relative_to(output_root)),
                }
            )

    for split, cases in split_cases.items():
        split_file = output_root / f"{split}.txt"
        split_file.write_text("\n".join(sorted(cases)) + "\n", encoding="utf-8")

    with (output_root / "export_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "split", "view", "image", "task_a_mask", "task_b_mask"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source": str(source_root),
        "export": str(output_root),
        "train_cases": len(split_cases["train"]),
        "val_cases": len(split_cases["val"]),
        "test_cases": len(split_cases["test"]),
        "images": len(manifest_rows),
        "task_a_masks": len(manifest_rows),
        "task_b_masks": len(manifest_rows),
        "task_a": "lesion: 0 background, 1 benign, 2 malignant",
        "task_b": "bone: 0 background, 1..12 regions",
    }
    with (output_root / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\research\research-wbbs-multitask_uq\dataset\nnunet\nnUNet_raw\Dataset260_BS80KLesionBoneMT"),
        help="nnU-Net raw dataset root containing imagesTr/, labelsTr/lesion/, labelsTr/bone/, split_seed42.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\research\research-wbbs-multitask_uq\dataset\segformer\bs80k_multitask"),
        help="Destination folder-dataset root",
    )
    args = parser.parse_args()

    summary = export(args.source_root, args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

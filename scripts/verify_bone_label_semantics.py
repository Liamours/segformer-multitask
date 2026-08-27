"""Correctness gate: do SegFormer's task_b bone labels mean the same integers as nnU-Net's?

Both backends are trained on masks derived from BS-80K, but they were exported by different
pipelines. If the integer-to-region mapping differs, any cross-backend bone comparison is silently
wrong. This compares the exported SegFormer task_b masks against the nnU-Net bone labels for the
same case ids and views, pixel for pixel, and reports the full label-to-label contingency.

Exit status is non-zero when the mapping is not the identity, so this can gate downstream work.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

VIEWS = ("_0000", "_0001")
PALETTE = {
    0: "Background", 1: "Skull", 2: "Cervical Vert", 3: "Thoracic Vert", 4: "Ribs",
    5: "Sternum", 6: "Clavicle", 7: "Scapula", 8: "Humerus", 9: "Lumbar Vert",
    10: "Sacrum", 11: "Pelvis", 12: "Femur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segformer-root", required=True, help="Folder dataset root holding task_b_masks/.")
    parser.add_argument("--nnunet-labels", required=True, help="nnU-Net labelsTr/bone directory.")
    parser.add_argument("--split-file", default=None, help="Optional split .txt of case ids; default is all cases.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on case count.")
    parser.add_argument("--report", default=None, help="Optional CSV path for the contingency table.")
    return parser.parse_args()


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def main() -> int:
    args = parse_args()
    segformer_masks = Path(args.segformer_root) / "task_b_masks"
    nnunet_labels = Path(args.nnunet_labels)
    for directory in (segformer_masks, nnunet_labels):
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")

    if args.split_file:
        case_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        case_ids = sorted({path.stem[:-5] for path in segformer_masks.glob(f"*{VIEWS[0]}.png")})
    if args.limit:
        case_ids = case_ids[: args.limit]
    if not case_ids:
        raise ValueError("No case ids resolved.")

    pairs: Counter = Counter()
    shape_mismatches: list[str] = []
    missing: list[str] = []
    compared = 0

    for case_id in tqdm(case_ids, desc="cases"):
        for view in VIEWS:
            segformer_path = segformer_masks / f"{case_id}{view}.png"
            nnunet_path = nnunet_labels / f"{case_id}{view}.png"
            if not segformer_path.is_file() or not nnunet_path.is_file():
                missing.append(f"{case_id}{view}")
                continue
            segformer_mask = _read(segformer_path)
            nnunet_mask = _read(nnunet_path)
            if segformer_mask.shape != nnunet_mask.shape:
                shape_mismatches.append(f"{case_id}{view}: {segformer_mask.shape} vs {nnunet_mask.shape}")
                continue
            combined = segformer_mask.astype(np.int64) * 256 + nnunet_mask.astype(np.int64)
            for code, count in zip(*np.unique(combined, return_counts=True)):
                pairs[(int(code) // 256, int(code) % 256)] += int(count)
            compared += 1

    print(f"\ncompared {compared} mask pairs over {len(case_ids)} cases")
    if missing:
        print(f"missing files: {len(missing)} (first: {missing[:3]})")
    if shape_mismatches:
        print(f"shape mismatches: {len(shape_mismatches)} (first: {shape_mismatches[:3]})")

    segformer_values = sorted({pair[0] for pair in pairs})
    nnunet_values = sorted({pair[1] for pair in pairs})
    print(f"SegFormer task_b label values present: {segformer_values}")
    print(f"nnU-Net bone label values present:     {nnunet_values}")

    off_diagonal = {pair: count for pair, count in pairs.items() if pair[0] != pair[1]}
    total = sum(pairs.values())
    print(f"\ntotal pixels compared: {total:,}")
    print(f"pixels where the two labels disagree: {sum(off_diagonal.values()):,}")

    rows = []
    for (segformer_value, nnunet_value), count in sorted(pairs.items()):
        rows.append({
            "segformer_label": segformer_value,
            "nnunet_label": nnunet_value,
            "segformer_region": PALETTE.get(segformer_value, "?"),
            "nnunet_region": PALETTE.get(nnunet_value, "?"),
            "pixels": count,
            "agrees": segformer_value == nnunet_value,
        })
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {report_path}")

    if off_diagonal:
        print("\nLABEL MAPPING IS NOT THE IDENTITY. Largest disagreements:")
        for (segformer_value, nnunet_value), count in sorted(off_diagonal.items(), key=lambda item: -item[1])[:15]:
            print(
                f"  segformer={segformer_value} ({PALETTE.get(segformer_value, '?')}) -> "
                f"nnunet={nnunet_value} ({PALETTE.get(nnunet_value, '?')}): {count:,} px"
            )
        return 1

    print("\nMapping is the identity: SegFormer task_b labels equal nnU-Net bone labels pixel for pixel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

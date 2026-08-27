"""Collect the three SegFormer runs' own evaluation JSON into one comparable table.

These numbers come from `src/evaluate.py`, which is the SegFormer project's internal metric code.
They are NOT computed the same way as the nnU-Net side, so the emitted Markdown carries an explicit
definition block; read it before putting any of these values in a table next to an nnU-Net number.
"""

import argparse
import csv
import json
from pathlib import Path

RUNS = (
    ("single_task", "single_task_100epochs", "SegFormer, single-task"),
    ("multi_task_head", "dual_head", "SegFormer, multi-task-head"),
    ("multi_task_decoder", "dual_decoder", "SegFormer, multi-task-decoder"),
)
SPLITS = ("val", "test")

# Lesion metrics are unprefixed for single-task and task_a_-prefixed for the multi-task runs.
LESION_METRICS = ("class_1_dice", "class_2_dice", "foreground_mean_dice", "mean_dice", "pixel_accuracy", "loss")
BONE_METRICS = ("task_b_foreground_mean_dice", "task_b_mean_dice", "task_b_pixel_accuracy", "task_b_loss")

DEFINITIONS = """\
## How these numbers were computed, and why they are not nnU-Net numbers

Source: each run's `eval_{split}.json`, written by `repo/segformer_multitask/src/evaluate.py`, which
averages `src/metrics.py::segmentation_scores` over evaluation batches. Four properties differ from
`nnunetv2/evaluation/evaluate_multitask_predictions.py`, which produced every nnU-Net number in
`analyses/evaluations/six_experiment_best_val_test.csv`:

1. **Aggregation unit.** SegFormer pools all pixels in a batch of 4 paired canvases, computes one
   Dice per class for that batch, then averages over batches. nnU-Net computes one Dice per
   (case, view, class) and averages over those rows. A patient with a small lesion contributes
   equally to the nnU-Net mean and is swamped in the SegFormer mean.
2. **View handling.** SegFormer scores the concatenated anterior-plus-posterior canvas as one image,
   so the reported number is a both-views-pooled score. nnU-Net scores each view separately.
3. **Empty-class batches.** `class_N_dice` is written as 0.0 for a batch in which neither the
   prediction nor the ground truth contains class N, and that zero enters the average. On the test
   split this deflates malignant Dice: 4 of 74 batches contain no malignant pixels at all.
   `foreground_mean_dice` excludes such classes for that batch, so the two are not consistent with
   each other.
4. **Undefined-Dice policy.** nnU-Net drops a (case, view, class) row when Dice is undefined
   (empty ground truth and empty prediction) rather than scoring it 0.

Protocol-matched numbers, produced by running SegFormer predictions through nnU-Net's own evaluator,
are in `analyses/evaluations/segformer_vs_nnunet/`. Use those for any cross-backend table. Use this
file only for SegFormer-versus-SegFormer comparison, where all three runs share these definitions.

Class indices: lesion 1 = benign, 2 = malignant. Bone classes follow
`context/nnunetv2/bone-region-palette.md`, verified identical across backends.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", default=r"C:\research\research-wbbs-multitask_uq\weights")
    parser.add_argument("--output-stem", default=r"C:\research\research-wbbs-multitask_uq\analyses\evaluations\segformer_eval_summary")
    return parser.parse_args()


def _metric(metrics: dict, name: str, multitask: bool) -> float | None:
    if name.startswith("task_b_"):
        return metrics.get(name)
    key = f"task_a_{name}" if multitask else name
    return metrics.get(key, metrics.get(name))


def main() -> None:
    args = parse_args()
    weights_root = Path(args.weights_root)
    rows = []

    for registry_name, run_dir, prose_name in RUNS:
        multitask = registry_name != "single_task"
        for split in SPLITS:
            payload_path = weights_root / run_dir / f"eval_{split}.json"
            if not payload_path.is_file():
                raise FileNotFoundError(f"Missing evaluation file: {payload_path}")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            row = {
                "model": registry_name,
                "prose_name": prose_name,
                "run_dir": run_dir,
                "split": split,
                "checkpoint": payload.get("checkpoint", ""),
            }
            for name in LESION_METRICS:
                row[f"lesion_{name}"] = _metric(metrics, name, multitask)
            for name in BONE_METRICS:
                row[name] = _metric(metrics, name, multitask)
            rows.append(row)

    stem = Path(args.output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def cell(value) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "# SegFormer evaluation summary, all three runs",
        "",
        "Generated by `repo/segformer_multitask/scripts/build_segformer_eval_summary.py` from each",
        "run's own `eval_{val,test}.json`. Machine-readable copy: `segformer_eval_summary.csv`.",
        "",
        "## Lesion task",
        "",
        "| Model | Split | Benign Dice | Malignant Dice | Foreground mean Dice | Mean Dice | Pixel acc. | Loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['prose_name']} | {row['split']} | {cell(row['lesion_class_1_dice'])} | "
            f"{cell(row['lesion_class_2_dice'])} | {cell(row['lesion_foreground_mean_dice'])} | "
            f"{cell(row['lesion_mean_dice'])} | {cell(row['lesion_pixel_accuracy'])} | "
            f"{cell(row['lesion_loss'])} |"
        )
    lines += [
        "",
        "## Bone task",
        "",
        "Single-task has no bone head, so its row is N/A by construction rather than missing data.",
        "",
        "| Model | Split | Foreground mean Dice | Mean Dice | Pixel acc. | Loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['prose_name']} | {row['split']} | {cell(row['task_b_foreground_mean_dice'])} | "
            f"{cell(row['task_b_mean_dice'])} | {cell(row['task_b_pixel_accuracy'])} | "
            f"{cell(row['task_b_loss'])} |"
        )
    lines += ["", DEFINITIONS]

    md_path = stem.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()

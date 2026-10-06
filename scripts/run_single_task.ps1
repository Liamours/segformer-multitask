$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

cd (Join-Path $__d 'segformer_multitask')

uv run segformer-multitask-train --config configs/folder_single_task.json

uv run segformer-multitask-evaluate --checkpoint "$WeightsRoot\single_task\checkpoints\best.pt" --split val --output "$WeightsRoot\single_task\eval_val.json"
uv run segformer-multitask-evaluate --checkpoint "$WeightsRoot\single_task\checkpoints\best.pt" --split test --output "$WeightsRoot\single_task\eval_test.json"

uv run python services/inference-segmentation/main.py --checkpoint "$WeightsRoot\single_task\checkpoints\best.pt" --anterior $SegformerRoot/bs80k_lesion/images/bs80k_0007_0000.png --posterior $SegformerRoot/bs80k_lesion/images/bs80k_0007_0001.png --output-dir "$InferencesRoot\segformer\single_task\bs80k_0007" --device cuda --height 1024 --width 512

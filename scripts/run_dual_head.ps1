$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

cd "C:\research\research-wbbs-multitask_uq\repo\segformer_multitask"

uv run segformer-multitask-train --config configs/folder_dual_head.json

uv run segformer-multitask-evaluate --checkpoint "$WeightsRoot\dual_head\checkpoints\best.pt" --split val --output "$WeightsRoot\dual_head\eval_val.json"
uv run segformer-multitask-evaluate --checkpoint "$WeightsRoot\dual_head\checkpoints\best.pt" --split test --output "$WeightsRoot\dual_head\eval_test.json"

uv run python services/inference-segmentation/main.py --checkpoint "$WeightsRoot\dual_head\checkpoints\best.pt" --anterior C:/research/research-wbbs-multitask_uq/dataset/segformer/bs80k_multitask/images/bs80k_0007_0000.png --posterior C:/research/research-wbbs-multitask_uq/dataset/segformer/bs80k_multitask/images/bs80k_0007_0001.png --output-dir "$InferencesRoot\segformer\dual_head\bs80k_0007" --device cuda --height 1024 --width 512

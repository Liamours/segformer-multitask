$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

cd "C:\research\research-wbbs-multitask_uq\repo\segformer_multitask"

uv run python services/inference-segmentation/batch_infer.py --checkpoint "$WeightsRoot\single_task_100epochs\checkpoints\best.pt" --root-dir C:/research/research-wbbs-multitask_uq/dataset/segformer/bs80k_lesion --split val --output-dir "$InferencesRoot\segformer\single_task_100epochs\val" --device cuda --height 1024 --width 512
uv run python services/inference-segmentation/batch_infer.py --checkpoint "$WeightsRoot\single_task_100epochs\checkpoints\best.pt" --root-dir C:/research/research-wbbs-multitask_uq/dataset/segformer/bs80k_lesion --split test --output-dir "$InferencesRoot\segformer\single_task_100epochs\test" --device cuda --height 1024 --width 512

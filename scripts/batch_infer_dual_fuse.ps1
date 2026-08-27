$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

cd "C:\research\research-wbbs-multitask_uq\repo\segformer_multitask"

uv run python services/inference-segmentation/batch_infer.py --checkpoint "$WeightsRoot\dual_fuse\checkpoints\best.pt" --root-dir C:/research/research-wbbs-multitask_uq/dataset/segformer/bs80k_multitask --split val --output-dir "$InferencesRoot\segformer\dual_fuse\val" --device cuda --height 1024 --width 512
uv run python services/inference-segmentation/batch_infer.py --checkpoint "$WeightsRoot\dual_fuse\checkpoints\best.pt" --root-dir C:/research/research-wbbs-multitask_uq/dataset/segformer/bs80k_multitask --split test --output-dir "$InferencesRoot\segformer\dual_fuse\test" --device cuda --height 1024 --width 512

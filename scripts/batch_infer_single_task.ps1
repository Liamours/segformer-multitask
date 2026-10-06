$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

cd (Join-Path $__d 'segformer_multitask')

uv run python services/inference-segmentation/batch_infer.py --checkpoint "$WeightsRoot\single_task\checkpoints\best.pt" --root-dir $SegformerRoot/bs80k_lesion --split val --output-dir "$InferencesRoot\segformer\single_task\val" --device cuda --height 1024 --width 512
uv run python services/inference-segmentation/batch_infer.py --checkpoint "$WeightsRoot\single_task\checkpoints\best.pt" --root-dir $SegformerRoot/bs80k_lesion --split test --output-dir "$InferencesRoot\segformer\single_task\test" --device cuda --height 1024 --width 512

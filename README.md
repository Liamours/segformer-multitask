# segformer-multitask

SegFormer multitask research scaffold.

Current direction

- shared MiT backbone
- single-task baseline
- dual-head multitask variant
- dual-decoder multitask variant
- flexible backbone presets such as B0 to B5

Current scope

- real MiT B0 to B5 backbone presets
- SegFormer decoder and dense segmentation heads
- single-task, dual-head, and dual-decoder models
- real folder-based dataset loading for single-task and multitask segmentation
- config-driven training, evaluation, logging, and checkpointing
- dummy dataset only for smoke tests and unit tests

Dataset contract

Single-task folder dataset

- `images/<sample_id>.png`
- `masks/<sample_id>.png`
- `train.txt`
- `val.txt`

Multitask folder dataset

- `images/<sample_id>.png`
- `task_a_masks/<sample_id>.png`
- `task_b_masks/<sample_id>.png`
- `train.txt`
- `val.txt`

Each split file contains one sample id per line.

Config surface

- `model.variant`
  - `mit_b0`
  - `mit_b1`
  - `mit_b2`
  - `mit_b3`
  - `mit_b4`
  - `mit_b5`
- `model.task_mode`
  - `single_task`
  - `dual_head`
  - `dual_decoder`
- `data.dataset_name`
  - `dummy`
  - `folder`

Training behavior

- one shared trainer path for single-task and both multitask modes
- per-epoch train and val evaluation
- optional JSONL metrics logging
- optional latest and best checkpoints
- differential learning rates
  - backbone `1e-5`
  - head `1e-4`

Num workers policy

- worker candidates are generated from CPU count
- the trainer logs:
  - available worker options
  - recommended worker count
  - configured train and val worker counts
- recommendation is conservative by default and derived from:
  - CPU count
  - batch size

Next implementation focus

1. add task-specific losses beyond plain cross entropy
2. add richer evaluation metrics such as IoU and Dice
3. add experiment templates for real datasets

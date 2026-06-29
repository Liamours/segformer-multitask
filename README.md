# segformer-multitask

SegFormer multitask research scaffold.

Current direction

- shared MiT backbone
- single-task baseline
- dual-head multitask variant
- dual-decoder multitask variant
- flexible backbone presets such as B0 and B2

Current scope

- real MiT B0 and B2 backbone presets
- SegFormer decoder and dense segmentation heads
- single-task, dual-head, and dual-decoder models
- dummy dataset, smoke training path, and contract tests

Next implementation focus

1. add real dataset adapters
2. extend training and evaluation workflows
3. benchmark multitask variants against the baseline

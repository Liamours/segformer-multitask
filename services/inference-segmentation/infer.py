from __future__ import annotations

import torch

from model import LoadedSegFormer
from preprocess import PreprocessedPair


def run_inference(loaded: LoadedSegFormer, pair: PreprocessedPair) -> dict[str, torch.Tensor]:
    image = pair.image.to(loaded.device)
    with torch.inference_mode():
        outputs = loaded.model(image)
    return {task: logits.detach().cpu() for task, logits in loaded.output_to_tasks(outputs).items()}

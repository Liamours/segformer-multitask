from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from palette import BONE_LABELS, colorize_mask


@dataclass(frozen=True)
class PostprocessResult:
    masks: dict[str, dict[str, np.ndarray]]
    lesion_malignant: dict[str, np.ndarray]
    bone_color: dict[str, np.ndarray]
    metadata: dict[str, object]


def logits_to_mask(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [N, C, H, W].")
    return torch.argmax(logits, dim=1).to(torch.int64).cpu().numpy()


def split_view_masks(mask_batch: np.ndarray, views: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Split a single side-by-side canvas mask back into per-view masks along width."""
    if mask_batch.shape[0] != 1:
        raise ValueError(f"Expected a single paired-view canvas (batch size 1), got batch size {mask_batch.shape[0]}.")
    canvas = mask_batch[0]
    canvas_width = canvas.shape[1]
    if canvas_width % len(views) != 0:
        raise ValueError(f"Canvas width {canvas_width} is not evenly divisible by {len(views)} views.")
    view_width = canvas_width // len(views)
    return {
        view: canvas[:, index * view_width : (index + 1) * view_width]
        for index, view in enumerate(views)
    }


def malignant_binary(lesion_mask: np.ndarray, malignant_label: int = 2) -> np.ndarray:
    return (np.asarray(lesion_mask) == malignant_label).astype(np.uint8)


def postprocess_logits(
    logits: dict[str, torch.Tensor],
    views: tuple[str, ...] = ("anterior", "posterior"),
    malignant_label: int = 2,
) -> PostprocessResult:
    masks: dict[str, dict[str, np.ndarray]] = {}
    for task, task_logits in logits.items():
        masks[task] = split_view_masks(logits_to_mask(task_logits), views)

    lesion_masks = masks.get("lesion", {})
    lesion_malignant = {
        view: malignant_binary(mask, malignant_label=malignant_label)
        for view, mask in lesion_masks.items()
    }
    bone_color = {
        view: colorize_mask(mask)
        for view, mask in masks.get("bone", {}).items()
    }
    return PostprocessResult(
        masks=masks,
        lesion_malignant=lesion_malignant,
        bone_color=bone_color,
        metadata={
            "views": list(views),
            "tasks": sorted(logits),
            "malignant_label": malignant_label,
            "bone_labels": BONE_LABELS,
            "overlap_allowed": True,
        },
    )

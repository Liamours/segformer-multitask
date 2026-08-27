from __future__ import annotations

from typing import Mapping

import numpy as np

BONE_LABELS: dict[int, str] = {
    0: "Background",
    1: "Skull",
    2: "Cervical Vert",
    3: "Thoracic Vert",
    4: "Ribs",
    5: "Sternum",
    6: "Clavicle",
    7: "Scapula",
    8: "Humerus",
    9: "Lumbar Vert",
    10: "Sacrum",
    11: "Pelvis",
    12: "Femur",
}

BONE_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (230, 25, 75),
    2: (60, 180, 75),
    3: (255, 225, 25),
    4: (0, 130, 200),
    5: (245, 130, 48),
    6: (145, 30, 180),
    7: (70, 240, 240),
    8: (240, 50, 230),
    9: (210, 245, 60),
    10: (250, 190, 190),
    11: (0, 128, 128),
    12: (230, 190, 255),
}


def colorize_mask(mask: np.ndarray, palette: Mapping[int, tuple[int, int, int]] = BONE_PALETTE) -> np.ndarray:
    labels = np.asarray(mask, dtype=np.int64)
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, rgb in palette.items():
        color[labels == label] = rgb
    return color

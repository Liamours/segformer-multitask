from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def remove_small_components(
    mask: np.ndarray,
    min_pixels: int,
    target_labels: Iterable[int] | None = None,
) -> np.ndarray:
    """Remove small connected components without applying unsafe largest-component rules."""
    if min_pixels <= 1:
        return np.asarray(mask).copy()

    labels = np.asarray(mask)
    cleaned = labels.copy()
    candidate_labels = set(int(label) for label in target_labels) if target_labels is not None else set(np.unique(labels))
    candidate_labels.discard(0)
    visited = np.zeros(labels.shape, dtype=bool)

    for label in sorted(candidate_labels):
        ys, xs = np.where((labels == label) & ~visited)
        for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=False):
            if visited[start_y, start_x] or labels[start_y, start_x] != label:
                continue
            component = _flood_fill(labels, visited, start_y, start_x, label)
            if len(component) < min_pixels:
                for y, x in component:
                    cleaned[y, x] = 0
    return cleaned


def _flood_fill(labels: np.ndarray, visited: np.ndarray, start_y: int, start_x: int, label: int) -> list[tuple[int, int]]:
    height, width = labels.shape
    stack = [(start_y, start_x)]
    component: list[tuple[int, int]] = []
    visited[start_y, start_x] = True

    while stack:
        y, x = stack.pop()
        component.append((y, x))
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                continue
            if visited[next_y, next_x] or labels[next_y, next_x] != label:
                continue
            visited[next_y, next_x] = True
            stack.append((next_y, next_x))
    return component

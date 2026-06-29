import os
import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def list_num_workers_options(cpu_count: int | None = None) -> list[int]:
    cpu_total = cpu_count or (os.cpu_count() or 1)
    candidates = {
        0,
        1,
        max(1, cpu_total // 4),
        max(1, cpu_total // 2),
        max(1, cpu_total - 2),
    }
    return sorted(value for value in candidates if value < cpu_total or value == 0)


def recommend_num_workers(batch_size: int, cpu_count: int | None = None) -> int:
    cpu_total = cpu_count or (os.cpu_count() or 1)
    if cpu_total <= 2:
        return 0
    upper_bound = max(1, cpu_total - 2)
    return min(upper_bound, max(1, batch_size * 2))

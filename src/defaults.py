from dataclasses import dataclass

from .backbones import MIT_B0


@dataclass(frozen=True)
class TrainDefaults:
    seed: int = 42
    image_size: tuple[int, int] = (64, 64)
    num_classes: int = 4
    batch_size: int = 2
    learning_rate_backbone: float = 1e-5
    learning_rate_head: float = 1e-4
    weight_decay: float = 0.01
    variant_name: str = MIT_B0.name
    task_mode: str = "single_task"
    steps: int = 1


DEFAULTS = TrainDefaults()

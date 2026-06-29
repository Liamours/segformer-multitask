import torch
from torch.utils.data import Dataset


class DummySegmentationDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        image_size: tuple[int, int] = (64, 64),
        num_classes: int = 4,
        multitask: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.multitask = multitask
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand((3, *self.image_size), generator=generator)
        mask = torch.randint(0, self.num_classes, self.image_size, generator=generator)
        sample = {"image": image, "mask": mask}

        if self.multitask:
            task_b_mask = torch.randint(0, self.num_classes, self.image_size, generator=generator)
            sample = {"image": image, "task_a_mask": mask, "task_b_mask": task_b_mask}

        return sample

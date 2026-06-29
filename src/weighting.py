from collections import deque

import torch


class FixedLossWeighting:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = dict(weights)

    def reduce(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([self.weights[key] * losses[key] for key in self.weights]).sum()


class DynamicWeightAverage:
    def __init__(self, keys: tuple[str, ...], temperature: float = 2.0) -> None:
        self.keys = keys
        self.temperature = temperature
        self.history: deque[dict[str, torch.Tensor]] = deque(maxlen=2)
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.weights = {key: 1.0 for key in keys}

    def _update_weights(self) -> None:
        if self.buffer:
            self.history.append(
                {key: torch.stack([item[key] for item in self.buffer]).mean() for key in self.keys}
            )

        if len(self.history) < 2:
            self.weights = {key: 1.0 for key in self.keys}
            return

        ratios = torch.stack([self.history[-1][key] / self.history[-2][key] for key in self.keys])
        scaled = len(self.keys) * torch.softmax(ratios / self.temperature, dim=0)
        self.weights = {key: value.item() for key, value in zip(self.keys, scaled, strict=True)}

    def reduce(self, losses: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            self._update_weights()

        detached = {key: losses[key].detach().clone() for key in self.keys}
        if len(self.buffer) == batch_idx:
            self.buffer.append(detached)
        else:
            self.buffer[batch_idx] = detached

        return torch.stack([self.weights[key] * losses[key] for key in self.keys]).sum()

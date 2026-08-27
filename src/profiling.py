from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelProfile:
    total_params: int
    trainable_params: int
    flops: int


def profile_model(model: nn.Module, input_size: tuple[int, int, int, int], device: str = "cpu") -> ModelProfile:
    target_device = torch.device(device)
    model = model.to(target_device)
    was_training = model.training
    model.eval()
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels // module.groups
        flops += int(batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w * 2)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        flops += int(output.numel() * module.in_features * 2)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.inference_mode():
        model(torch.zeros(input_size, device=target_device))

    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()

    return ModelProfile(
        total_params=sum(parameter.numel() for parameter in model.parameters()),
        trainable_params=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        flops=flops,
    )

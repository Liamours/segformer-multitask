from dataclasses import asdict

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbones import MIT_B0, MIT_B2, MixVisionTransformer
from .datasets import DummySegmentationDataset
from .decoders import SegFormerDecoder
from .defaults import DEFAULTS
from .heads import SegmentationHead
from .losses import multitask_segmentation_losses, segmentation_loss
from .metrics import pixel_accuracy
from .models import DualDecoderSegFormer, DualHeadSegFormer, SingleTaskSegFormer
from .utils import set_seed
from .weighting import FixedLossWeighting


def build_model(
    variant: str = "mit_b0",
    task_mode: str = "single_task",
    num_classes: int = 4,
) -> nn.Module:
    spec = MIT_B0 if variant == MIT_B0.name else MIT_B2
    backbone = MixVisionTransformer(spec)
    decoder = SegFormerDecoder(spec.stage_channels)

    if task_mode == "single_task":
        return SingleTaskSegFormer(backbone, decoder, SegmentationHead(decoder.output_dim, num_classes))

    if task_mode == "dual_head":
        return DualHeadSegFormer(
            backbone,
            decoder,
            SegmentationHead(decoder.output_dim, num_classes),
            SegmentationHead(decoder.output_dim, num_classes),
        )

    if task_mode == "dual_decoder":
        return DualDecoderSegFormer(
            backbone,
            SegFormerDecoder(spec.stage_channels),
            SegFormerDecoder(spec.stage_channels),
            SegmentationHead(decoder.output_dim, num_classes),
            SegmentationHead(decoder.output_dim, num_classes),
        )

    raise ValueError(f"Unsupported task_mode={task_mode}.")


def build_optimizer(model: nn.Module) -> AdamW:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    return AdamW(
        [
            {"params": backbone_params, "lr": DEFAULTS.learning_rate_backbone},
            {"params": head_params, "lr": DEFAULTS.learning_rate_head},
        ],
        weight_decay=DEFAULTS.weight_decay,
    )


def run_smoke_test(
    variant: str = DEFAULTS.variant_name,
    task_mode: str = DEFAULTS.task_mode,
    num_classes: int = DEFAULTS.num_classes,
    image_size: tuple[int, int] = DEFAULTS.image_size,
    steps: int = DEFAULTS.steps,
) -> dict[str, float]:
    set_seed(DEFAULTS.seed)
    model = build_model(variant=variant, task_mode=task_mode, num_classes=num_classes)
    optimizer = build_optimizer(model)
    multitask = task_mode != "single_task"
    dataset = DummySegmentationDataset(
        length=max(steps, DEFAULTS.batch_size),
        image_size=image_size,
        num_classes=num_classes,
        multitask=multitask,
        seed=DEFAULTS.seed,
    )
    dataloader = DataLoader(dataset, batch_size=DEFAULTS.batch_size, shuffle=False)
    weighting = FixedLossWeighting({"task_a": 1.0, "task_b": 1.0})

    model.train()
    last_loss = 0.0
    last_accuracy = 0.0

    for step_index, batch in enumerate(tqdm(dataloader, total=steps, desc="smoke")):
        if step_index >= steps:
            break
        images = batch["image"]
        outputs = model(images)

        if task_mode == "single_task":
            loss = segmentation_loss(outputs["logits"], batch["mask"])
            accuracy = pixel_accuracy(outputs["logits"], batch["mask"])
        else:
            losses = multitask_segmentation_losses(
                outputs["task_a_logits"],
                batch["task_a_mask"],
                outputs["task_b_logits"],
                batch["task_b_mask"],
            )
            loss = weighting.reduce(losses)
            accuracy = pixel_accuracy(outputs["task_a_logits"], batch["task_a_mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        last_accuracy = float(accuracy.detach().cpu().item())

    return {"loss": last_loss, "pixel_accuracy": last_accuracy}


def main() -> None:
    result = run_smoke_test()
    print(asdict(DEFAULTS))
    print(result)


if __name__ == "__main__":
    main()

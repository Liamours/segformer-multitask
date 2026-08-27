import torch
from torch import nn

from .backbones import PyramidBackbone
from .decoders import SegFormerDecoder, SegFormerFusionHead, SegFormerProjectionTrunk
from .heads import SegmentationHead


class _FreezableBackboneModel(nn.Module):
    backbone: nn.Module

    def __init__(self) -> None:
        super().__init__()
        self._backbone_frozen = False

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self._backbone_frozen = True
        self.backbone.eval()

    def train(self, mode: bool = True) -> "_FreezableBackboneModel":
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self


class SingleTaskSegFormer(_FreezableBackboneModel):
    def __init__(
        self,
        backbone: PyramidBackbone,
        decoder: SegFormerDecoder,
        head: SegmentationHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.head = head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        decoded = self.decoder(features)
        return {"logits": self.head(decoded, output_size=x.shape[2:])}


class DualHeadSegFormer(_FreezableBackboneModel):
    def __init__(
        self,
        backbone: PyramidBackbone,
        decoder: SegFormerDecoder,
        head_a: SegmentationHead,
        head_b: SegmentationHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.head_a = head_a
        self.head_b = head_b

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        decoded = self.decoder(features)
        return {
            "task_a_logits": self.head_a(decoded, output_size=x.shape[2:]),
            "task_b_logits": self.head_b(decoded, output_size=x.shape[2:]),
        }


class DualDecoderSegFormer(_FreezableBackboneModel):
    def __init__(
        self,
        backbone: PyramidBackbone,
        decoder_a: SegFormerDecoder,
        decoder_b: SegFormerDecoder,
        head_a: SegmentationHead,
        head_b: SegmentationHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder_a = decoder_a
        self.decoder_b = decoder_b
        self.head_a = head_a
        self.head_b = head_b

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        decoded_a = self.decoder_a(features)
        decoded_b = self.decoder_b(features)
        return {
            "task_a_logits": self.head_a(decoded_a, output_size=x.shape[2:]),
            "task_b_logits": self.head_b(decoded_b, output_size=x.shape[2:]),
        }


class DualFuseSegFormer(_FreezableBackboneModel):
    def __init__(
        self,
        backbone: PyramidBackbone,
        trunk: SegFormerProjectionTrunk,
        fuse_a: SegFormerFusionHead,
        fuse_b: SegFormerFusionHead,
        head_a: SegmentationHead,
        head_b: SegmentationHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.trunk = trunk
        self.fuse_a = fuse_a
        self.fuse_b = fuse_b
        self.head_a = head_a
        self.head_b = head_b

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        projected = self.trunk(features)
        return {
            "task_a_logits": self.head_a(self.fuse_a(projected), output_size=x.shape[2:]),
            "task_b_logits": self.head_b(self.fuse_b(projected), output_size=x.shape[2:]),
        }

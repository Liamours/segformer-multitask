import torch
from torch import nn

from .backbones import PyramidBackbone
from .decoders import SegFormerDecoder
from .heads import SegmentationHead


class SingleTaskSegFormer(nn.Module):
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


class DualHeadSegFormer(nn.Module):
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


class DualDecoderSegFormer(nn.Module):
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

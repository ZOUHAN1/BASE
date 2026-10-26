import torch
from torch import nn
from ultralytics.nn.modules import C2f

from .adaptive_residual_relay import AdaptiveResidualRelay
from .background_referenced_reading import BackgroundReferencedReading
from .brsa_original import OriginalBRSA
from .geometry import valid_cells


class AsymmetricYOLOv8(nn.Module):
    """YOLOv8m port of the same BRR + BRSA + ARR neck used by YOLO11m."""

    def __init__(self, native):
        super().__init__()
        self.backbone = nn.Sequential(*list(native.model[:10]))
        self.brr4 = BackgroundReferencedReading(
            native.model[12], 16, [32], anchor_channels=384,
            source_channels=[576], semantic_channels=576)
        self.brr3 = BackgroundReferencedReading(
            native.model[15], 8, [16, 32], anchor_channels=192,
            source_channels=[384, 576], semantic_channels=384)
        self.brsa = OriginalBRSA(shallow_channels=96, semantic_channels=192)
        self.arr4 = AdaptiveResidualRelay(
            2, base_channels=384, correction_channels=192,
            refine=C2f(384, 384, n=2, shortcut=True))
        self.arr5 = AdaptiveResidualRelay(
            4, base_channels=576, correction_channels=192,
            refine=C2f(576, 576, n=2, shortcut=True))
        self.detect = native.model[-1]
        self.stride = self.detect.stride

    def forward(self, images, pixel_valid=None):
        features = {}
        x = images
        for index, layer in enumerate(self.backbone):
            x = layer(x)
            if index in (2, 4, 6, 9):
                features[index] = x
        c2, c3, c4, c5 = (features[index] for index in (2, 4, 6, 9))
        if pixel_valid is None:
            pixel_valid = images.new_ones((images.shape[0], 1, *images.shape[-2:]))
        mask8, mask16, mask32 = (
            valid_cells(pixel_valid, stride) for stride in (8, 16, 32)
        )
        middle4 = self.brr4(c4, [c5], [mask32])
        top3 = self.brr3(c3, [c4, c5], [mask16, mask32])
        enhanced3, _ = self.brsa(c2, top3, pixel_valid)
        correction = enhanced3 - top3
        p4 = self.arr4(middle4, correction, mask8)
        p5 = self.arr5(c5, correction, mask8)
        return self.detect([enhanced3, p4, p5])

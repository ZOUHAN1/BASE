"""Build the exact YOLOv8m graph used by the historical native/BRSA runs."""

from copy import deepcopy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "third_party/yolov8_ultralytics"))

from ultralytics.nn.modules import Conv  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from brsa import BRSA  # noqa: E402


def _annotate(module, index, source):
    module.i = index
    module.f = source
    module.type = type(module).__name__
    module.np = sum(parameter.numel() for parameter in module.parameters())
    return module


class ExperimentModel(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.adapters = [module for module in detector.modules() if isinstance(module, BRSA)]

    def forward(self, image, pixel_valid=None):
        for adapter in self.adapters:
            adapter.valid = pixel_valid
        result = self.detector(image)
        for adapter in self.adapters:
            adapter.valid = None
        return result

    def fuse(self):
        self.detector.fuse(verbose=False)
        return self


def _load_coco(detector, weights):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    source_model = checkpoint.get("ema") or checkpoint["model"]
    source = source_model.float().state_dict()
    current = detector.state_dict()
    loaded = []
    missing = []
    for key, value in current.items():
        if key in source and source[key].shape == value.shape:
            value.copy_(source[key])
            loaded.append(key)
        else:
            missing.append(key)
    head_index = len(detector.model) - 1
    if not all(key.startswith(f"model.{head_index}.cv3.") for key in missing):
        raise RuntimeError(f"incompatible COCO checkpoint: {missing}")
    return loaded, missing


def build_yolov8m(variant, nc, weights=None, seed=0):
    if variant not in ("native", "brsa"):
        raise ValueError(variant)
    torch.manual_seed(seed)
    detector = DetectionModel("yolov8m.yaml", nc=nc, verbose=False)

    # Ultralytics v8.3.0 uses the newer depthwise classification branch in the
    # generic Detect constructor. Historical runs restored the native YOLOv8
    # two-3x3-convolution classification branches before COCO transfer.
    head = detector.model[-1]
    channels = [detector.model[index].cv2.conv.out_channels for index in (15, 18, 21)]
    hidden = max(channels[0], min(nc, 100))
    head.cv3 = nn.ModuleList(
        nn.Sequential(
            Conv(channels_in, hidden, 3),
            Conv(hidden, hidden, 3),
            nn.Conv2d(hidden, nc, 1),
        )
        for channels_in in channels
    )
    head.bias_init()
    loaded, missing = _load_coco(detector, weights) if weights else ([], [])

    if variant == "brsa":
        old = deepcopy(detector.model)
        p3_index = 15
        shallow_channels = old[2].cv2.conv.out_channels
        semantic_channels = old[p3_index].cv2.conv.out_channels
        if (shallow_channels, semantic_channels) != (96, 192):
            raise RuntimeError((shallow_channels, semantic_channels))

        modules = list(detector.model)
        adapter_index = p3_index + 1
        modules.insert(
            adapter_index,
            _annotate(
                BRSA(shallow_channels, semantic_channels),
                adapter_index,
                [2, p3_index],
            ),
        )
        mapping = {
            index: index if index <= p3_index else index + 1
            for index in range(len(old))
        }
        for index, module in enumerate(modules):
            if index == adapter_index:
                continue
            source = module.f

            def remap(source_index):
                return source_index if source_index < 0 else mapping[source_index]

            module.i = index
            module.f = (
                [remap(value) for value in source]
                if isinstance(source, list)
                else remap(source)
            )
        modules[-1].f = [adapter_index, p3_index + 4, p3_index + 7]
        detector.model = nn.Sequential(*modules)
        detector.stride = detector.model[-1].stride
        detector.save = sorted(
            {
                source_index
                for module in modules
                for source_index in (
                    module.f if isinstance(module.f, list) else [module.f]
                )
                if source_index >= 0
            }
        )

    model = ExperimentModel(detector)
    provenance = {
        "variant": variant,
        "initialization": "coco_transfer" if weights else "random",
        "nc": nc,
        "coco_tensors_loaded": len(loaded),
        "classification_tensors_reinitialized": missing,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, provenance

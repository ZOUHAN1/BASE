"""Training contract for native YOLOv8m and the full paper BASE neck."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights

from .model import build_yolov8m
from .full_at.asymmetric_yolov8 import AsymmetricYOLOv8


class PairedYOLOv8(nn.Module):
    def __init__(self, native, arm):
        super().__init__()
        if arm not in ('baseline_native', 'full_asymmetric'):
            raise ValueError(arm)
        self.arm = arm
        self.network = native if arm == 'baseline_native' else AsymmetricYOLOv8(native)
        self.nc, self.names, self.yaml = native.model[-1].nc, native.names, native.yaml
        self.stride = self.network.stride
        self.args = None
        self.criterion = None
        self.lambda_fg = .1 if arm == 'full_asymmetric' else 0.
        self.last_shapes = None
        self.last_diagnostics = None
        self.loss_report = None
        self.deployment_variant = 'full'
        if arm == 'full_asymmetric':
            initialize_weights(self.network)

    @property
    def model(self):
        return self.network.model if self.arm == 'baseline_native' else (self.network.detect,)

    def _apply(self, fn):
        result = super()._apply(fn)
        self.stride = self.network.stride
        return result

    def predict(self, images, pixel_valid=None, diagnostics=False, return_aux=False):
        if self.arm == 'baseline_native':
            result = self.network(images)
            return (result, None) if return_aux else result
        return self.network(images, pixel_valid, return_aux, diagnostics)

    def forward(self, images, pixel_valid=None, diagnostics=False, **kwargs):
        if isinstance(images, dict):
            return self.loss(images)
        return self.predict(images, pixel_valid, diagnostics)

    def loss(self, batch, preds=None):
        if self.criterion is None:
            self.criterion = v8DetectionLoss(self)
        aux = None
        if preds is None:
            preds, aux = self.predict(batch['img'], batch.get('pixel_valid_mask'), return_aux=True)
        native_loss, items = self.criterion(preds, batch)
        foreground = (self.network.brsa.occupancy_loss(aux, batch) if aux is not None
                      else native_loss * 0)
        weighted = self.lambda_fg * foreground
        local_batch = batch['img'].shape[0]
        total = native_loss + local_batch * weighted
        self.loss_report = dict(native_batch_scaled=float(native_loss.detach()),
                                fg_raw=float(foreground.detach()), fg_weighted=float(weighted.detach()),
                                local_batch=local_batch, total_before_world_size=float(total.detach()))
        return total, torch.cat((items, weighted.detach().reshape(1)))

    def fingerprint(self):
        structure = [(key, list(value.shape)) for key, value in self.state_dict().items()]
        return hashlib.sha256(json.dumps([self.arm, structure]).encode()).hexdigest()


def create_initialization(weights, nc, output):
    native, report = build_yolov8m('native', nc, weights, seed=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=native.detector.state_dict(), nc=nc, seed=0,
                    coco_loading=report), output)
    return report


def load_paired(arm, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    torch.manual_seed(0)
    native, _ = build_yolov8m('native', checkpoint['nc'])
    native.detector.load_state_dict(checkpoint['state_dict'], strict=True)
    model = PairedYOLOv8(native.detector, arm)
    return model, dict(arm=arm, fingerprint=model.fingerprint(),
                       initialization='shared_COCO_detector_transfer',
                       native_state_tensors=len(checkpoint['state_dict']),
                       coco_loading=checkpoint['coco_loading'])

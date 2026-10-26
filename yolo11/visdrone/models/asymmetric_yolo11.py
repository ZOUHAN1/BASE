import copy
import hashlib
import json
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights
from .background_referenced_reading import BackgroundReferencedReading
from .adaptive_residual_relay import AdaptiveResidualRelay
from .brsa_original import OriginalBRSA
from .geometry import valid_cells


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_model(nc=10):
    cfg = yaml_model_load('yolo11m.yaml')
    return DetectionModel(cfg, nc=nc, verbose=False)


class PairedDetectionModel(nn.Module):
    def __init__(self, native, arm):
        super().__init__()
        self.arm = arm
        self.nc, self.names, self.yaml = native.model[-1].nc, native.names, native.yaml
        self.args = None
        self.criterion = None
        self.lambda_fg = .1 if arm == 'full_asymmetric' else 0.
        self.last_diagnostics = None
        self.last_shapes = None
        self.loss_report = None
        self.weight_mapping = {}
        if arm == 'baseline_native':
            self.native = native
            self.weight_mapping = {k: 'native.' + k for k in native.state_dict()}
        elif arm == 'full_asymmetric':
            self.backbone = nn.Sequential(*list(native.model[:11]))
            self.brr4 = BackgroundReferencedReading(native.model[13], 16, [32])
            self.brr3 = BackgroundReferencedReading(native.model[16], 8, [16, 32])
            self.brsa = OriginalBRSA()
            self.arr4, self.arr5 = AdaptiveResidualRelay(2), AdaptiveResidualRelay(4)
            self.detect = native.model[-1]
            for k in native.state_dict():
                layer = int(k.split('.')[1])
                suffix = '.'.join(k.split('.')[2:])
                if layer <= 10:
                    self.weight_mapping[k] = 'backbone.%d.%s' % (layer, suffix)
                elif layer in (13, 16):
                    self.weight_mapping[k] = ('brr4' if layer == 13 else 'brr3') + '.refine.' + suffix
                elif layer == 23:
                    self.weight_mapping[k] = 'detect.' + suffix
            initialize_weights(self)
        else:
            raise ValueError(arm)
        self.stride = self.model[-1].stride

    @property
    def model(self):
        # Criterion compatibility without registering Detect twice.
        return self.native.model if self.arm == 'baseline_native' else (self.detect,)

    def _apply(self, fn):
        result = super()._apply(fn)
        if self.arm == 'full_asymmetric':
            self.detect.stride = fn(self.detect.stride)
            self.detect.anchors = fn(self.detect.anchors)
            self.detect.strides = fn(self.detect.strides)
        self.stride = self.model[-1].stride
        return result

    def forward(self, x, pixel_valid=None, diagnostics=False, **kwargs):
        if isinstance(x, dict):
            return self.loss(x)
        return self.predict(x, pixel_valid, diagnostics)

    def predict(self, images, pixel_valid=None, diagnostics=False, return_aux=False):
        self.last_diagnostics = None
        if self.arm == 'baseline_native':
            return self.native(images)
        if pixel_valid is None:
            pixel_valid = torch.ones_like(images[:, :1], dtype=torch.float32)
        features = {}
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in (2, 4, 6, 10):
                features[i] = x
        c2, c3, c4, c5 = (features[i] for i in (2, 4, 6, 10))
        assert [v.shape[1] for v in (c2, c3, c4, c5)] == [256, 512, 512, 512]
        mask8, mask16, mask32 = (valid_cells(pixel_valid, s) for s in (8, 16, 32))
        m4 = self.brr4(c4, [c5], [mask32], diagnostics)
        t3 = self.brr3(c3, [c4, c5], [mask16, mask32], diagnostics)
        detail = {}
        if diagnostics:
            m4, detail['brr4'] = m4
            t3, detail['brr3'] = t3
        e3, aux = self.brsa(c2, t3, pixel_valid, return_aux, diagnostics)
        delta = e3 - t3
        p4, p5 = self.arr4(m4, delta, mask8, diagnostics), self.arr5(c5, delta, mask8, diagnostics)
        if diagnostics:
            p4, detail['arr4'] = p4
            p5, detail['arr5'] = p5
            detail['brsa'] = aux.pop('diagnostics')
            self.last_diagnostics = detail
        self.last_shapes = {k: list(v.shape) for k, v in
                            dict(C2=c2, C3=c3, C4=c4, C5=c5, M4=m4, T3=t3, E3=e3,
                                 delta=delta, P3=e3, P4=p4, P5=p5).items()}
        predictions = self.detect([e3, p4, p5])
        return (predictions, aux) if return_aux else predictions

    def loss(self, batch, preds=None):
        if self.criterion is None:
            self.criterion = v8DetectionLoss(self)
        aux = None
        if preds is None:
            if self.arm == 'full_asymmetric':
                preds, aux = self.predict(batch['img'], batch.get('pixel_valid_mask'), return_aux=True)
            else:
                preds = self.predict(batch['img'])
        native_loss, items = self.criterion(preds, batch)
        fg = self.brsa.occupancy_loss(aux, batch) if aux is not None else native_loss * 0
        weighted = self.lambda_fg * fg
        local_batch = batch['img'].shape[0]
        total = native_loss + local_batch * weighted
        self.loss_report = dict(native_batch_scaled=float(native_loss.detach()), fg_raw=float(fg.detach()),
                                fg_weighted=float(weighted.detach()), local_batch=local_batch,
                                total_before_world_size=float(total.detach()))
        return total, torch.cat((items, weighted.detach().reshape(1)))

    def fingerprint(self):
        structure = [(k, list(v.shape)) for k, v in self.state_dict().items()]
        return hashlib.sha256(json.dumps([self.arm, structure]).encode()).hexdigest()

    def loading_report(self, source):
        state = self.state_dict()
        checks = []
        for src, dst in self.weight_mapping.items():
            assert source[src].shape == state[dst].shape, (src, dst)
            assert torch.equal(source[src].cpu(), state[dst].cpu()), (src, dst)
            checks.append(dict(source=src, target=dst, shape=list(state[dst].shape), sha256=tensor_hash(state[dst])))
        return dict(arm=self.arm, mapped=checks,
                    new_keys=sorted(set(state) - set(self.weight_mapping.values())),
                    removed_keys=sorted(set(source) - set(self.weight_mapping)),
                    fingerprint=self.fingerprint())


def load_paired(arm, path):
    init = torch.load(path, map_location='cpu', weights_only=False)
    native = DetectionModel(init['yaml'], nc=init['nc'], verbose=False)
    native.load_state_dict(init['state_dict'], strict=True)
    native.names = init['names']
    model = PairedDetectionModel(native, arm)
    return model, model.loading_report(init['state_dict'])

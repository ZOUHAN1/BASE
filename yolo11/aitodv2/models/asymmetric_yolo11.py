import copy
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2PSA, C3k2, SPPF
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights
from .background_referenced_reading import BackgroundReferencedReading
from .adaptive_residual_relay import AdaptiveResidualRelay
from .brsa_original import OriginalBRSA
from .geometry import valid_cells
from .fused_c3k2_cpp import (bind_fast_arr_refine, bind_fast_brr_refine, bind_fast_c2psa,
                             bind_fast_c3k2, bind_fast_detect,
                             bind_fast_sppf,
                             extension as c3k2_cpp_extension)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_model():
    cfg = yaml_model_load('yolo11m.yaml')
    return DetectionModel(cfg, nc=8, verbose=False)


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
        self.last_features = None
        self.capture_runtime_state = True
        self.loss_report = None
        self.deployment_variant = 'full'
        self.parallel_inference = True
        self.parallel_detect_head = True
        self.overlap_arr_queries = True
        self.overlap_brsa_project = True
        self.fast_cpp_c3k2 = True
        self._branch_streams = {}
        self._head_streams = {}
        self._project_streams = {}
        self._valid_mask_cache = {}
        self.shared_backbone_projections = None
        self.shared_delta_key_projection = None
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
        self._branch_streams.clear()
        self._head_streams.clear()
        self._project_streams.clear()
        self._valid_mask_cache.clear()
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

    @staticmethod
    def _record_stream(value, stream):
        if isinstance(value, torch.Tensor):
            value.record_stream(stream)
        elif isinstance(value, (tuple, list)):
            for item in value:
                PairedDetectionModel._record_stream(item, stream)

    def _launch_pair(self, left, right, dependencies):
        device = dependencies[0].device
        key = (device.type, device.index)
        streams = self._branch_streams.get(key)
        if streams is None:
            streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
            self._branch_streams[key] = streams
        current = torch.cuda.current_stream(device)
        for stream in streams:
            stream.wait_stream(current)
            for tensor in dependencies:
                if tensor is not None:
                    tensor.record_stream(stream)
        with torch.cuda.stream(streams[0]):
            left_result = left()
        with torch.cuda.stream(streams[1]):
            right_result = right()
        return current, streams, left_result, right_result

    def _join_pair(self, current, streams, left_result, right_result):
        current.wait_stream(streams[0])
        current.wait_stream(streams[1])
        self._record_stream(left_result, current)
        self._record_stream(right_result, current)
        return left_result, right_result

    def _parallel_pair(self, left, right, dependencies):
        launched = self._launch_pair(left, right, dependencies)
        return self._join_pair(*launched)

    def _detect_level(self, index, feature):
        return torch.cat((self.detect.cv2[index](feature), self.detect.cv3[index](feature)), 1)

    def _parallel_detect_level(self, index, feature):
        if not self.parallel_detect_head:
            return self._detect_level(index, feature)
        device = feature.device
        key = (device.type, device.index, index)
        streams = self._head_streams.get(key)
        if streams is None:
            streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
            self._head_streams[key] = streams
        current = torch.cuda.current_stream(device)
        for stream in streams:
            stream.wait_stream(current)
            feature.record_stream(stream)
        with torch.cuda.stream(streams[0]):
            box = self.detect.cv2[index](feature)
        with torch.cuda.stream(streams[1]):
            classes = self.detect.cv3[index](feature)
        current.wait_stream(streams[0])
        current.wait_stream(streams[1])
        box.record_stream(current)
        classes.record_stream(current)
        return torch.cat((box, classes), 1)

    def _decode_detect_levels(self, levels):
        if self.detect.end2end:
            raise RuntimeError('Parallel inference does not support end-to-end Detect heads')
        decoded = self.detect._inference(levels)
        return decoded if self.detect.export else (decoded, levels)

    def set_parallel_inference_(self, enabled):
        self.parallel_inference = bool(enabled)
        if self.arm == 'full_asymmetric':
            self.brr3.parallel_inference = bool(enabled)
            self.brr4.parallel_inference = bool(enabled)
        return self

    def set_parallel_detect_head_(self, enabled):
        self.parallel_detect_head = bool(enabled)
        return self

    def predict(self, images, pixel_valid=None, diagnostics=False, return_aux=False):
        self.last_diagnostics = None
        if self.arm == 'baseline_native':
            return self.native(images)
        implicit_all_valid = pixel_valid is None
        inference_parallel = (self.parallel_inference and not self.training and images.is_cuda and
                              not torch.is_grad_enabled() and not diagnostics)
        features = {}
        pending_brsa_project = None
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in (2, 4, 6, 10):
                features[i] = x
            if i == 2 and inference_parallel and self.overlap_brsa_project:
                device = x.device
                key = (device.type, device.index)
                stream = self._project_streams.get(key)
                if stream is None:
                    stream = torch.cuda.Stream(device=device)
                    self._project_streams[key] = stream
                current = torch.cuda.current_stream(device)
                stream.wait_stream(current)
                x.record_stream(stream)
                with torch.cuda.stream(stream):
                    projected = self.brsa.project(x)
                pending_brsa_project = (current, stream, projected)
        c2, c3, c4, c5 = (features[i] for i in (2, 4, 6, 10))
        assert [v.shape[1] for v in (c2, c3, c4, c5)] == [256, 512, 512, 512]
        shared_brr4_query = None
        shared_brr3_sources = None
        shared_brr4_sources = None
        shared_arr5_query = None
        if self.shared_backbone_projections is not None:
            projected_c4 = self.shared_backbone_projections['c4'](c4)
            projected_c5 = self.shared_backbone_projections['c5'](c5)
            shared_brr4_query = F.silu(projected_c4[:, :32])
            shared_brr3_sources = (projected_c4[:, 32:], projected_c5[:, 96:192])
            shared_brr4_sources = (projected_c5[:, :96],)
            shared_arr5_query = F.silu(projected_c5[:, 192:])
        if implicit_all_valid:
            key = (images.shape[0], *c3.shape[-2:], *c4.shape[-2:], *c5.shape[-2:],
                   images.device.type, images.device.index)
            masks = self._valid_mask_cache.get(key)
            if masks is None:
                masks = tuple(torch.ones((images.shape[0], 1, *feature.shape[-2:]),
                                         dtype=torch.bool, device=images.device)
                              for feature in (c3, c4, c5))
                self._valid_mask_cache[key] = masks
            mask8, mask16, mask32 = masks
        else:
            mask8 = valid_cells(pixel_valid, 8)
            mask16 = valid_cells(mask8, 2)
            mask32 = valid_cells(mask16, 2)
        detail = {}
        parallel_levels = None
        arr_queries = None
        brsa_project = None
        if self.deployment_variant == 'speed_v2':
            # Parameter-free 512 -> 256 adapter used by the deployment student.
            # Pairwise averaging keeps both backbone channel halves active while
            # avoiding another pointwise convolution at stride 8.
            t3 = (c3[:, :256] + c3[:, 256:]) * .5
            m4 = self.brr4(c4, [c5], [mask32], diagnostics, skip_refine=True)
            if diagnostics:
                m4, detail['brr4'] = m4
        elif self.deployment_variant == 'speed_v3':
            # Keep the inexpensive BRR4 read and approximate BRR3 with the
            # already-trained input projection from its refinement block. This
            # preserves a top-down semantic path without BRR3's dense sampling.
            m4 = self.brr4(c4, [c5], [mask32], diagnostics, skip_refine=True)
            if diagnostics:
                m4, detail['brr4'] = m4
            p4_residual = c4 + m4
            context3 = F.interpolate(p4_residual, size=c3.shape[-2:], mode='nearest')
            adapter = self.brr3.adapter if hasattr(self.brr3, 'adapter') else self.brr3.refine.cv1
            t3 = adapter(torch.cat((context3, c3), 1))
        elif self.deployment_variant == 'speed_v4':
            # A cheap spatial gate carries BRR4 context into P3. Channel-pair
            # averaging replaces the large 1024 -> 256 projection while the
            # learned BRR3 query still decides where context is useful.
            m4 = self.brr4(c4, [c5], [mask32], diagnostics, skip_refine=True)
            if diagnostics:
                m4, detail['brr4'] = m4
            context3 = F.interpolate(m4, size=c3.shape[-2:], mode='nearest')
            base3 = (c3[:, :256] + c3[:, 256:]) * .5
            context3 = (context3[:, :256] + context3[:, 256:]) * .5
            gate3 = self.brr3.scale_logits(self.brr3.query(c3))[:, :1].sigmoid()
            t3 = base3 + gate3 * context3
        elif self.deployment_variant == 'speed_v5':
            # Low-rank BRR3 relay: retain its trained value/output projections
            # at 64 channels, then collapse paired output channels for P3.
            # This is substantially cheaper than the speed_v3 1024 -> 256
            # projection while preserving learned top-down semantics.
            m4 = self.brr4(c4, [c5], [mask32], diagnostics, skip_refine=True)
            if diagnostics:
                m4, detail['brr4'] = m4
            p4_residual = c4 + m4
            base3 = (c3[:, :256] + c3[:, 256:]) * .5
            value3 = self.brr3.values[0](p4_residual)
            value3 = F.interpolate(value3, size=c3.shape[-2:], mode='nearest')
            context3 = self.brr3.output(value3)
            context3 = (context3[:, :256] + context3[:, 256:]) * .5
            gate3 = self.brr3.scale_logits(self.brr3.query(c3))[:, :1].sigmoid()
            t3 = base3 + gate3 * context3
        elif self.deployment_variant == 'full':
            parallel = inference_parallel
            if parallel:
                pending_brr = self._launch_pair(
                    lambda: self.brr4(c4, [c5], [mask32], False,
                                      implicit_valid=implicit_all_valid,
                                      precomputed_query=shared_brr4_query,
                                      precomputed_sources=shared_brr4_sources),
                    lambda: self.brr3(c3, [c4, c5], [mask16, mask32], False,
                                      implicit_valid=implicit_all_valid,
                                      precomputed_sources=shared_brr3_sources),
                    (c3, c4, c5, mask16, mask32),
                )
                m4, t3 = self._join_pair(*pending_brr)
            else:
                m4 = self.brr4(c4, [c5], [mask32], diagnostics,
                               implicit_valid=implicit_all_valid,
                               precomputed_query=shared_brr4_query,
                               precomputed_sources=shared_brr4_sources)
                t3 = self.brr3(c3, [c4, c5], [mask16, mask32], diagnostics,
                               implicit_valid=implicit_all_valid,
                               precomputed_sources=shared_brr3_sources)
            if diagnostics:
                m4, detail['brr4'] = m4
                t3, detail['brr3'] = t3
        else:
            raise ValueError('Unknown deployment variant: %s' % self.deployment_variant)
        if (self.deployment_variant == 'full' and parallel and
                self.overlap_arr_queries and
                shared_arr5_query is None):
            pending_arr_queries = self._launch_pair(
                lambda: self.arr4.query(m4),
                lambda: self.arr5.query(c5),
                (m4, c5),
            )
        else:
            pending_arr_queries = None
        if pending_brsa_project is not None:
            current, stream, brsa_project = pending_brsa_project
            current.wait_stream(stream)
            brsa_project.record_stream(current)
        e3, aux, delta = self.brsa(
            c2, t3, pixel_valid, return_aux, diagnostics, block_valid=mask16,
            return_recovered_delta=True, precomputed_project=brsa_project)
        shared_delta_keys = (self.shared_delta_key_projection(delta)
                             if self.shared_delta_key_projection is not None else None)
        shared_arr4_key = (shared_delta_keys[:, :32]
                           if shared_delta_keys is not None else None)
        shared_arr5_key = (shared_delta_keys[:, 32:]
                           if shared_delta_keys is not None else None)
        if pending_arr_queries is not None:
            arr_queries = self._join_pair(*pending_arr_queries)
        elif shared_arr5_query is not None:
            arr_queries = (None, shared_arr5_query)
        if self.deployment_variant == 'speed_v2':
            p4 = m4
            p5 = self.arr5(c5, delta, mask8, diagnostics, skip_refine=True)
            if diagnostics:
                p5, detail['arr5'] = p5
        elif self.deployment_variant == 'speed_v3':
            p4 = p4_residual
            p5 = self.arr5(c5, delta, mask8, diagnostics, skip_refine=True)
            if diagnostics:
                p5, detail['arr5'] = p5
        elif self.deployment_variant == 'speed_v4':
            p4 = m4
            p5 = self.arr5(c5, delta, mask8, diagnostics, skip_refine=True)
            if diagnostics:
                p5, detail['arr5'] = p5
        elif self.deployment_variant == 'speed_v5':
            p4 = p4_residual
            p5 = self.arr5(c5, delta, mask8, diagnostics, skip_refine=True)
            if diagnostics:
                p5, detail['arr5'] = p5
        else:
            if parallel:
                def p4_branch():
                    result = self.arr4(m4, delta, mask8, False,
                                       implicit_valid=implicit_all_valid,
                                       precomputed_query=arr_queries[0] if arr_queries else None,
                                       precomputed_key=shared_arr4_key)
                    return result, self._detect_level(1, result)

                def p5_branch():
                    result = self.arr5(
                        c5, delta, mask8, False, implicit_valid=implicit_all_valid,
                        precomputed_query=arr_queries[1] if arr_queries else None,
                        precomputed_key=shared_arr5_key)
                    return result, self._parallel_detect_level(2, result)

                launched = self._launch_pair(
                    p4_branch,
                    p5_branch,
                    (m4, c5, delta, mask8, *(arr_queries or ())),
                )
                p3_level = self._detect_level(0, e3)
                (p4, p4_level), (p5, p5_level) = self._join_pair(*launched)
                parallel_levels = [p3_level, p4_level, p5_level]
            else:
                p4 = self.arr4(m4, delta, mask8, diagnostics,
                               implicit_valid=implicit_all_valid,
                               precomputed_key=shared_arr4_key)
                p5 = self.arr5(c5, delta, mask8, diagnostics,
                               implicit_valid=implicit_all_valid,
                               precomputed_query=shared_arr5_query,
                               precomputed_key=shared_arr5_key)
            if diagnostics:
                p4, detail['arr4'] = p4
                p5, detail['arr5'] = p5
        if diagnostics:
            detail['brsa'] = aux.pop('diagnostics')
            self.last_diagnostics = detail
        if self.capture_runtime_state:
            self.last_shapes = {k: list(v.shape) for k, v in
                                dict(C2=c2, C3=c3, C4=c4, C5=c5, M4=m4, T3=t3, E3=e3,
                                     delta=delta, P3=e3, P4=p4, P5=p5).items()}
            self.last_features = dict(P3=e3, P4=p4, P5=p5)
        predictions = (self._decode_detect_levels(parallel_levels) if parallel_levels is not None
                       else self.detect([e3, p4, p5]))
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
        distill = native_loss * 0
        teacher = getattr(self, '_distillation_teacher', None)
        distill_weight = float(getattr(self, '_distillation_weight', 0.))
        if teacher is not None and distill_weight > 0:
            student_features = self.last_features
            with torch.no_grad():
                teacher.predict(batch['img'], batch.get('pixel_valid_mask'))
            distill = torch.stack([
                F.smooth_l1_loss(student_features[key].float(),
                                 teacher.last_features[key].float(), beta=.1)
                for key in ('P3', 'P4', 'P5')
            ]).mean()
            teacher.last_features = None
        local_batch = batch['img'].shape[0]
        total = native_loss + local_batch * (weighted + distill_weight * distill)
        self.loss_report = dict(native_batch_scaled=float(native_loss.detach()), fg_raw=float(fg.detach()),
                                fg_weighted=float(weighted.detach()), local_batch=local_batch,
                                distill_raw=float(distill.detach()), distill_weight=distill_weight,
                                total_before_world_size=float(total.detach()))
        self.last_features = None
        return total, torch.cat((items, weighted.detach().reshape(1)))

    def attach_distillation_teacher(self, teacher, weight):
        # Keep the teacher outside Module registration so it is absent from EMA,
        # optimizer groups, checkpoints, and deployment state dictionaries.
        object.__setattr__(self, '_distillation_teacher', teacher)
        object.__setattr__(self, '_distillation_weight', float(weight))

    def prune_speed_v2_(self):
        if self.arm != 'full_asymmetric' or self.deployment_variant != 'speed_v2':
            raise ValueError('speed_v2 pruning requires the full_asymmetric speed_v2 path')
        self.brr3 = nn.Identity()
        self.arr4 = nn.Identity()
        self.brr4.refine = nn.Identity()
        self.arr5.refine = nn.Identity()
        return self

    def prune_speed_v3_(self):
        if self.arm != 'full_asymmetric' or self.deployment_variant != 'speed_v3':
            raise ValueError('speed_v3 pruning requires the full_asymmetric speed_v3 path')
        adapter = self.brr3.refine.cv1
        self.brr3 = nn.Module()
        self.brr3.add_module('adapter', adapter)
        self.arr4 = nn.Identity()
        self.brr4.refine = nn.Identity()
        self.arr5.refine = nn.Identity()
        return self

    def prune_speed_v4_(self):
        if self.arm != 'full_asymmetric' or self.deployment_variant != 'speed_v4':
            raise ValueError('speed_v4 pruning requires the full_asymmetric speed_v4 path')
        query, scale_logits = self.brr3.query, self.brr3.scale_logits
        self.brr3 = nn.Module()
        self.brr3.add_module('query', query)
        self.brr3.add_module('scale_logits', scale_logits)
        self.arr4 = nn.Identity()
        self.brr4.refine = nn.Identity()
        self.arr5.refine = nn.Identity()
        return self

    def prune_speed_v5_(self):
        if self.arm != 'full_asymmetric' or self.deployment_variant != 'speed_v5':
            raise ValueError('speed_v5 pruning requires the full_asymmetric speed_v5 path')
        query = self.brr3.query
        value = self.brr3.values[0]
        output = self.brr3.output
        scale_logits = self.brr3.scale_logits
        self.brr3 = nn.Module()
        self.brr3.add_module('query', query)
        self.brr3.add_module('values', nn.ModuleList([value]))
        self.brr3.add_module('output', output)
        self.brr3.add_module('scale_logits', scale_logits)
        self.arr4 = nn.Identity()
        self.brr4.refine = nn.Identity()
        self.arr5.refine = nn.Identity()
        return self

    def prune_deployment_(self):
        if self.deployment_variant == 'speed_v2':
            return self.prune_speed_v2_()
        if self.deployment_variant == 'speed_v3':
            return self.prune_speed_v3_()
        if self.deployment_variant == 'speed_v4':
            return self.prune_speed_v4_()
        if self.deployment_variant == 'speed_v5':
            return self.prune_speed_v5_()
        raise ValueError('No pruning recipe for deployment variant: %s' % self.deployment_variant)

    def fuse_for_inference_(self, bind_fast=True):
        """Fuse every eligible Conv-BN pair without changing the deployed graph."""
        if self.training:
            raise RuntimeError('Inference fusion requires eval mode')
        self.capture_runtime_state = False
        if self.arm == 'baseline_native':
            self.native.fuse(verbose=False)
            if bind_fast:
                self._bind_fast_c3k2_()
            return self

        # Match Ultralytics BaseModel.fuse() for the copied backbone and head.
        for module in self.modules():
            module_name = type(module).__name__
            if module_name in ('Conv', 'Conv2', 'DWConv') and hasattr(module, 'bn'):
                if module_name == 'Conv2':
                    module.fuse_convs()
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, 'bn')
                module.forward = module.forward_fuse
            elif module_name == 'RepConv' and hasattr(module, 'fuse_convs'):
                module.fuse_convs()
                module.forward = module.forward_fuse
            elif module_name == 'RepVGGDW' and hasattr(module, 'fuse'):
                module.fuse()
                module.forward = module.forward_fuse

        # BRR/ARR projections use plain Sequential(Conv2d, BatchNorm2d, SiLU).
        for module in self.modules():
            if not isinstance(module, nn.Sequential):
                continue
            for index in range(len(module) - 1):
                conv, norm = module[index], module[index + 1]
                if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                    module[index] = fuse_conv_and_bn(conv, norm)
                    module[index + 1] = nn.Identity()
        self.brr3.fuse_source_projections_()
        self.brr4.fuse_source_projections_()
        if self.deployment_variant == 'full':
            self._fuse_shared_inference_projections_()
            if bind_fast:
                self._bind_fast_c3k2_()
        return self

    def bind_fast_eager_(self):
        """Bind eager C++ scheduling after all parameter dtype conversions."""
        if self.training:
            raise RuntimeError('Fast eager binding requires eval mode')
        return self._bind_fast_c3k2_()

    def _bind_fast_c3k2_(self):
        if (not self.fast_cpp_c3k2 or not next(self.parameters()).is_cuda or
                c3k2_cpp_extension() is None):
            return self
        for module in list(self.modules()):
            if isinstance(module, C3k2) and not hasattr(module, '_native_forward'):
                bind_fast_c3k2(module)
            elif isinstance(module, SPPF) and not hasattr(module, '_native_forward'):
                bind_fast_sppf(module)
            elif isinstance(module, C2PSA) and not hasattr(module, '_native_forward'):
                bind_fast_c2psa(module)
        detect = self.native.model[-1] if self.arm == 'baseline_native' else self.detect
        bind_fast_detect(detect)
        if self.arm == 'full_asymmetric':
            bind_fast_brr_refine(self.brr3)
            bind_fast_brr_refine(self.brr4)
            bind_fast_arr_refine(self.arr4)
            bind_fast_arr_refine(self.arr5)
        return self

    @staticmethod
    def _concatenate_pointwise_convolutions(convolutions):
        first = convolutions[0]
        if not all(isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and
                   module.in_channels == first.in_channels for module in convolutions):
            raise RuntimeError('Shared projection fusion requires compatible 1x1 convolutions')
        use_bias = any(module.bias is not None for module in convolutions)
        projection = nn.Conv2d(
            first.in_channels, sum(module.out_channels for module in convolutions), 1,
            bias=use_bias).to(device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            projection.weight.copy_(torch.cat([module.weight for module in convolutions], 0))
            if use_bias:
                biases = [(module.bias if module.bias is not None else
                           torch.zeros(module.out_channels, device=module.weight.device,
                                       dtype=module.weight.dtype))
                          for module in convolutions]
                projection.bias.copy_(torch.cat(biases))
        return projection

    def _fuse_shared_inference_projections_(self):
        """Coalesce projections that consume the same feature map in deployed inference."""
        if self.shared_backbone_projections is not None:
            return self
        brr4_query = self.brr4.query[0]
        arr5_query = self.arr5.query[0]
        c4_projection = self._concatenate_pointwise_convolutions(
            (brr4_query, self.brr3.combined_projections[0]))
        c5_projection = self._concatenate_pointwise_convolutions(
            (self.brr4.combined_projections[0], self.brr3.combined_projections[1], arr5_query))
        delta_key_projection = self._concatenate_pointwise_convolutions(
            (self.arr4.key, self.arr5.key))
        self.shared_backbone_projections = nn.ModuleDict(
            {'c4': c4_projection, 'c5': c5_projection})
        self.shared_delta_key_projection = delta_key_projection

        # These weights now live in the shared projections. Removing the old
        # modules prevents duplicate parameters in the deployed model.
        self.brr4.query = nn.Identity()
        self.brr4.combined_projections = nn.ModuleList([nn.Identity()])
        self.brr3.combined_projections = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.arr5.query = nn.Identity()
        self.arr4.key = nn.Identity()
        self.arr5.key = nn.Identity()
        return self

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
    nc = init.get('nc', init['yaml'].get('nc', 8))
    native = DetectionModel(init['yaml'], nc=nc, verbose=False)
    native.load_state_dict(init['state_dict'], strict=True)
    if 'names' in init:
        native.names = init['names']
    model = PairedDetectionModel(native, arm)
    return model, model.loading_report(init['state_dict'])

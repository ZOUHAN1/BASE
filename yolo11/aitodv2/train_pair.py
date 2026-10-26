import argparse
import bootstrap
from bootstrap import ROOT
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
import yaml
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK, LOCAL_RANK
from ultralytics.utils.torch_utils import de_parallel, init_seeds
from models.asymmetric_yolo11 import load_paired
from data_pipeline import CocoGeometryDataset, loader
from eval_aitodv2 import evaluate_model
from common import dump_json, sha256, cuda_uuid


TRAIN_DEFAULTS = dict(task='detect', model='yolo11m.yaml', epochs=300, imgsz=800,
    batch=16, nbs=16, optimizer='AdamW', lr0=.001, lrf=.01, momentum=.9, weight_decay=.0005,
    warmup_epochs=3., warmup_bias_lr=0., cos_lr=True, patience=0, time=None,
    seed=0, deterministic=False, workers=4, cache=False, rect=False, multi_scale=False,
    close_mosaic=10, freeze=None, amp=True, save=True, save_period=-1, plots=False,
    val=True, max_det=1500, conf=.001, iou=.7, agnostic_nms=False, exist_ok=True,
    pretrained=False, device='0,1', fliplr=.5, flipud=0., hsv_h=.015, hsv_s=.7,
    hsv_v=.4, translate=.1, scale=.5, degrees=0., shear=0., perspective=0.,
    mosaic=1., mixup=0., copy_paste=0., bgr=0., fraction=1.)


def resolve_run_layout(spec, smoke=False):
    epochs = 2 if smoke else int(spec.get('epochs', TRAIN_DEFAULTS['epochs']))
    assert epochs > 0, 'epochs must be positive'
    output_group = 'smoke' if smoke else str(spec.get('output_group', 'runs'))
    project = Path(output_group)
    assert not project.is_absolute() and '..' not in project.parts, 'output_group must stay under ROOT'
    name_key = 'smoke_name' if smoke else 'run_name'
    name = str(spec.get(name_key, spec['arm']))
    assert name and Path(name).name == name, '%s must be a single directory name' % name_key
    return ROOT / project, name, epochs


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    numpy_state = list(state['numpy'])
    if isinstance(numpy_state[1], list):
        numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
    np.random.set_state(tuple(numpy_state))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


class PairTrainer(DetectionTrainer):
    def __init__(self, spec, smoke=False, resume=None):
        self.spec, self.smoke, self.resume_path = spec, smoke, resume
        self.data_root = Path(spec['data_root'])
        self.arm = spec['arm']
        self.paired_init = Path(spec.get('paired_init', ROOT / 'weights/paired_native_init.pt'))
        self.pretrained_weights = Path(spec.get('pretrained_weights', ROOT / 'weights/yolo11m.pt'))
        self.deployment_variant = str(spec.get('deployment_variant', 'full'))
        if self.arm == 'full_asymmetric' and self.deployment_variant == 'speed_v2':
            self.optimizer_excluded_prefixes = ('brr3', 'arr4', 'brr4.refine', 'arr5.refine')
        elif self.arm == 'full_asymmetric' and self.deployment_variant == 'speed_v3':
            self.optimizer_excluded_prefixes = (
                'brr3.query', 'brr3.keys', 'brr3.values', 'brr3.controllers',
                'brr3.scale_logits', 'brr3.output', 'brr3.refine.cv2', 'brr3.refine.m',
                'arr4', 'brr4.refine', 'arr5.refine',
            )
        elif self.arm == 'full_asymmetric' and self.deployment_variant == 'speed_v4':
            self.optimizer_excluded_prefixes = (
                'brr3.keys', 'brr3.values', 'brr3.controllers', 'brr3.output',
                'brr3.refine', 'arr4', 'brr4.refine', 'arr5.refine',
            )
        elif self.arm == 'full_asymmetric' and self.deployment_variant == 'speed_v5':
            self.optimizer_excluded_prefixes = (
                'brr3.keys', 'brr3.values.1', 'brr3.controllers', 'brr3.refine',
                'arr4', 'brr4.refine', 'arr5.refine',
            )
        else:
            self.optimizer_excluded_prefixes = ()
        project, name, epochs = resolve_run_layout(spec, smoke)
        tunable = {key: value for key, value in spec.items()
                   if key in TRAIN_DEFAULTS and key != 'epochs'}
        overrides = dict(TRAIN_DEFAULTS, **tunable, data=str(self.data_root / 'aitodv2.yaml'),
                         project=str(project), name=name, epochs=epochs)
        self.optimizer_steps = 0
        self.optimizer_attempts = 0
        self.saved_best_epoch = None
        self.new_best = False
        self.current_eval = None
        self.all_rng = None
        self.batch_number = 0
        self.epoch_start = None
        self.start_wall = time.time()
        super().__init__(overrides=overrides)
        # External integrations must not change the controlled trainer or contact external trackers.
        self.callbacks = {key: [] for key in self.callbacks}
        self.add_callback('on_pretrain_routine_end', self.pretrain_complete)
        self.add_callback('on_train_epoch_start', self.epoch_begin)
        self.add_callback('on_train_batch_end', self.batch_complete)
        self.add_callback('on_train_epoch_end', self.collect_rank_state)
        self.add_callback('on_fit_epoch_end', self.epoch_complete)

    def setup_model(self):
        init_seeds(0, deterministic=False)
        factory = load_paired
        if self.spec.get('model_family') == 'yolov8m':
            from yolov8.training import load_paired as factory
        self.model, report = factory(self.arm, self.paired_init)
        self.model.deployment_variant = self.deployment_variant
        self.expected_fingerprint = self.model.fingerprint()
        if self.resume_path:
            checkpoint = torch.load(self.resume_path, map_location='cpu', weights_only=False)
            assert checkpoint['arm'] == self.arm
            assert checkpoint['fingerprint'] == self.expected_fingerprint
            assert checkpoint.get('deployment_variant', 'full') == self.deployment_variant
            self.model.load_state_dict(checkpoint['model_state'], strict=True)
        else:
            checkpoint = None
            initial_path = self.spec.get('initial_checkpoint')
            if initial_path:
                initial_path = Path(initial_path)
                initial_path = initial_path if initial_path.is_absolute() else ROOT / initial_path
                initial = torch.load(initial_path, map_location='cpu', weights_only=False)
                assert initial['arm'] == self.arm
                assert initial['fingerprint'] == self.expected_fingerprint
                self.model.load_state_dict(initial['ema_state'], strict=True)
                report['initial_checkpoint'] = str(initial_path)
                report['initial_epoch'] = initial['epoch'] + 1
                report['initial_best_fitness'] = initial['best_fitness']
        excluded = self.optimizer_excluded_prefixes
        optimized_parameters = sum(parameter.numel() for key, parameter in self.model.named_parameters()
                                   if not any(key == prefix or key.startswith(prefix + '.')
                                              for prefix in excluded))
        if RANK in (-1, 0):
            dump_json(self.save_dir / 'weight_loading_report.json', report)
            dump_json(self.save_dir / 'architecture.json', dict(arm=self.arm,
                      deployment_variant=self.deployment_variant,
                      optimizer_excluded_modules=list(excluded),
                      fingerprint=self.expected_fingerprint, strides=self.model.stride.tolist(),
                      parameters=sum(p.numel() for p in self.model.parameters()),
                      optimized_parameters=optimized_parameters,
                      flops=None, flops_note='custom grid_sample and DCT operators not counted'))
        print('TRAIN_IDENTITY', RANK, self.arm, self.deployment_variant,
              self.expected_fingerprint, flush=True)
        return checkpoint

    def build_dataset(self, img_path, mode='train', batch=None):
        return CocoGeometryDataset(self.data_root / ('aitodv2_%s.json' % ('train' if mode == 'train' else 'val')),
                 img_path=str(self.data_root / 'smoke_train.txt') if self.smoke and mode == 'train' else img_path,
                 data=self.data, imgsz=800, batch_size=batch, augment=mode == 'train',
                 hyp=copy.copy(self.args), rect=False, cache=False, stride=32, pad=0,
                 prefix=mode + ': ', task='detect')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        data_loader, _ = loader(dataset, batch_size if mode == 'train' else 1,
                                self.args.workers, rank if mode == 'train' else -1,
                                int(os.environ.get('WORLD_SIZE', 1)))
        return data_loader

    def get_validator(self):
        self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss', 'fg_loss')
        return SimpleNamespace(metrics=SimpleNamespace(keys=['native/AP']))

    def build_optimizer(self, model, name='AdamW', lr=.001, momentum=.9, decay=.0005, iterations=1):
        groups, report, seen = [[], []], [], set()
        for key, parameter in de_parallel(model).named_parameters():
            if not parameter.requires_grad:
                continue
            if any(key == prefix or key.startswith(prefix + '.')
                   for prefix in self.optimizer_excluded_prefixes):
                continue
            assert id(parameter) not in seen
            seen.add(id(parameter))
            no_decay = parameter.ndim < 2 or key.endswith('.bias') or key.endswith('.tau') or key.endswith('position_bias')
            groups[0 if no_decay else 1].append(parameter)
            report.append(dict(parameter=key, shape=list(parameter.shape), decay=0. if no_decay else decay))
        expected = {id(parameter) for key, parameter in de_parallel(model).named_parameters()
                    if parameter.requires_grad and not any(
                        key == prefix or key.startswith(prefix + '.')
                        for prefix in self.optimizer_excluded_prefixes)}
        assert seen == expected
        optimizer = torch.optim.AdamW([
            dict(params=groups[0], weight_decay=0.), dict(params=groups[1], weight_decay=decay)],
            lr=lr, betas=(.9, .999), eps=1e-8)
        if RANK in (-1, 0):
            dump_json(self.save_dir / 'optimizer_groups.json', report)
        return optimizer

    def _setup_scheduler(self):
        warmup = 3
        total = self.epochs
        self.lf = lambda epoch: (1. if epoch <= warmup else .01 + .99 *
                   (1 + np.cos(np.pi * min((epoch - warmup) / max(total - 1 - warmup, 1), 1.))) / 2)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def resume_training(self, checkpoint):
        if checkpoint is None:
            return
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer_steps = checkpoint['optimizer_steps']
        self.optimizer_attempts = checkpoint.get('optimizer_attempts', self.optimizer_steps)
        self.best_fitness = checkpoint['best_fitness']
        self.saved_best_epoch = checkpoint['best_epoch']
        if self.ema:
            self.ema.ema.load_state_dict(checkpoint['ema_state'], strict=True)
            self.ema.updates = checkpoint['ema_updates']
        self.resume_rng = checkpoint['rank_states'][max(RANK, 0)]

    def pretrain_complete(self, trainer):
        assert self.accumulate == 1 and self.args.batch == self.args.nbs == 16
        visible = ','.join(gpu['uuid'] for gpu in self.spec['gpus'])
        assert os.environ.get('BRSA_PAIR_VISIBLE_UUIDS') == visible
        actual_uuid = cuda_uuid(torch.cuda.current_device())
        assert actual_uuid == self.spec['gpus'][LOCAL_RANK]['uuid'], (RANK, actual_uuid, visible)
        dump_json(self.save_dir / ('diagnostics/device_rank%d.json' % RANK),
                  dict(pid=os.getpid(), actual_uuid=actual_uuid, logical_cuda_index=LOCAL_RANK,
                       physical_index=self.spec['gpus'][LOCAL_RANK]['physical_index'], visible=visible))
        print('GPU_UUID_VERIFIED', RANK, actual_uuid, flush=True)
        init_seeds(0 + max(RANK, 0), deterministic=False)
        if self.resume_path:
            restore_rng(self.resume_rng['rng'])
            self.train_loader.generator.set_state(self.resume_rng['loader_generator'])
        assert de_parallel(self.model).fingerprint() == self.expected_fingerprint
        if self.ema:
            assert self.ema.ema.fingerprint() == self.expected_fingerprint
        distill_weight = float(self.spec.get('distillation_weight', 0.))
        if distill_weight > 0:
            assert self.deployment_variant != 'full'
            teacher_path = Path(self.spec['initial_checkpoint'])
            teacher_checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=False)
            teacher, _ = load_paired(self.arm, ROOT / 'weights/paired_native_init.pt')
            teacher.load_state_dict(teacher_checkpoint['ema_state'], strict=True)
            teacher.deployment_variant = 'full'
            teacher.requires_grad_(False).to(self.device).eval()
            de_parallel(self.model).attach_distillation_teacher(teacher, distill_weight)
            print('DISTILLATION_TEACHER_READY', RANK, str(teacher_path),
                  'weight', distill_weight, flush=True)
        self.csv = self.save_dir / 'epoch_metrics.csv'
        if RANK in (-1, 0):
            (self.save_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(vars(self.args)))
            manifest = json.loads((ROOT / 'artifacts/environment.json').read_text())
            manifest.update(pair_id=self.spec['pair_id'], arm=self.arm,
                data=json.loads((self.data_root / 'manifest.json').read_text()),
                gpu_mapping=self.spec['gpus'], ddp_world_size=2, port=self.spec['port'],
                physical_batch_per_rank=8, effective_batch=16, accumulation=1,
                paired_init_sha256=sha256(self.paired_init),
                pretrained_sha256=sha256(self.pretrained_weights),
                grad_clip=10., ema_decay=0.9999, ema_tau=2000,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)
            dump_json(self.save_dir / 'manifest.json', manifest)
        print('READY', RANK, self.arm, 'AMP', self.amp, 'batch/rank', 8, 'accumulate', self.accumulate, flush=True)

    def epoch_begin(self, trainer):
        self.epoch_start = time.time()
        self.batch_number = 0
        self.train_loader.dataset.shared_epoch.value = self.epoch

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        batch['pixel_valid_mask'] = batch['pixel_valid_mask'].to(self.device, non_blocking=True)
        if self.epoch == 0 and self.batch_number < 3:
            record = dict(rank=RANK, batch=self.batch_number, image_ids=batch['image_id'],
                 boxes_sha256=hashlib.sha256(batch['bboxes'].numpy().tobytes()).hexdigest(),
                 pixels_sha256=hashlib.sha256(batch['img'].cpu().numpy().tobytes()).hexdigest(),
                 mask_valid=float(batch['pixel_valid_mask'].float().mean()))
            dump_json(self.save_dir / ('diagnostics/batch_r%d_%d.json' % (RANK, self.batch_number)), record)
            if self.batch_number == 0:
                image = (batch['img'][0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8).copy()
                import cv2
                mask = batch['pixel_valid_mask'][0, 0].cpu().numpy()
                image[mask == 0] = [80, 0, 80]
                for box in batch['bboxes'][batch['batch_idx'] == 0].numpy():
                    cx, cy, w, h = box * 800
                    cv2.rectangle(image, (int(cx - w/2), int(cy - h/2)),
                                  (int(cx + w/2), int(cy + h/2)), (0, 255, 0), 1)
                cv2.imwrite(str(self.save_dir / ('diagnostics/mask_boxes_r%d.jpg' % RANK)), image)
        return batch

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        modules = {}
        if self.optimizer_steps < 6:
            for key, parameter in de_parallel(self.model).named_parameters():
                if parameter.requires_grad:
                    module = key.split('.')[0]
                    entry = modules.setdefault(module, dict(norm_sum=0., missing=[], nonzero=0, params=0,
                                                             parameter_norms={}, nonfinite=[]))
                    entry['params'] += 1
                    if parameter.grad is None:
                        entry['missing'].append(key)
                    else:
                        norm = float(parameter.grad.float().norm())
                        if np.isfinite(norm):
                            entry['norm_sum'] += norm
                            entry['nonzero'] += norm > 0
                            entry['parameter_norms'][key] = norm
                        else:
                            entry['nonfinite'].append(key)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self.optimizer_attempts += 1
        successful = self.scaler.get_scale() >= old_scale
        if modules:
            if successful:
                dump_json(self.save_dir / ('diagnostics/grad_r%d_step%d.json' % (RANK, self.optimizer_steps)), modules)
            else:
                dump_json(self.save_dir / ('diagnostics/amp_overflow_r%d_attempt%d.json' % (RANK, self.optimizer_attempts)),
                          dict(old_scale=old_scale, new_scale=self.scaler.get_scale(), modules=modules))
        self.optimizer_steps += int(successful)
        if self.optimizer_steps <= 6:
            print('OPTIMIZER_STEP', RANK, self.optimizer_steps, 'scaler', old_scale,
                  self.scaler.get_scale(), 'successful', successful,
                  'loss', de_parallel(self.model).loss_report, flush=True)

    def batch_complete(self, trainer):
        self.batch_number += 1
        assert self.accumulate == 1
        assert torch.isfinite(self.loss).all(), 'Nonfinite loss'

    def collect_rank_state(self, trainer):
        state = dict(rank=RANK, optimizer_steps=self.optimizer_steps, rng=rng_state(),
                     loader_generator=self.train_loader.generator.get_state())
        gathered = [None] * dist.get_world_size() if dist.is_initialized() else [state]
        if dist.is_initialized():
            dist.all_gather_object(gathered, state)
        assert len(set(x['optimizer_steps'] for x in gathered)) == 1
        self.all_rng = gathered

    def validate(self):
        assert self.ema.ema.fingerprint() == self.expected_fingerprint
        interval = int(self.spec.get('val_interval', 1))
        should_validate = self.epoch == 0 or (self.epoch + 1) % interval == 0 or self.epoch + 1 >= self.epochs
        if not should_validate:
            self.new_best = False
            self.current_eval = None
            print('VALIDATION_SKIPPED', self.epoch + 1, 'interval', interval, flush=True)
            return {}, self.best_fitness
        metrics = evaluate_model(self.ema.ema, self.data_root / 'aitodv2_val.json',
             Path(json.loads((self.data_root / 'manifest.json').read_text())['image_root']) / 'val',
             self.save_dir / 'metrics/val_last', amp=self.amp)
        ap = metrics['AP']
        assert ap is not None
        self.new_best = self.best_fitness is None or ap > self.best_fitness
        if self.new_best:
            self.best_fitness, self.saved_best_epoch = ap, self.epoch + 1
            shutil.copytree(self.save_dir / 'metrics/val_last', self.save_dir / 'metrics/val_best', dirs_exist_ok=True)
        self.current_eval = metrics
        return {'native/' + k: v for k, v in metrics.items() if isinstance(v, (float, int))}, ap

    def save_model(self):
        model = de_parallel(self.model)
        architecture = json.loads((self.save_dir / 'architecture.json').read_text())
        architecture['feature_shapes'] = model.last_shapes
        architecture['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
        dump_json(self.save_dir / 'architecture.json', architecture)
        checkpoint = dict(arm=self.arm, fingerprint=self.expected_fingerprint, epoch=self.epoch,
             deployment_variant=self.deployment_variant,
             best_epoch=self.saved_best_epoch, best_fitness=self.best_fitness,
             model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
             ema_state={k: v.detach().cpu() for k, v in self.ema.ema.state_dict().items()},
             ema_updates=self.ema.updates, optimizer=self.optimizer.state_dict(),
             scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
             rank_states=self.all_rng, optimizer_steps=self.optimizer_steps,
             optimizer_attempts=self.optimizer_attempts,
             train_args=vars(self.args), metrics=self.current_eval)
        checkpoint['rank_states'][0]['rng'] = rng_state()
        temp = self.last.with_suffix('.tmp')
        torch.save(checkpoint, temp)
        temp.replace(self.last)
        if self.new_best:
            shutil.copyfile(self.last, self.best)
        # Check serialized EMA and optimizer state, without retaining a live computation graph.
        if self.smoke or self.epoch == 0:
            loaded = torch.load(self.last, map_location='cpu', weights_only=False)
            for k, v in checkpoint['ema_state'].items():
                assert torch.equal(v, loaded['ema_state'][k]), k
            assert loaded['optimizer_steps'] == self.optimizer_steps
            assert len(loaded['rank_states']) == 2
            dump_json(self.save_dir / 'checkpoint_roundtrip.json', dict(epoch=self.epoch + 1, exact_ema=True,
               optimizer=True, scaler=True, scheduler=True, rng_ranks=2, fingerprint=self.expected_fingerprint))

    def epoch_complete(self, trainer):
        if RANK in (-1, 0):
            dump_json(self.save_dir / 'status.json', dict(arm=self.arm, epoch=self.epoch + 1,
                  epochs=self.epochs, best_epoch=self.saved_best_epoch, val=self.current_eval,
                  lr=[g['lr'] for g in self.optimizer.param_groups], loss=self.tloss.cpu().tolist(),
                  optimizer_steps=self.optimizer_steps, epoch_seconds=time.time() - self.epoch_start,
                  total_seconds=time.time() - self.start_wall, state='training', updated=time.time()))
        if self.smoke and self.spec.get('pause_smoke_after_epoch') == self.epoch + 1:
            self.stop = True
        print('EPOCH_FINISHED', RANK, self.arm, self.epoch + 1, 'optimizer_steps', self.optimizer_steps, flush=True)

    def _close_dataloader_mosaic(self):
        self.train_loader.dataset.close_mosaic(copy.copy(self.args))

    def final_eval(self):
        if self.smoke:
            dump_json(self.save_dir / 'smoke_complete.json', dict(epochs=self.epochs, completed_epoch=self.epoch + 1,
                      best_epoch=self.saved_best_epoch,
                      val=self.current_eval, checkpoint_roundtrip=True))
            return
        checkpoint = torch.load(self.best, map_location='cpu', weights_only=False)
        train_seconds = time.time() - self.train_time_start
        self.ema.ema.load_state_dict(checkpoint['ema_state'], strict=True)
        image_root = Path(json.loads((self.data_root / 'manifest.json').read_text())['image_root'])
        docs = json.loads((self.data_root / 'aitodv2_val.json').read_text())
        selected = set(random.Random(0).sample([im['id'] for im in docs['images']], 12))
        val_metrics = evaluate_model(self.ema.ema, self.data_root / 'aitodv2_val.json', image_root / 'val',
                                     self.save_dir / 'metrics/val_best_final', diagnostics_ids=selected)
        if self.spec.get('validation_only', False):
            dump_json(self.save_dir / 'completed.json', dict(best_epoch=checkpoint['best_epoch'],
                      best_val_AP=checkpoint['best_fitness'], val=val_metrics, test=None,
                      train_seconds=train_seconds,
                      final_evaluation_seconds=time.time() - self.train_time_start - train_seconds))
            return
        metrics = evaluate_model(self.ema.ema, self.data_root / 'aitodv2_test.json', image_root / 'test',
                                 self.save_dir / 'metrics/test_best')
        dump_json(self.save_dir / 'completed.json', dict(best_epoch=checkpoint['best_epoch'],
                  best_val_AP=checkpoint['best_fitness'], test=metrics,
                  train_seconds=train_seconds,
                  final_evaluation_seconds=time.time() - self.train_time_start - train_seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    assert int(os.environ.get('WORLD_SIZE', '1')) == 2, 'Each arm requires exactly two DDP ranks'
    manifest = json.loads((Path(spec['data_root']) / 'manifest.json').read_text())
    assert args.smoke or manifest['formal_allowed']
    if not args.smoke:
        assert (ROOT / 'artifacts/pair_smoke_verified.json').exists(), 'Smoke gate has not passed'
    trainer = PairTrainer(spec, args.smoke, args.resume)
    trainer.train()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

"""Prepare a seed-0, twelve-job suite without launching training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    target.symlink_to(Path(source).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--aitod-root', type=Path, required=True)
    parser.add_argument('--visdrone-root', type=Path, required=True)
    parser.add_argument('--yolo11-aitod-init', type=Path, required=True)
    parser.add_argument('--yolo11-visdrone-init', type=Path, required=True)
    parser.add_argument('--yolo11-weights', type=Path, required=True)
    parser.add_argument('--yolo8-weights', type=Path, required=True)
    parser.add_argument('--rtdetr-weights', type=Path, required=True)
    parser.add_argument('--skip-rtdetr-init', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / 'queue_started.json').exists():
        raise RuntimeError('Do not rewrite a suite after its queue has started')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root / 'runtime/yolo'))
    sys.path.insert(0, str(REPOSITORY))
    sys.path.insert(0, str(REPOSITORY / 'third_party/yolo11_ultralytics'))
    import torch
    torch.set_num_threads(2)
    from yolov8.training import create_initialization
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), source=str(REPOSITORY), seed=0)
    write_json(root / 'artifacts/environment.json', environment)
    jobs = []
    manifests = {}
    for dataset, source in [('aitodv2', args.aitod_root), ('visdrone', args.visdrone_root)]:
        manifest = json.loads((source / 'manifest.json').read_text())
        manifests[dataset] = manifest
        data_root = root / 'data' / dataset
        data_root.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                link(path, data_root / path.name)
        if dataset == 'aitodv2':
            task_yaml = yaml.safe_load((source / 'aitodv2.yaml').read_text())
            task_yaml['path'] = str(source.resolve())
        else:
            task_yaml = yaml.safe_load((source / 'visdrone.yaml').read_text())
        prefix = 'aitodv2' if dataset == 'aitodv2' else 'visdrone'
        # The copied training YAML is generated separately so original data are never edited.
        target_yaml = data_root / (prefix + '.yaml')
        target_yaml.unlink(missing_ok=True)
        target_yaml.write_text(yaml.safe_dump(task_yaml, sort_keys=False))
        nc, size = (8, 800) if dataset == 'aitodv2' else (10, 640)
        for family in ('yolo11m', 'yolov8m', 'rtdetr'):
            paired = root / 'weights' / f'{family}_{dataset}_paired.pt'
            if family == 'yolo11m':
                original = args.yolo11_aitod_init if dataset == 'aitodv2' else args.yolo11_visdrone_init
                link(original, paired)
                pretrained = args.yolo11_weights
            elif family == 'yolov8m':
                pretrained = args.yolo8_weights
                if not paired.exists():
                    create_initialization(pretrained, nc, paired)
            else:
                pretrained = args.rtdetr_weights
            pair_id = f'{family}_{dataset}_s0'
            for arm in ('baseline_native', 'full_asymmetric'):
                identifier = f'{pair_id}_{arm}'
                if family != 'rtdetr':
                    spec = dict(dataset=dataset, model_family=family, arm=arm, seed=0, epochs=100,
                                deployment_variant='full', imgsz=size, nc=nc, workers=2,
                                model='yolov8m.yaml' if family == 'yolov8m' else 'yolo11m.yaml',
                                data_root=str(data_root), pair_id=pair_id, port=29941 if arm == 'baseline_native' else 29942,
                                output_group='runs', run_name=identifier, validation_only=True,
                                experiment_root=str(root), paired_init=str(paired),
                                pretrained_weights=str(pretrained.resolve()),
                                preflight_report=str(root / 'preflight' / f'{pair_id}.json'))
                    config_path = root / 'configs' / (identifier + '.yaml')
                else:
                    config_path = root / 'configs' / (identifier + '.yml')
                    config = REPOSITORY / 'rtdetr/dota_hbb/configs/rtdetr/dota1_hbb_r50vd_common_e100.yml'
                    spec = dict(__include__=[str(config)], num_classes=nc, remap_mscoco_category=False,
                                output_dir=str(root / 'runs' / identifier), seed=0,
                                PResNet=dict(pretrained=False, expose_c2=arm == 'full_asymmetric'),
                                HybridEncoder=dict(eval_spatial_size=[size, size], use_brsa=False,
                                                   use_asymmetric_neck=arm == 'full_asymmetric'),
                                RTDETRTransformer=dict(eval_spatial_size=[size, size]),
                                train_dataloader=dict(dataset=dict(img_folder='', ann_file=''), num_workers=2),
                                val_dataloader=dict(dataset=dict(img_folder='', ann_file=''), num_workers=2))
                    for split, loader in [('train', 'train_dataloader'), ('val', 'val_dataloader')]:
                        annotation = json.loads((source / f'{prefix}_{split}.json').read_text())
                        mapping = {category['id']: index for index, category in enumerate(annotation['categories'])}
                        for category in annotation['categories']:
                            category['id'] = mapping[category['id']]
                        for target in annotation['annotations']:
                            target['category_id'] = mapping[target['category_id']]
                        protocol = dict(dataset=dataset)
                        if dataset == 'visdrone':
                            protocol['raw_annotation_dir'] = manifest['splits'][split]['annotation_dir']
                        annotation.setdefault('info', {})['native_evaluation'] = protocol
                        ann_path = root / 'data' / 'rtdetr' / f'{dataset}_{split}.json'
                        if not ann_path.exists():
                            write_json(ann_path, annotation)
                        image_dir = (str(Path(manifest['image_root']) / split) if dataset == 'aitodv2'
                                     else manifest['splits'][split]['image_dir'])
                        spec[loader]['dataset'].update(img_folder=image_dir, ann_file=str(ann_path))
                        spec[loader]['dataset']['transforms'] = dict(ops=[
                            dict(type='RandomPhotometricDistort', p=.5), dict(type='RandomZoomOut', fill=0),
                            dict(type='RandomIoUCrop', p=.8), dict(type='SanitizeBoundingBox', min_size=1),
                            dict(type='RandomHorizontalFlip'), dict(type='Resize', size=[size, size]),
                            dict(type='ToImageTensor'), dict(type='ConvertDtype'),
                            dict(type='SanitizeBoundingBox', min_size=1),
                            dict(type='ConvertBox', out_fmt='cxcywh', normalize=True)] if split == 'train' else [
                            dict(type='Resize', size=[size, size]), dict(type='ToImageTensor'), dict(type='ConvertDtype')])
                    spec['SetCriterion'] = dict(weight_dict=dict(loss_vfl=1, loss_bbox=5, loss_giou=2,
                        **({'loss_brsa_fg': .1} if arm == 'full_asymmetric' else {})))
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(spec, sort_keys=False))
                jobs.append(dict(id=identifier, family=family, dataset=dataset, arm=arm,
                                 config=str(config_path), paired_init=str(paired),
                                 output=str(root / 'runs' / identifier), state='pending', seed=0, epochs=100))
    if not args.skip_rtdetr_init:
        sys.path.insert(0, str(REPOSITORY / 'rtdetr/dota_hbb'))
        from src.core import YAMLConfig
        source_checkpoint = torch.load(args.rtdetr_weights, map_location='cpu', weights_only=False)
        source_state = source_checkpoint['ema']['module'] if 'ema' in source_checkpoint else source_checkpoint['model']
        for dataset in ('aitodv2', 'visdrone'):
            torch.manual_seed(0)
            job = next(job for job in jobs if job['family'] == 'rtdetr' and job['dataset'] == dataset and job['arm'] == 'baseline_native')
            config = YAMLConfig(job['config'])
            model = config.model
            state = model.state_dict()
            loaded = {key: value for key, value in source_state.items() if key in state and state[key].shape == value.shape}
            missing = sorted(set(state) - set(loaded))
            if any(not key.startswith(('decoder.denoising_class_embed', 'decoder.enc_score_head', 'decoder.dec_score_head')) for key in missing):
                raise RuntimeError(f'Unexpected RT-DETR pretrained transfer mismatch: {missing}')
            model.load_state_dict(loaded, strict=False)
            torch.save(dict(model=model.state_dict(), seed=0, coco_sha256=digest(args.rtdetr_weights),
                            loaded=len(loaded), reinitialized=missing), job['paired_init'])
    write_json(root / 'suite.json', dict(root=str(root), repository=str(REPOSITORY),
        seed=0, repetitions=1, epochs=100, jobs=jobs, environment=environment,
        note='Full paper method only; existing two-GPU training recipes retained. No paper three-run averaging.'))
    print(root / 'suite.json')


if __name__ == '__main__':
    main()

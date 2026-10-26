import argparse
import collections
import json
import os
import random
from pathlib import Path

import bootstrap
import cv2
import torch
import yaml
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.torch_utils import init_seeds

from bootstrap import ROOT
from common import dump_json, sha256
from models.asymmetric_yolo11 import load_paired, native_model


DATASET_ROOT = Path(os.environ.get('VISDRONE_ROOT', str(ROOT / 'data/VisDrone2019-DET')))
RAW_ROOT = DATASET_ROOT / 'raw'
OUT = ROOT / 'data/visdrone'
CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
           'awning-tricycle', 'bus', 'motor']
SPLITS = {
    'train': ('VisDrone2019-DET-train', 6471),
    'val': ('VisDrone2019-DET-val', 548),
    'test': ('VisDrone2019-DET-test-dev', 1610),
}


def prepare_initialization():
    init_seeds(0, deterministic=False)
    native = native_model(nc=10)
    coco, _ = attempt_load_one_weight(ROOT / 'weights/yolo11m.pt', device='cpu', fuse=False)
    source, target = coco.float().state_dict(), native.state_dict()
    loaded = {key: value for key, value in source.items()
              if key in target and target[key].shape == value.shape}
    missing = sorted(set(target) - set(loaded))
    assert missing and all(key.startswith('model.23.cv3.') for key in missing), missing
    target.update(loaded)
    native.load_state_dict(target, strict=True)
    names = {index: name for index, name in enumerate(CLASSES)}
    native.names = names
    paired = ROOT / 'weights/paired_native_init.pt'
    torch.save(dict(yaml=native.yaml, state_dict=native.state_dict(), seed=0, nc=10, names=names,
                    coco_sha256=sha256(ROOT / 'weights/yolo11m.pt')), paired)
    dump_json(ROOT / 'artifacts/pretrained_loading.json', dict(
        loaded=sorted(loaded), reinitialized_classification=missing,
        coco_sha256=sha256(ROOT / 'weights/yolo11m.pt'), paired_sha256=sha256(paired)))
    for arm in ('baseline_native', 'full_asymmetric'):
        init_seeds(0, deterministic=False)
        _, report = load_paired(arm, paired)
        dump_json(ROOT / ('artifacts/%s_loading.json' % arm), report)


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            values = [int(float(value)) for value in line.split(',')[:8]]
            assert len(values) == 8
            rows.append(values)
    return rows


def convert_split(split, folder, expected):
    source = RAW_ROOT / folder
    image_dir, annotation_dir = source / 'images', source / 'annotations'
    images = sorted(image_dir.glob('*.jpg'))
    annotations = sorted(annotation_dir.glob('*.txt'))
    assert len(images) == len(annotations) == expected, (split, len(images), len(annotations))
    assert {path.stem for path in images} == {path.stem for path in annotations}
    categories = [dict(id=index + 1, name=name, supercategory='object')
                  for index, name in enumerate(CLASSES)]
    coco_images, coco_annotations = [], []
    skipped = collections.Counter()
    annotation_id = 1
    dimensions = collections.Counter()
    for image_id, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        dimensions['%dx%d' % (width, height)] += 1
        coco_images.append(dict(id=image_id, file_name=path.name, width=width, height=height))
        for x, y, w, h, score, category, truncation, occlusion in read_rows(annotation_dir / (path.stem + '.txt')):
            if score == 0 or category == 0:
                skipped['ignored_region'] += 1
                continue
            if category == 11:
                skipped['others'] += 1
                continue
            if category not in range(1, 11) or w <= 0 or h <= 0:
                skipped['invalid'] += 1
                continue
            coco_annotations.append(dict(id=annotation_id, image_id=image_id, category_id=category,
                bbox=[x, y, w, h], area=w * h, iscrowd=0, truncation=truncation, occlusion=occlusion))
            annotation_id += 1
    document = dict(images=coco_images, annotations=coco_annotations, categories=categories)
    output = OUT / ('visdrone_%s.json' % split)
    dump_json(output, document)
    paths = [str(image_dir / image['file_name']) for image in coco_images]
    (OUT / ('%s.txt' % split)).write_text('\n'.join(paths) + '\n')
    return dict(images=len(coco_images), annotations=len(coco_annotations),
                annotation_sha256=sha256(output), image_dir=str(image_dir),
                annotation_dir=str(annotation_dir), skipped=dict(skipped),
                dimensions=dict(dimensions))


def prepare_data():
    OUT.mkdir(parents=True, exist_ok=True)
    split_reports = {split: convert_split(split, folder, expected)
                     for split, (folder, expected) in SPLITS.items()}
    archives = {}
    for path in sorted((DATASET_ROOT / 'archives').glob('*.zip')):
        archives[path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    manifest = dict(origin='Ultralytics GitHub release of official VisDrone2019-DET archives',
                    formal_allowed=True, raw_root=str(RAW_ROOT), imgsz=640,
                    categories=[dict(id=index + 1, name=name) for index, name in enumerate(CLASSES)],
                    archives=archives, splits=split_reports,
                    training_policy='score=0/category=0 ignored regions and category=11 others excluded',
                    evaluation_policy='official VisDrone ignored-region protocol, maxDets=500')
    dump_json(OUT / 'manifest.json', manifest)
    names = {index: name for index, name in enumerate(CLASSES)}
    config = dict(path=str(OUT), train=str(OUT / 'train.txt'), val=str(OUT / 'val.txt'),
                  test=str(OUT / 'test.txt'), names=names, nc=10)
    (OUT / 'visdrone.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    train_paths = (OUT / 'train.txt').read_text().splitlines()
    selected = random.Random(0).sample(train_paths, 128)
    (OUT / 'smoke_train.txt').write_text('\n'.join(selected) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()
    if not args.data and not args.init:
        args.data = args.init = True
    if args.data:
        prepare_data()
    if args.init:
        prepare_initialization()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box


CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'swimming-pool',
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
SUBSIZE = 1024
GAP = 200
STRIDE = SUBSIZE - GAP
VISIBLE_THRESHOLD = 0.7


def starts(length):
    last = max(length - SUBSIZE, 0)
    result = list(range(0, last + 1, STRIDE))
    if not result or result[-1] != last:
        result.append(last)
    return result


def read_objects(path):
    objects = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        fields = line.split()
        if len(fields) < 10 or fields[8] not in CLASS_TO_ID:
            continue
        points = np.asarray([float(value) for value in fields[:8]], dtype=np.float64).reshape(4, 2)
        polygon = Polygon(points)
        repaired = False
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            repaired = True
        if polygon.is_empty or polygon.area <= 0:
            continue
        objects.append({
            'polygon': polygon,
            'category_id': CLASS_TO_ID[fields[8]],
            'category': fields[8],
            'difficult': int(fields[9]),
            'repaired': repaired,
        })
    return objects


def clip_object(obj, left, top):
    window = box(left, top, left + SUBSIZE, top + SUBSIZE)
    clipped = obj['polygon'].intersection(window)
    if clipped.is_empty or clipped.area <= 0:
        return None
    if clipped.geom_type == 'MultiPolygon':
        clipped = max(clipped.geoms, key=lambda item: item.area)
    if clipped.geom_type != 'Polygon' or clipped.area <= 0:
        return None
    visible = min(float(clipped.area / obj['polygon'].area), 1.0)
    x1, y1, x2, y2 = clipped.bounds
    x1, x2 = x1 - left, x2 - left
    y1, y2 = y1 - top, y2 - top
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    return {
        'bbox': [x1, y1, width, height],
        'area': width * height,
        'category_id': obj['category_id'],
        'iscrowd': int(obj['difficult'] != 0 or visible < VISIBLE_THRESHOLD),
        'visible_ratio': visible,
        'original_difficult': obj['difficult'],
    }


def process_image(job):
    image_path, label_path, output_dir = map(Path, job)
    objects = read_objects(label_path)
    with Image.open(image_path) as source:
        source = source.convert('RGB')
        width, height = source.size
        patches = []
        for top in starts(height):
            for left in starts(width):
                valid_width = min(SUBSIZE, width - left)
                valid_height = min(SUBSIZE, height - top)
                crop = source.crop((left, top, left + valid_width, top + valid_height))
                canvas = Image.new('RGB', (SUBSIZE, SUBSIZE))
                canvas.paste(crop, (0, 0))
                name = f'{image_path.stem}__1__{left}___{top}.png'
                canvas.save(output_dir / name, format='PNG', compress_level=3)
                annotations = []
                for obj in objects:
                    clipped = clip_object(obj, left, top)
                    if clipped is not None:
                        annotations.append(clipped)
                patches.append({
                    'file_name': name,
                    'width': SUBSIZE,
                    'height': SUBSIZE,
                    'source_id': image_path.stem,
                    'source_file': image_path.name,
                    'source_width': width,
                    'source_height': height,
                    'x_offset': left,
                    'y_offset': top,
                    'valid_width': valid_width,
                    'valid_height': valid_height,
                    'annotations': annotations,
                })
    scene_objects = []
    repaired = 0
    for obj in objects:
        x1, y1, x2, y2 = obj['polygon'].bounds
        scene_objects.append({
            'category_id': obj['category_id'],
            'category': obj['category'],
            'bbox_xyxy': [x1, y1, x2, y2],
            'difficult': obj['difficult'],
        })
        repaired += int(obj['repaired'])
    scene = {
        'id': image_path.stem,
        'file_name': image_path.name,
        'width': width,
        'height': height,
        'objects': scene_objects,
    }
    return patches, scene, repaired


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(',', ':'))
    temporary.replace(path)


def prepare_split(raw_root, output_root, split, workers):
    image_root = raw_root / 'images' / split
    label_root = raw_root / 'labels' / f'{split}_original'
    output_images = output_root / 'images' / split
    output_images.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_root.glob('*.jpg'))
    jobs = []
    for image_path in image_paths:
        label_path = label_root / f'{image_path.stem}.txt'
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        jobs.append((str(image_path), str(label_path), str(output_images)))

    images, annotations, scenes = [], [], []
    annotation_id = 1
    positive_count = ignore_count = repaired_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for source_index, (patches, scene, repaired) in enumerate(executor.map(process_image, jobs), 1):
            scenes.append(scene)
            repaired_count += repaired
            for patch in patches:
                patch_annotations = patch.pop('annotations')
                image_id = len(images) + 1
                patch['id'] = image_id
                images.append(patch)
                for annotation in patch_annotations:
                    annotation['id'] = annotation_id
                    annotation['image_id'] = image_id
                    annotation_id += 1
                    annotations.append(annotation)
                    if annotation['iscrowd']:
                        ignore_count += 1
                    else:
                        positive_count += 1
            if source_index % 100 == 0 or source_index == len(jobs):
                print(f'{split}: {source_index}/{len(jobs)} scenes, {len(images)} patches', flush=True)

    categories = [{'id': index, 'name': name} for index, name in enumerate(CLASSES)]
    coco = {
        'info': {
            'description': 'DOTA-v1.0 Task2 HBB, split1024 gap200 rate1',
            'subsize': SUBSIZE,
            'gap': GAP,
            'stride': STRIDE,
            'visible_threshold': VISIBLE_THRESHOLD,
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    annotations_root = output_root / 'annotations'
    annotations_root.mkdir(parents=True, exist_ok=True)
    annotation_path = annotations_root / f'instances_{split}.json'
    scene_path = annotations_root / f'scenes_{split}.json'
    write_json(annotation_path, coco)
    write_json(scene_path, {'categories': categories, 'scenes': scenes})
    digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    return {
        'source_scenes': len(jobs),
        'patches': len(images),
        'positive_annotations': positive_count,
        'ignore_annotations': ignore_count,
        'repaired_source_polygons': repaired_count,
        'annotation_file': str(annotation_path),
        'annotation_sha256': digest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f'Output is not empty: {args.output_root}')
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        'protocol_id': 'dota10_hbb_jpeg_split1024_gap200_input640_rtdetr_e100_s0_v1',
        'source_root': str(args.raw_root.resolve()),
        'classes': list(CLASSES),
        'subsize': SUBSIZE,
        'gap': GAP,
        'stride': STRIDE,
        'visible_threshold': VISIBLE_THRESHOLD,
        'splits': {},
    }
    for split in ('train', 'val'):
        summary['splits'][split] = prepare_split(
            args.raw_root.resolve(), args.output_root.resolve(), split, args.workers
        )
    write_json(args.output_root / 'data_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

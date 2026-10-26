import argparse
import bootstrap
from bootstrap import ROOT
import collections
import json
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.torch_utils import init_seeds
from models.asymmetric_yolo11 import native_model, load_paired
from common import dump_json, sha256

IMAGE_ROOT = Path(os.environ.get('AITODV2_IMAGE_ROOT', str(ROOT / 'data/images')))
MIRROR_ROOT = Path(os.environ.get('AITODV2_ANNOTATION_ROOT', str(ROOT / 'data/annotations')))


def prepare_initialization():
    init_seeds(0, deterministic=False)
    native = native_model()
    coco, _ = attempt_load_one_weight(ROOT / 'weights/yolo11m.pt', device='cpu', fuse=False)
    source, target = coco.float().state_dict(), native.state_dict()
    loaded = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    missing = sorted(set(target) - set(loaded))
    assert all(k.startswith('model.23.cv3.') for k in missing), missing
    target.update(loaded)
    native.load_state_dict(target, strict=True)
    paired = ROOT / 'weights/paired_native_init.pt'
    torch.save(dict(yaml=native.yaml, state_dict=native.state_dict(), seed=0,
                    coco_sha256=sha256(ROOT / 'weights/yolo11m.pt')), paired)
    dump_json(ROOT / 'artifacts/pretrained_loading.json', dict(loaded=sorted(loaded),
              reinitialized_classification=missing, coco_sha256=sha256(ROOT / 'weights/yolo11m.pt'),
              paired_sha256=sha256(paired)))
    for arm in ('baseline_native', 'full_asymmetric'):
        init_seeds(0, deterministic=False)
        model, report = load_paired(arm, paired)
        dump_json(ROOT / ('artifacts/%s_loading.json' % arm), report)
    print('paired initialization saved', flush=True)


def prepare_data(annotation_root, diagnostic=False, local=False):
    annotation_root = Path(annotation_root)
    out = ROOT / ('data/local' if local else 'data/diagnostic' if diagnostic else 'data/official')
    out.mkdir(parents=True, exist_ok=True)
    documents = {}
    for split in ('train', 'val', 'test'):
        path = annotation_root / (('instances_%s.json' if diagnostic or local else 'aitodv2_%s.json') % split)
        if split == 'val' and not path.exists() and not diagnostic:
            trainval = json.loads((annotation_root / 'aitodv2_trainval.json').read_text())
            train_names = {x['file_name'] for x in documents['train']['images']}
            images = [x for x in trainval['images'] if x['file_name'] not in train_names]
            ids = {x['id'] for x in images}
            doc = dict(trainval, images=images,
                       annotations=[a for a in trainval['annotations'] if a['image_id'] in ids])
            dump_json(out / ('aitodv2_%s.json' % split), doc)
        else:
            doc = json.loads(path.read_text())
            shutil.copyfile(path, out / ('aitodv2_%s.json' % split))
        documents[split] = doc
    cats = documents['train']['categories']
    assert len(cats) == 8
    mapping = {c['id']: i for i, c in enumerate(cats)}
    manifest = dict(origin='local_aitodv2_user_authorized' if local else 'third_party_converted_diagnostic' if diagnostic else 'official_json',
                    formal_allowed=local or not diagnostic, image_root=str(IMAGE_ROOT), categories=cats,
                    category_id_to_train_index=mapping, splits={})
    if local:
        manifest['conversion_note'] = ('Existing local HF export remaps image/annotation IDs and drops '
             'nonpositive width/height boxes (40 train, 9 val, 2 test). Original raw JSON unavailable; '
             'user explicitly requested use of local AI-TODv2 on 2026-09-30. No further filtering in this task.')
    for split, doc in documents.items():
        assert doc['categories'] == cats
        expected = dict(train=11214, val=2804, test=14018)[split]
        assert len(doc['images']) == expected
        ids = [im['id'] for im in doc['images']]
        assert len(set(ids)) == len(ids)
        ann_counts = collections.Counter(a['image_id'] for a in doc['annotations'])
        for image in doc['images']:
            assert (IMAGE_ROOT / split / Path(image['file_name']).name).is_file(), image
        invalid = sum(a['bbox'][2] <= 0 or a['bbox'][3] <= 0 for a in doc['annotations'])
        ignore = sum(bool(a.get('ignore', False) or a.get('iscrowd', False)) for a in doc['annotations'])
        # Ignore geometry requires a synchronized annotation mask transform; fail explicitly if encountered.
        assert ignore == 0, 'ignore/crowd annotations require the dedicated aux mask implementation'
        manifest['splits'][split] = dict(images=len(ids), annotations=len(doc['annotations']),
             annotation_sha256=sha256(out / ('aitodv2_%s.json' % split)), invalid_nonpositive_boxes=invalid,
             ignore_crowd=ignore, empty_images=sum(ann_counts[i] == 0 for i in ids),
             dimensions=dict(collections.Counter('%dx%d' % (im['width'], im['height']) for im in doc['images'])))
        dump_json(out / ('%s_image_ids.json' % split), ids)
        paths = [str(IMAGE_ROOT / split / Path(im['file_name']).name) for im in doc['images']]
        (out / ('%s.txt' % split)).write_text('\n'.join(paths) + '\n')
    train_names = {im['file_name'] for im in documents['train']['images']}
    assert not train_names.intersection(im['file_name'] for im in documents['val']['images'])
    dump_json(out / 'manifest.json', manifest)
    names = {i: cat['name'] for i, cat in enumerate(cats)}
    data_cfg = dict(path=str(out), train=str(out / 'train.txt'), val=str(out / 'val.txt'),
                    test=str(out / 'test.txt'), names=names, nc=8)
    (out / 'aitodv2.yaml').write_text(yaml.safe_dump(data_cfg))
    train_ids = sorted(ids for ids in train_names)
    selected = random.Random(0).sample(train_ids, 128)
    smoke_paths = [str(IMAGE_ROOT / 'train' / Path(name).name) for name in selected]
    (out / 'smoke_train.txt').write_text('\n'.join(smoke_paths) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-root')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()
    if args.init:
        prepare_initialization()
    if args.annotation_root or args.diagnostic or args.local:
        prepare_data(args.annotation_root or MIRROR_ROOT, args.diagnostic, args.local)

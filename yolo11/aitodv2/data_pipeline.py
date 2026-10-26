import bootstrap
import collections
import json
import multiprocessing
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Albumentations, Compose, Mosaic


class CocoGeometryDataset(YOLODataset):
    def __init__(self, annotation_file, *args, **kwargs):
        self.annotation_file = Path(annotation_file)
        self.shared_epoch = multiprocessing.Value('i', 0)
        super().__init__(*args, **kwargs)

    def get_labels(self):
        doc = json.loads(self.annotation_file.read_text())
        index = {Path(im['file_name']).name: im for im in doc['images']}
        mapping = {cat['id']: i for i, cat in enumerate(doc['categories'])}
        annotations = collections.defaultdict(list)
        for ann in doc['annotations']:
            if ann.get('ignore', False) or ann.get('iscrowd', False):
                raise RuntimeError('Ignore annotations need explicit training policy')
            if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                annotations[ann['image_id']].append(ann)
        labels = []
        for path in self.im_files:
            im = index[Path(path).name]
            boxes, classes = [], []
            for ann in annotations[im['id']]:
                x, y, w, h = ann['bbox']
                boxes.append([(x + w / 2) / im['width'], (y + h / 2) / im['height'],
                              w / im['width'], h / im['height']])
                classes.append(mapping[ann['category_id']])
            labels.append(dict(im_file=path, shape=(im['height'], im['width']),
                  cls=np.asarray(classes, dtype=np.float32).reshape(-1, 1),
                  bboxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4), segments=[],
                  keypoints=None, normalized=True, bbox_format='xywh', image_id=im['id']))
        return labels

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        label['pixel_valid_mask'] = np.ones(label['img'].shape[:2], dtype=np.float32)
        return label

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        def remove_extra(compose):
            if isinstance(compose, Compose):
                compose.transforms = [x for x in compose.transforms if not isinstance(x, Albumentations)]
                for transform in compose.transforms:
                    remove_extra(transform)
                    if isinstance(transform, Mosaic):
                        transform.get_indexes = lambda: [random.randint(0, len(self) - 1) for _ in range(3)]
                    if getattr(transform, 'pre_transform', None):
                        remove_extra(transform.pre_transform)
        remove_extra(transforms)
        return transforms

    def __getitem__(self, index):
        py_state, np_state = random.getstate(), np.random.get_state()
        seed = (self.shared_epoch.value * 1000003 + index * 97) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        try:
            return super().__getitem__(index)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    @staticmethod
    def collate_fn(batch):
        masks = torch.stack([x.pop('pixel_valid_mask') for x in batch])
        ids = [x.pop('image_id') for x in batch]
        result = YOLODataset.collate_fn(batch)
        result['pixel_valid_mask'] = masks
        result['image_id'] = ids
        return result


def loader(dataset, batch_size, workers, rank, world_size=2):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=0, drop_last=False) if rank >= 0 else None
    generator = torch.Generator().manual_seed(12345 + max(rank, 0))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=workers, pin_memory=True, collate_fn=dataset.collate_fn,
                      generator=generator, persistent_workers=False), sampler


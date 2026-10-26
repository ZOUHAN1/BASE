import json
from pathlib import Path

import numpy as np
import torch.distributed as torch_dist

from src.misc import dist


def inclusive_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    area_a = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
    area_b = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def nms_inclusive(detections, threshold=0.5):
    if not len(detections):
        return detections
    order = detections[:, 0].argsort()[::-1]
    keep = []
    while order.size:
        index = order[0]
        keep.append(index)
        if order.size == 1:
            break
        overlaps = inclusive_iou(detections[index, 1:5], detections[order[1:], 1:5])
        order = order[1:][overlaps <= threshold]
    return detections[np.asarray(keep, dtype=np.int64)]


def voc07_ap(recall, precision):
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        selected = precision[recall >= threshold]
        ap += (selected.max() if selected.size else 0.0) / 11.0
    return float(ap)


class DotaHBBEvaluator:
    def __init__(self, dataset, score_threshold=0.001, nms_threshold=0.5):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.patch_metadata = dataset.coco.imgs
        scene_path = Path(dataset.ann_file).with_name('scenes_val.json')
        payload = json.loads(scene_path.read_text())
        self.categories = payload['categories']
        self.scenes = {scene['id']: scene for scene in payload['scenes']}
        self.predictions = {}

    @classmethod
    def maybe_create(cls, dataset):
        if not getattr(dataset, 'ann_file', None) or not dataset.coco.imgs:
            return None
        first = next(iter(dataset.coco.imgs.values()))
        scene_path = Path(dataset.ann_file).with_name('scenes_val.json')
        if 'source_id' not in first or not scene_path.is_file():
            return None
        return cls(dataset)

    def update(self, predictions):
        for image_id, output in predictions.items():
            keep = output['scores'] >= self.score_threshold
            boxes = output['boxes'][keep].detach().cpu().numpy().astype(np.float32)
            scores = output['scores'][keep].detach().cpu().numpy().astype(np.float32)
            labels = output['labels'][keep].detach().cpu().numpy().astype(np.int16)
            self.predictions[int(image_id)] = (boxes, scores, labels)

    def synchronize_and_evaluate(self):
        gathered = dist.all_gather(self.predictions)
        metrics = None
        if dist.is_main_process():
            merged = {}
            for rank_predictions in gathered:
                merged.update(rank_predictions)
            metrics = self.evaluate(merged)
        if dist.is_dist_available_and_initialized():
            payload = [metrics]
            torch_dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        return metrics

    def merge_scenes(self, predictions):
        merged = {}
        for image_id, (boxes, scores, labels) in predictions.items():
            metadata = self.patch_metadata[image_id]
            source_id = metadata['source_id']
            valid_width = metadata['valid_width']
            valid_height = metadata['valid_height']
            x_offset = metadata['x_offset']
            y_offset = metadata['y_offset']
            source_width = metadata['source_width']
            source_height = metadata['source_height']
            if boxes.size == 0:
                continue
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            valid = (centers[:, 0] < valid_width) & (centers[:, 1] < valid_height)
            boxes, scores, labels = boxes[valid].copy(), scores[valid], labels[valid]
            boxes[:, 0::2] += x_offset
            boxes[:, 1::2] += y_offset
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, source_width - 1)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, source_height - 1)
            for category_id in np.unique(labels):
                selected = labels == category_id
                rows = np.column_stack((scores[selected], boxes[selected]))
                merged.setdefault((source_id, int(category_id)), []).append(rows)

        output = {}
        for key, chunks in merged.items():
            detections = np.concatenate(chunks, axis=0)
            output[key] = nms_inclusive(detections, self.nms_threshold)
        return output

    def evaluate_class(self, scene_detections, category_id):
        records = {}
        positives = 0
        for scene_id, scene in self.scenes.items():
            objects = [obj for obj in scene['objects'] if obj['category_id'] == category_id]
            boxes = np.asarray([obj['bbox_xyxy'] for obj in objects], dtype=np.float32).reshape(-1, 4)
            difficult = np.asarray([obj['difficult'] != 0 for obj in objects], dtype=bool)
            records[scene_id] = [boxes, difficult, np.zeros(len(objects), dtype=bool)]
            positives += int((~difficult).sum())

        detections = []
        for (scene_id, label), rows in scene_detections.items():
            if label == category_id:
                detections.extend((float(row[0]), scene_id, row[1:5]) for row in rows)
        detections.sort(key=lambda item: item[0], reverse=True)
        tp = np.zeros(len(detections), dtype=np.float64)
        fp = np.zeros(len(detections), dtype=np.float64)
        for index, (_, scene_id, detection) in enumerate(detections):
            boxes, difficult, detected = records[scene_id]
            if not len(boxes):
                fp[index] = 1
                continue
            overlaps = inclusive_iou(detection, boxes)
            best = int(overlaps.argmax())
            if overlaps[best] > 0.5:
                if difficult[best]:
                    continue
                if not detected[best]:
                    tp[index] = 1
                    detected[best] = True
                else:
                    fp[index] = 1
            else:
                fp[index] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / max(positives, 1)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        return voc07_ap(recall, precision)

    def evaluate(self, predictions):
        scene_detections = self.merge_scenes(predictions)
        per_class = [self.evaluate_class(scene_detections, category['id'])
                     for category in self.categories]
        return {
            'map50': float(np.mean(per_class)),
            'per_class': {category['name']: ap for category, ap in zip(self.categories, per_class)},
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'protocol': 'DOTA Task2 HBB VOC07 11-point, inclusive IoU>0.5',
        }

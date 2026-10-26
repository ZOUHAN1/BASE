import contextlib
import io
import json
import sys
import time
from pathlib import Path

import bootstrap
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.utils.ops import clip_boxes, non_max_suppression

from common import dump_json


TOOLKIT = bootstrap.ROOT / 'third_party/visdrone_eval'
sys.path.insert(0, str(TOOLKIT))
from viseval import eval_det  # noqa: E402


CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
               'awning-tricycle', 'bus', 'motor']


def load_annotation(path, columns, dtype=np.float32):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = [float(value) for value in line.split(',')[:columns]]
            assert len(row) == columns
            rows.append(row)
    return np.asarray(rows, dtype=dtype).reshape(-1, columns)


class LetterboxEvaluationDataset(Dataset):
    def __init__(self, annotation_file, image_dir, raw_annotation_dir, imgsz, image_ids=None):
        self.doc = json.loads(Path(annotation_file).read_text())
        self.images = [image for image in self.doc['images']
                       if image_ids is None or image['id'] in image_ids]
        self.image_dir = Path(image_dir)
        self.raw_annotation_dir = Path(raw_annotation_dir)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.image_dir / info['file_name']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        assert (width, height) == (info['width'], info['height'])
        ratio = min(self.imgsz / height, self.imgsz / width)
        new_width, new_height = round(width * ratio), round(height * ratio)
        left = round((self.imgsz - new_width) / 2 - .1)
        top = round((self.imgsz - new_height) / 2 - .1)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        output = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        output[top:top + new_height, left:left + new_width] = resized
        mask = np.zeros((1, self.imgsz, self.imgsz), dtype=np.float32)
        mask[:, top:top + new_height, left:left + new_width] = 1
        rgb = np.ascontiguousarray(output[:, :, ::-1].transpose(2, 0, 1))
        ground_truth = load_annotation(self.raw_annotation_dir / (path.stem + '.txt'), 8,
                                       dtype=np.int32)
        return dict(img=torch.from_numpy(rgb), mask=torch.from_numpy(mask), image_id=info['id'],
                    ratio=ratio, pad=(left, top), original_shape=(height, width),
                    file_name=info['file_name'], ground_truth=ground_truth)


def undo_letterbox(boxes, ratio, pad, original_shape):
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    clip_boxes(boxes, original_shape)
    return boxes


def run_evaluator(ground_truth, detections, heights, widths, output):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        values = eval_det(ground_truth, detections, heights, widths, per_class=True)
    ap, ap50, ap75, ar1, ar10, ar100, ar500, classwise = values
    metrics = dict(AP=ap / 100, AP50=ap50 / 100, AP75=ap75 / 100,
                   **{'AR@1': ar1 / 100, 'AR@10': ar10 / 100,
                      'AR@100': ar100 / 100, 'AR@500': ar500 / 100},
                   per_class_AP={name: float(value / 100)
                                 for name, value in zip(CLASS_NAMES, classwise)})
    (output / 'evaluator_stdout.txt').write_text(stdout.getvalue())
    dump_json(output / 'native.json', dict(
        protocol='official VisDrone DET Python port; IoU=.50:.05:.95; maxDets=500',
        raw=metrics, display={key: value * 100 for key, value in metrics.items()
                              if isinstance(value, (float, int))}))
    return metrics


@torch.inference_mode()
def evaluate_model(model, annotation_file, image_dir, raw_annotation_dir, output,
                   amp=True, diagnostics_ids=None, imgsz=640, nc=10):
    assert imgsz == 640 and nc == 10
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / 'submission'
    result_dir.mkdir(exist_ok=True)
    dataset = LetterboxEvaluationDataset(annotation_file, image_dir, raw_annotation_dir, imgsz)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=lambda batch: batch[0])
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    expected_fingerprint = model.fingerprint()
    print('EVAL_IDENTITY', model.arm, expected_fingerprint, 'images=', len(dataset), flush=True)
    all_ground_truth, all_detections, heights, widths = [], [], [], []
    records, covered, empty = [], [], 0
    start = time.time()
    for index, batch in enumerate(loader):
        image = batch['img'][None].to(device, non_blocking=True).float() / 255
        mask = batch['mask'][None].to(device, non_blocking=True)
        diagnostic = diagnostics_ids is not None and batch['image_id'] in diagnostics_ids
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == 'cuda'):
            predictions = model(image, pixel_valid=mask, diagnostics=diagnostic)
        raw = predictions[0] if isinstance(predictions, tuple) else predictions
        detections = non_max_suppression(raw.float(), conf_thres=.001, iou_thres=.7,
            multi_label=True, agnostic=False, max_det=500, max_nms=100000,
            max_time_img=1000000, nc=nc, in_place=False)[0]
        rows = []
        if detections.numel():
            boxes = undo_letterbox(detections[:, :4], batch['ratio'], batch['pad'],
                                   batch['original_shape'])
            for box, score, category in zip(boxes.cpu().tolist(), detections[:, 4].cpu().tolist(),
                                            detections[:, 5].long().cpu().tolist()):
                x1, y1, x2, y2 = box
                row = [x1, y1, x2 - x1, y2 - y1, score, category + 1]
                rows.append(row)
                records.append(dict(image_id=batch['image_id'], file_name=batch['file_name'],
                                    bbox=row[:4], score=score, category_id=category + 1))
        else:
            empty += 1
        array = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        if len(array):
            array = array[np.argsort(-array[:, 4])]
        np.savetxt(result_dir / (Path(batch['file_name']).stem + '.txt'), array,
                   fmt='%.6f', delimiter=',')
        all_ground_truth.append(batch['ground_truth'])
        all_detections.append(array)
        heights.append(batch['original_shape'][0])
        widths.append(batch['original_shape'][1])
        covered.append(batch['image_id'])
        if diagnostic:
            torch.save(dict(image_id=batch['image_id'], file_name=batch['file_name'],
                            predictions=detections.cpu(), internal=model.last_diagnostics,
                            ratio=batch['ratio'], pad=batch['pad']),
                       output / ('diagnostic_%s.pt' % batch['image_id']))
            model.last_diagnostics = None
        if (index + 1) % 100 == 0:
            print('EVAL_PROGRESS', index + 1, '/', len(dataset),
                  'seconds=', round(time.time() - start, 1), flush=True)
    assert model.fingerprint() == expected_fingerprint
    assert len(covered) == len(set(covered)) == len(dataset)
    dump_json(output / 'predictions.json', records)
    metrics = run_evaluator(all_ground_truth, all_detections, heights, widths, output)
    dump_json(output / 'coverage.json', dict(ids=covered, empty_predictions=empty,
              skipped_batches=0, seconds=time.time() - start, batch=1, imgsz=imgsz,
              max_det=500, preprocess='fixed_letterbox', model_fingerprint=expected_fingerprint))
    model.train(was_training)
    print('NATIVE_METRICS', json.dumps(metrics), flush=True)
    return metrics


if __name__ == '__main__':
    raise SystemExit('Use train_pair.py so model identity and paired initialization are verified.')

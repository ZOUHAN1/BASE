"""Dataset-native AI-TODv2 and VisDrone evaluation for paired RT-DETR runs."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch.distributed as torch_dist

from src.misc import dist


class NativeDetectionEvaluator:
    def __init__(self, dataset, protocol):
        self.dataset = dataset
        self.protocol = protocol
        self.predictions = {}

    @classmethod
    def maybe_create(cls, dataset):
        protocol = dataset.coco.dataset.get('info', {}).get('native_evaluation')
        return cls(dataset, protocol) if protocol else None

    def update(self, predictions):
        for image_id, item in predictions.items():
            self.predictions[int(image_id)] = {
                key: value.detach().cpu().numpy() for key, value in item.items()}

    def synchronize_and_evaluate(self, output_dir):
        gathered = dist.all_gather(self.predictions)
        metrics = None
        if dist.is_main_process():
            predictions = {}
            for part in gathered:
                predictions.update(part)
            if set(predictions) != set(self.dataset.coco.imgs):
                raise RuntimeError('Native validation image coverage mismatch')
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                metrics = (self.aitod(predictions) if self.protocol['dataset'] == 'aitodv2'
                           else self.visdrone(predictions))
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'native_eval_latest.json').write_text(json.dumps(metrics, indent=2) + '\n')
            (output_dir / 'native_eval_stdout.txt').write_text(capture.getvalue())
            print('NATIVE_EVALUATION', self.protocol['dataset'], metrics, flush=True)
        if dist.is_dist_available_and_initialized():
            payload = [metrics]
            torch_dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        return metrics

    def aitod(self, predictions):
        repository = Path(__file__).resolve().parents[4]
        source = repository / 'third_party/cocoapi_aitod'
        if list((source / 'aitodpycocotools').glob('_mask*.so')) or list(
                (source / 'aitodpycocotools').glob('_mask*.pyd')):
            sys.path.insert(0, str(source))
        from aitodpycocotools.coco import COCO
        from aitodpycocotools.cocoeval import COCOeval
        ground_truth = COCO(self.dataset.ann_file)
        rows = []
        for image_id, prediction in predictions.items():
            for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
                x1, y1, x2, y2 = box.tolist()
                rows.append(dict(image_id=image_id, category_id=int(label), score=float(score),
                                 bbox=[x1, y1, max(0., x2-x1), max(0., y2-y1)]))
        if rows:
            detections = ground_truth.loadRes(rows)
        else:
            detections = COCO()
            detections.dataset = dict(images=list(ground_truth.imgs.values()),
                                      categories=list(ground_truth.cats.values()), annotations=[])
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        precision = evaluator.eval['precision']
        params = evaluator.params
        max_det = params.maxDets.index(1500)

        def ap(area='all', iou=None):
            indices = np.arange(len(params.iouThrs)) if iou is None else np.flatnonzero(
                np.isclose(params.iouThrs, iou))
            values = precision[indices, :, :, params.areaRngLbl.index(area), max_det]
            values = values[values > -1]
            return float(values.mean()) if values.size else None

        return dict(AP=ap(), AP50=ap(iou=.5), AP75=ap(iou=.75), APvt=ap('verytiny'),
                    APt=ap('tiny'), APs=ap('small'), APm=ap('medium'),
                    max_det=1500, prediction_limit=300)

    def visdrone(self, predictions):
        repository = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(repository / 'yolo11/visdrone/third_party/visdrone_eval'))
        from viseval import eval_det
        ground_truth, detections, heights, widths = [], [], [], []
        for image_id in sorted(predictions):
            metadata = self.dataset.coco.imgs[image_id]
            raw = Path(self.protocol['raw_annotation_dir']) / (Path(metadata['file_name']).stem + '.txt')
            with raw.open(newline='') as stream:
                rows = [[int(float(value)) for value in row[:8]] for row in csv.reader(stream) if row]
            gt = np.asarray(rows, dtype=np.int32).reshape(-1, 8)
            prediction = predictions[image_id]
            boxes = prediction['boxes'].copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, metadata['width'])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, metadata['height'])
            boxes[:, 2:] -= boxes[:, :2]
            values = np.column_stack((boxes, prediction['scores'], prediction['labels'] + 1))
            values = values[np.argsort(-values[:, 4])]
            ground_truth.append(gt)
            detections.append(values)
            heights.append(metadata['height'])
            widths.append(metadata['width'])
        values = eval_det(ground_truth, detections, heights, widths, per_class=True)
        ap, ap50, ap75, ar1, ar10, ar100, ar500, classwise = values
        return dict(AP=float(ap/100), AP50=float(ap50/100), AP75=float(ap75/100),
                    AR1=float(ar1/100), AR10=float(ar10/100), AR100=float(ar100/100),
                    AR500=float(ar500/100), per_class_AP=(np.asarray(classwise)/100).tolist(),
                    prediction_limit=300)

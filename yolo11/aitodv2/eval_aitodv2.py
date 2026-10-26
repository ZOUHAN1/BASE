import argparse
import bootstrap
import contextlib
import copy
import io
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from aitodpycocotools.coco import COCO
from aitodpycocotools.cocoeval import COCOeval
from ultralytics.utils.ops import non_max_suppression, clip_boxes
from common import dump_json


def evaluator_params(evaluator):
    p = evaluator.params
    return dict(iou_type=p.iouType, iou_thresholds=p.iouThrs.tolist(),
                rec_thresholds=p.recThrs.tolist(), max_dets=p.maxDets,
                ap_max_dets=p.maxDets[-1], area_labels=p.areaRngLbl, area_ranges=p.areaRng)


def extract_metrics(evaluator, categories):
    p = evaluator.params
    precision, recall = evaluator.eval['precision'], evaluator.eval['recall']
    md = p.maxDets.index(1500)
    ious = np.flatnonzero((p.iouThrs >= .5 - 1e-8) & (p.iouThrs <= .95 + 1e-8))
    def mean(values):
        good = values[values > -1]
        return float(good.mean()) if good.size else None
    def ap(area='all', iou=None, category=None):
        selected_ious = ious if iou is None else np.flatnonzero(np.isclose(p.iouThrs, iou))
        values = precision[selected_ious, :, :, p.areaRngLbl.index(area), md]
        if category is not None:
            values = values[:, :, p.catIds.index(category)]
        return mean(values)
    result = dict(AP=ap(), AP50=ap(iou=.5), AP75=ap(iou=.75),
                  APvt=ap('verytiny'), APt=ap('tiny'), APs=ap('small'), APm=ap('medium'))
    for det in (1, 100, 1500):
        result['AR@%d' % det] = mean(recall[ious, :, 0, p.maxDets.index(det)])
    result['per_class_AP'] = {cat['name']: ap(category=cat['id']) for cat in categories}
    return result


def run_evaluator(annotation_file, predictions, out_dir, covered_ids):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        gt = COCO(str(annotation_file))
        assert len(covered_ids) == len(set(covered_ids))
        assert set(covered_ids) == set(gt.getImgIds()), 'Missing or duplicate evaluation image IDs'
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(images=copy.deepcopy(gt.dataset['images']),
                              categories=copy.deepcopy(gt.dataset['categories']), annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        params = evaluator_params(evaluator)
        assert params['max_dets'] == [1, 100, 1500]
        assert params['area_labels'] == ['all', 'verytiny', 'tiny', 'small', 'medium']
        assert np.allclose(params['iou_thresholds'], np.arange(.5, 1, .05))
        evaluator.params.imgIds = list(covered_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = extract_metrics(evaluator, gt.dataset['categories'])
    (out_dir / 'evaluator_stdout.txt').write_text(stdout.getvalue())
    dump_json(out_dir / 'evaluator_params.json', params)
    dump_json(out_dir / 'native.json', dict(raw=metrics,
              display={k: v * 100 for k, v in metrics.items() if isinstance(v, (float, int))},
              evaluated_images=len(covered_ids), predictions=len(predictions), params=params,
              official_stats_raw=evaluator.stats.tolist()))
    return metrics


class LetterboxEvaluationDataset(Dataset):
    def __init__(self, annotation_file, image_dir, image_ids=None):
        self.doc = json.loads(Path(annotation_file).read_text())
        self.images = [im for im in self.doc['images'] if image_ids is None or im['id'] in image_ids]
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.image_dir / Path(info['file_name']).name
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        assert (w, h) == (info['width'], info['height'])
        ratio = min(800 / h, 800 / w)
        nw, nh = round(w * ratio), round(h * ratio)
        left, top = round((800 - nw) / 2 - .1), round((800 - nh) / 2 - .1)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((800, 800, 3), 114, dtype=np.uint8)
        out[top:top + nh, left:left + nw] = resized
        mask = np.zeros((1, 800, 800), dtype=np.float32)
        mask[:, top:top + nh, left:left + nw] = 1
        rgb = np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1))
        return dict(img=torch.from_numpy(rgb), mask=torch.from_numpy(mask), image_id=info['id'],
                    ratio=ratio, pad=(left, top), original_shape=(h, w), file_name=info['file_name'])


def undo_letterbox(boxes, ratio, pad, original_shape):
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    clip_boxes(boxes, original_shape)
    return boxes


@torch.inference_mode()
def evaluate_model(model, annotation_file, image_dir, output, amp=True, diagnostics_ids=None,
                   workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LetterboxEvaluationDataset(annotation_file, image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=True, collate_fn=lambda batch: batch[0])
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    expected_fingerprint = model.fingerprint()
    print('EVAL_IDENTITY', model.arm, expected_fingerprint, 'images=', len(dataset), flush=True)
    categories = dataset.doc['categories']
    records, covered, cap_hits, empty = [], [], 0, 0
    start = time.time()
    for index, batch in enumerate(loader):
        image = batch['img'][None].to(device, non_blocking=True).float() / 255
        if getattr(model, '_channels_last_inference', False):
            image = image.contiguous(memory_format=torch.channels_last)
        mask = batch['mask'][None].to(device, non_blocking=True)
        diagnostic = diagnostics_ids is not None and batch['image_id'] in diagnostics_ids
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            predictions = model(image, pixel_valid=mask, diagnostics=diagnostic)
        raw = predictions[0] if isinstance(predictions, tuple) else predictions
        candidate_count = int((raw[:, 4:12] > .001).sum())
        cap_hits += candidate_count > 200000
        detections = non_max_suppression(raw.float(), conf_thres=.001, iou_thres=.7,
                      multi_label=True, agnostic=False, max_det=1500, max_nms=200000,
                      max_time_img=1000000, nc=8, in_place=False)[0]
        if detections.numel():
            boxes = undo_letterbox(detections[:, :4], batch['ratio'], batch['pad'], batch['original_shape'])
            for box, score, category in zip(boxes.cpu().tolist(), detections[:, 4].cpu().tolist(),
                                           detections[:, 5].long().cpu().tolist()):
                x1, y1, x2, y2 = box
                records.append(dict(image_id=batch['image_id'], category_id=categories[category]['id'],
                                    bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
        else:
            empty += 1
        covered.append(batch['image_id'])
        if diagnostic:
            torch.save(dict(image_id=batch['image_id'], file_name=batch['file_name'],
                            predictions=detections.cpu(), internal=model.last_diagnostics,
                            ratio=batch['ratio'], pad=batch['pad']),
                       output / ('diagnostic_%s.pt' % batch['image_id']))
            model.last_diagnostics = None
        if (index + 1) % 200 == 0:
            print('EVAL_PROGRESS', index + 1, '/', len(dataset), 'seconds=', round(time.time() - start, 1), flush=True)
    assert model.fingerprint() == expected_fingerprint
    dump_json(output / 'predictions.json', records)
    metrics = run_evaluator(annotation_file, records, output, covered)
    dump_json(output / 'coverage.json', dict(ids=covered, empty_predictions=empty, max_nms_cap_hits=cap_hits,
              skipped_batches=0, nms_timeouts=0, seconds=time.time() - start, batch=1, imgsz=800,
              preprocess='fixed_letterbox', model_fingerprint=expected_fingerprint))
    model.train(was_training)
    print('NATIVE_METRICS', json.dumps(metrics), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deployment-variant', choices=('full', 'speed_v2', 'speed_v3', 'speed_v4', 'speed_v5'))
    parser.add_argument('--disable-fast-reconstruct', action='store_true')
    parser.add_argument('--fuse', action='store_true')
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--optimized-eager', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    from models.asymmetric_yolo11 import load_paired
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model, _ = load_paired(ckpt['arm'], bootstrap.ROOT / 'weights/paired_native_init.pt')
    deployment_variant = args.deployment_variant or ckpt.get('deployment_variant', 'full')
    model.deployment_variant = deployment_variant
    if ckpt.get('format') == 'brsa_speed_v2_deployment':
        model.prune_deployment_()
        model.load_state_dict(ckpt['state_dict'], strict=True)
    else:
        model.load_state_dict(ckpt['ema_state'], strict=True)
    if hasattr(model, 'brsa'):
        model.brsa.fast_reconstruct = not args.disable_fast_reconstruct
    model.cuda().eval()
    if args.optimized_eager:
        model.fuse_for_inference_(bind_fast=False)
        roots = ((model.native,) if model.arm == 'baseline_native' else
                 (model.backbone, model.detect, model.brr3, model.brr4,
                  model.arr4, model.arr5, model.shared_backbone_projections,
                  model.shared_delta_key_projection))
        for root in roots:
            for module in root.modules():
                if isinstance(module, (
                        torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
                    module.to(dtype=torch.float16)
        model.bind_fast_eager_()
        if model.arm == 'full_asymmetric':
            model.set_parallel_inference_(False)
            model.set_parallel_detect_head_(False)
            model.overlap_arr_queries = False
            model.overlap_brsa_project = False
            model.brr3.overlap_query = model.brr4.overlap_query = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    elif args.fuse:
        model.fuse_for_inference_()
    if args.channels_last or args.optimized_eager:
        model.to(memory_format=torch.channels_last)
        object.__setattr__(model, '_channels_last_inference', True)
    evaluate_model(model, args.annotations, args.images, args.output, workers=args.workers)

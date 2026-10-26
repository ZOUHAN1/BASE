import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/yolo11_ultralytics'))
sys.path.insert(0, str(ROOT / 'rtdetr/dota_hbb'))

from yolov8.model import build_yolov8m
from yolov8.training import PairedYOLOv8
from ultralytics.cfg import get_cfg
from src.data.native_detection_eval import NativeDetectionEvaluator


@pytest.mark.parametrize('arm', ['baseline_native', 'full_asymmetric'])
def test_yolov8_detection_and_occupancy_loss(arm):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    native, _ = build_yolov8m('native', 8)
    model = PairedYOLOv8(native.detector, arm).train()
    model.args = get_cfg()
    batch = dict(img=torch.rand(2,3,128,128), pixel_valid_mask=torch.ones(2,1,128,128),
                 cls=torch.tensor([[1.],[2.]]), batch_idx=torch.tensor([0,1]),
                 bboxes=torch.tensor([[.5,.5,.1,.1],[.3,.4,.1,.1]]))
    loss, items = model.loss(batch)
    assert torch.isfinite(loss) and len(items) == 4
    loss.backward()
    if arm == 'full_asymmetric':
        assert model.loss_report['fg_raw'] > 0
        assert model.network.brsa.modulation[-1].weight.grad.abs().sum() > 0
        assert model.network.arr4.score[1].__class__ is torch.nn.SiLU
        assert model.network.arr4.gate[1].__class__ is torch.nn.SiLU
    else:
        assert model.loss_report['fg_raw'] == 0


def test_native_aitod_evaluator_perfect_detection(tmp_path):
    document = dict(info={}, images=[dict(id=1,width=100,height=100,file_name='one.jpg')],
                    categories=[dict(id=0,name='object')], annotations=[dict(id=1,image_id=1,
                    category_id=0,bbox=[10,10,4,4],area=16,iscrowd=0)])
    path = tmp_path / 'annotation.json'
    path.write_text(json.dumps(document))
    dataset = SimpleNamespace(ann_file=str(path))
    evaluator = NativeDetectionEvaluator(dataset, dict(dataset='aitodv2'))
    metrics = evaluator.aitod({1:dict(boxes=np.array([[10.,10.,14.,14.]]),
                                     scores=np.array([.9]), labels=np.array([0]))})
    for key in ('AP','AP50','AP75','APvt'):
        assert metrics[key] == pytest.approx(1.)


def test_native_visdrone_evaluator_label_offset(tmp_path):
    (tmp_path / 'one.txt').write_text('10,10,4,4,1,1,0,0\n')
    dataset = SimpleNamespace(coco=SimpleNamespace(imgs={1:dict(file_name='one.jpg',width=100,height=100)}))
    evaluator = NativeDetectionEvaluator(dataset, dict(dataset='visdrone',raw_annotation_dir=str(tmp_path)))
    metrics = evaluator.visdrone({1:dict(boxes=np.array([[10.,10.,14.,14.]]),
                                        scores=np.array([.9]), labels=np.array([0]))})
    assert metrics['AP50'] == pytest.approx(1.)
    assert metrics['AP'] == pytest.approx(1.)

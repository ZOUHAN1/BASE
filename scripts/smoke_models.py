"""CPU-only source smoke checks. No checkpoints, downloads, timing or training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(name, arm, size):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    if name.startswith('yolo11'):
        task = name.split('-')[1]
        folder = ROOT / 'yolo11' / task
        sys.path.insert(0, str(folder))
        import bootstrap
        from models.asymmetric_yolo11 import native_model, PairedDetectionModel
        model = PairedDetectionModel(native_model(), arm)
    elif name == 'yolov8':
        sys.path.insert(0, str(ROOT / 'yolov8'))
        from model import build_yolov8m
        model, _ = build_yolov8m('brsa' if arm == 'brsa' else 'native', 8)
        if arm == 'full_at':
            from full_at.asymmetric_yolov8 import AsymmetricYOLOv8
            model = AsymmetricYOLOv8(model.detector)
    else:
        task = 'aitodv2' if name == 'rtdetr-aitodv2' else 'dota_hbb'
        folder = ROOT / 'rtdetr' / task
        sys.path.insert(0, str(folder))
        from src.core import YAMLConfig
        if task == 'aitodv2':
            filename = ('brsa_aitodv2_r50vd_brsa_enhance_aux_detached_e80_resume.yml'
                        if arm == 'brsa' else 'brsa_aitodv2_r50vd_baseline_e80_resume.yml')
        else:
            filename = ('dota1_hbb_r50vd_full_e100.yml' if arm == 'full_at'
                        else 'dota1_hbb_r50vd_baseline_e100.yml')
        config = YAMLConfig(str(folder / 'configs/rtdetr' / filename))
        config.yaml_cfg['HybridEncoder']['eval_spatial_size'] = [size, size]
        config.yaml_cfg['RTDETRTransformer']['eval_spatial_size'] = [size, size]
        config.yaml_cfg['PResNet']['pretrained'] = False
        model = config.model
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, size, size))

    def shape(value):
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), 'nonfinite output'
            return list(value.shape)
        if isinstance(value, dict):
            return {key: shape(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [shape(item) for item in value]
        return type(value).__name__

    result = dict(model=name, arm=arm, input_size=size,
                  parameters=sum(p.numel() for p in model.parameters()), output=shape(output))
    print('SMOKE_RESULT ' + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['yolov8', 'yolo11-aitodv2', 'yolo11-visdrone',
                                           'rtdetr', 'rtdetr-aitodv2'])
    parser.add_argument('--arm')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--report', type=Path, default=ROOT / 'verification-local/smoke.json')
    args = parser.parse_args()
    if args.model:
        check(args.model, args.arm, args.size)
        return
    matrix = [('yolov8', 'native'), ('yolov8', 'brsa'), ('yolov8', 'full_at'),
              ('yolo11-aitodv2', 'baseline_native'), ('yolo11-aitodv2', 'full_asymmetric'),
              ('yolo11-visdrone', 'baseline_native'), ('yolo11-visdrone', 'full_asymmetric'),
              ('rtdetr', 'baseline'), ('rtdetr', 'full_at'),
              ('rtdetr-aitodv2', 'baseline'), ('rtdetr-aitodv2', 'brsa')]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', WANDB_DISABLED='true',
                       YOLO_CONFIG_DIR=str(ROOT / 'runtime/yolo'),
                       MPLCONFIGDIR=str(ROOT / 'runtime/matplotlib'))
    results = []
    for name, arm in matrix:
        completed = subprocess.run([sys.executable, __file__, '--model', name, '--arm', arm,
                                    '--size', str(args.size)], env=environment, cwd=ROOT,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, end='', flush=True)
        completed.check_returncode()
        lines = [line[len('SMOKE_RESULT '):] for line in completed.stdout.splitlines()
                 if line.startswith('SMOKE_RESULT ')]
        if len(lines) != 1:
            raise RuntimeError(f'Missing smoke result for {name}/{arm}')
        results.append(json.loads(lines[0]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()

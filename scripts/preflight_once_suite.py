"""One disposable synthetic forward/backward per arm, not an extra training run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]


def check(suite, family, dataset, selected_device):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device(selected_device)
    jobs = [job for job in suite['jobs'] if job['family'] == family and job['dataset'] == dataset]
    sys.path.insert(0, str(REPOSITORY))
    if family != 'rtdetr':
        sys.path.insert(0, str(REPOSITORY / 'yolo11' / dataset))
        import bootstrap
        from ultralytics.cfg import get_cfg
        if family == 'yolov8m':
            from yolov8.training import load_paired
        else:
            from models.asymmetric_yolo11 import load_paired
    else:
        sys.path.insert(0, str(REPOSITORY / 'rtdetr/dota_hbb'))
        from src.core import YAMLConfig
    records = []
    for job in jobs:
        torch.manual_seed(0)
        if family != 'rtdetr':
            model, report = load_paired(job['arm'], job['paired_init'])
            model.to(device).train()
            model.args = get_cfg()
            batch = dict(img=torch.rand(2, 3, 128, 128, device=device),
                         bboxes=torch.tensor([[.5,.5,.1,.1], [.3,.4,.12,.08]], device=device),
                         batch_idx=torch.tensor([0,1], device=device),
                         cls=torch.tensor([[1.],[2.]], device=device),
                         pixel_valid_mask=torch.ones(2,1,128,128, device=device))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss, items = model.loss(batch)
            details = model.loss_report
        else:
            config = YAMLConfig(job['config'])
            config.yaml_cfg['HybridEncoder']['eval_spatial_size'] = [128,128]
            config.yaml_cfg['RTDETRTransformer']['eval_spatial_size'] = [128,128]
            model = config.model.to(device).train()
            state = torch.load(job['paired_init'], map_location='cpu', weights_only=False)['model']
            matched = {key: value for key, value in state.items()
                       if key in model.state_dict() and model.state_dict()[key].shape == value.shape}
            model.load_state_dict(matched, strict=False)
            criterion = config.criterion.to(device)
            targets = [dict(boxes=torch.tensor([[.5,.5,.1,.1]], device=device),
                            labels=torch.tensor([1], device=device),
                            pixel_valid_mask=torch.ones(128,128, device=device)) for _ in range(2)]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                result = model(torch.rand(2,3,128,128, device=device), targets)
            details = criterion(result, targets)
            loss = sum(details.values())
            details = {key: float(value.detach()) for key, value in details.items()}
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite preflight loss: {job["id"]}')
        loss.backward()
        bad = [name for name, parameter in model.named_parameters()
               if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        if bad:
            raise RuntimeError(f'Nonfinite gradients: {bad}')
        if job['arm'] == 'full_asymmetric':
            modulation = [(name, parameter) for name, parameter in model.named_parameters()
                          if 'modulation' in name and parameter.grad is not None]
            if not modulation or not any(parameter.grad.abs().sum() > 0 for _, parameter in modulation):
                raise RuntimeError('BASE spectral modulation has no gradient')
        records.append(dict(job=job['id'], loss=float(loss.detach()), details=details,
                            parameters=sum(parameter.numel() for parameter in model.parameters()),
                            finite_gradients=True))
        print('PREFLIGHT', records[-1], flush=True)
        del model, loss
        torch.cuda.empty_cache()
    destination = Path(suite['root']) / 'preflight' / f'{family}_{dataset}_s0.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(passed=True, checks=records,
        note='Disposable synthetic forward/backward only; formal runs load fresh paired weights.'), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--family', choices=['yolo11m','yolov8m','rtdetr'])
    parser.add_argument('--dataset', choices=['aitodv2','visdrone'])
    parser.add_argument('--only-yolo', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if args.family:
        check(suite, args.family, args.dataset, args.device)
        return
    families = ['yolo11m','yolov8m'] if args.only_yolo else ['yolo11m','yolov8m','rtdetr']
    for family in families:
        for dataset in ('aitodv2','visdrone'):
            subprocess.run([sys.executable, __file__, '--suite', str(args.suite), '--family', family,
                            '--dataset', dataset, '--device', args.device], cwd=REPOSITORY, check=True)


if __name__ == '__main__':
    main()

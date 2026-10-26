"""Run one native/full YOLO experiment using the preserved paired trainer."""
import argparse
import json
import os
from pathlib import Path
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    if spec['seed'] != 0 or spec['deployment_variant'] != 'full':
        raise ValueError('This single-run suite uses seed=0 and the full paper method only')
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise ValueError('Each arm uses two DDP ranks')
    check = json.loads(Path(spec['preflight_report']).read_text())
    if not check['passed']:
        raise RuntimeError('Single-batch method preflight must pass before formal training')
    sys.path.insert(0, str(REPOSITORY))
    task = REPOSITORY / 'yolo11' / spec['dataset']
    sys.path.insert(0, str(task))
    import train_pair
    import torch.distributed as dist
    train_pair.ROOT = Path(spec['experiment_root'])
    trainer = train_pair.PairTrainer(spec, smoke=False, resume=args.resume)
    trainer.train()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

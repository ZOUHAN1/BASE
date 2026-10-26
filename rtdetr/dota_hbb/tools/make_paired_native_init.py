"""Create the shared native RT-DETR initialization for paired experiments."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

from src.core import YAMLConfig
from src.misc import dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dist.set_seed(args.seed)
    cfg = YAMLConfig(args.config)
    model = cfg.model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'model': model.state_dict(), 'seed': args.seed, 'config': args.config}, args.output)
    print(f'saved paired native init to {args.output}')


if __name__ == '__main__':
    main()

"""Portable launcher for the preserved two-rank YOLO11 training code."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['aitodv2', 'visdrone'], required=True)
    parser.add_argument('--arm', choices=['native', 'full'], default='native')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--port', type=int, default=29571)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify-smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    folder = ROOT / 'yolo11' / args.dataset
    init = folder / 'weights/paired_native_init.pt'
    if not init.is_file():
        parser.error(f'Create paired initialization first: {init}')
    if args.verify_smoke:
        inputs = []
        for arm in ('baseline_native', 'full_asymmetric'):
            run = folder / 'smoke' / arm
            done = json.loads((run / 'smoke_complete.json').read_text())
            roundtrip = json.loads((run / 'checkpoint_roundtrip.json').read_text())
            launch = json.loads((run / 'release_launch.json').read_text())
            if done['completed_epoch'] < 2 or not done['checkpoint_roundtrip']:
                raise RuntimeError(f'Incomplete smoke: {arm}')
            if not roundtrip['exact_ema'] or roundtrip['rng_ranks'] != 2:
                raise RuntimeError(f'Checkpoint smoke failed: {arm}')
            if launch['init_sha256'] != digest(init):
                raise RuntimeError('Paired initialization changed after smoke')
            inputs.append(launch['manifest_sha256'])
        if inputs[0] != inputs[1]:
            raise RuntimeError('Smoke arms used different dataset manifests')
        destination = folder / 'artifacts/pair_smoke_verified.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(dict(verified=True, manifest_sha256=inputs[0],
                                               init_sha256=digest(init)), indent=2) + '\n')
        print(destination)
        return
    if args.data_root is None:
        parser.error('--data-root is required for training')
    devices = [int(part) for part in args.devices.split(',')]
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error('Choose exactly two distinct GPU indices')
    spec = yaml.safe_load((folder / 'configs' / (args.arm + '.yaml')).read_text())
    spec.pop('devices', None)
    spec['data_root'] = str(args.data_root.resolve())
    spec['epochs'] = args.epochs
    listing = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    gpu_map = {int(index.strip()): uuid.strip() for index, uuid in csv.reader(listing.splitlines())}
    spec['gpus'] = [dict(physical_index=index, logical_index=rank, uuid=gpu_map[index])
                    for rank, index in enumerate(devices)]
    identity = dict(init_sha256=digest(init),
                    manifest_sha256=digest(args.data_root.resolve() / 'manifest.json'))
    if not args.smoke:
        marker = json.loads((folder / 'artifacts/pair_smoke_verified.json').read_text())
        if any(marker[key] != value for key, value in identity.items()):
            raise RuntimeError('Repeat paired smoke for the current weights and dataset')
    config = folder / 'runtime' / (args.arm + '.yaml')
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(spec, sort_keys=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpu['uuid'] for gpu in spec['gpus']))
    env['BRSA_PAIR_VISIBLE_UUIDS'] = env['CUDA_VISIBLE_DEVICES']
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port', str(args.port), 'train_pair.py', '--config', str(config)]
    if args.smoke:
        command.append('--smoke')
    if args.resume:
        command += ['--resume', str(args.resume.resolve())]
    subprocess.run(command, cwd=folder, env=env, check=True)
    if args.smoke:
        (folder / 'smoke' / spec['smoke_name'] / 'release_launch.json').write_text(
            json.dumps(identity, indent=2) + '\n')


if __name__ == '__main__':
    main()

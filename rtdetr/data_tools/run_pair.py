#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'dota_hbb'
WORKDIR = ROOT
PYTHON = sys.executable
INITIALIZATION = ROOT / 'runtime' / 'paired_native_init.pth'
STATUS_PATH = ROOT / 'runtime' / 'pair_status.json'

RUNS = {
    'baseline': {
        'devices': os.environ.get('BRSA_BASELINE_DEVICES', '0,1'),
        'port': '29871',
        'config': 'configs/rtdetr/dota1_hbb_r50vd_baseline_e100.yml',
        'log': ROOT / 'logs' / 'baseline_train.log',
    },
    'brr_brsa_arr': {
        'devices': os.environ.get('BRSA_FULL_DEVICES', '2,3'),
        'port': '29872',
        'config': 'configs/rtdetr/dota1_hbb_r50vd_full_e100.yml',
        'log': ROOT / 'logs' / 'brr_brsa_arr_train.log',
    },
}


children = {}
stopping = False


def now():
    return datetime.now().astimezone().isoformat()


def write_status(stage, message=None):
    payload = {
        'stage': stage,
        'updated_at': now(),
        'supervisor_pid': os.getpid(),
        'message': message,
        'runs': {},
    }
    for name, item in RUNS.items():
        process = children.get(name)
        payload['runs'][name] = {
            'pid': process.pid if process else None,
            'returncode': process.poll() if process else None,
            'devices': item['devices'],
            'config': item['config'],
            'log': str(item['log']),
        }
    temporary = STATUS_PATH.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATUS_PATH)


def stop_children(signum, _frame):
    global stopping
    stopping = True
    write_status('stopping', f'received signal {signum}')
    for process in children.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


def launch(name, item):
    environment = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    environment.update({
        'CUDA_VISIBLE_DEVICES': item['devices'],
        'OMP_NUM_THREADS': '1',
        'PYTHONUNBUFFERED': '1',
    })
    command = [
        PYTHON, '-m', 'torch.distributed.run',
        '--nproc_per_node=2', f"--master_port={item['port']}",
        'tools/train.py', '-c', item['config'], '-t', str(INITIALIZATION),
        '--amp', '--seed', '0',
    ]
    item['log'].parent.mkdir(parents=True, exist_ok=True)
    log_handle = item['log'].open('ab', buffering=0)
    process = subprocess.Popen(
        command,
        cwd=WORKDIR,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process.log_handle = log_handle
    children[name] = process


def main():
    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    if any((ROOT / 'runs' / directory).exists() for directory in (
            'rtdetr_r50vd_baseline_hbb_640_e100_s0',
            'rtdetr_r50vd_brr_brsa_arr_hbb_640_e100_s0')):
        write_status('failed', 'formal output directory already exists')
        return 2
    for name, item in RUNS.items():
        launch(name, item)
    write_status('running', 'both 2-GPU jobs launched')

    while True:
        returncodes = {name: process.poll() for name, process in children.items()}
        if all(code is not None for code in returncodes.values()):
            break
        write_status('stopping' if stopping else 'running')
        time.sleep(10)

    for process in children.values():
        process.log_handle.close()
    if stopping:
        write_status('stopped', 'supervisor was asked to stop')
        return 143
    failed = {name: code for name, code in returncodes.items() if code != 0}
    if failed:
        write_status('failed', f'nonzero child return codes: {failed}')
        return 1
    write_status('completed', 'both 100-epoch jobs completed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

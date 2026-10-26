"""Run each of twelve experiments exactly once; pair native/full on four GPUs."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

STOP = False


def stop(*_):
    global STOP
    STOP = True


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    root, repository = Path(suite['root']), Path(suite['repository'])
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = root / 'queue_started.json'
    if marker.exists():
        raise RuntimeError('This single-run queue has already been started; no automatic reruns allowed')
    if len(suite['jobs']) != 12 or len({job['id'] for job in suite['jobs']}) != 12:
        raise RuntimeError('Expected twelve unique seed-0 jobs')
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                       '--format=csv,noheader,nounits'], text=True)
    devices = {int(index): uuid.strip() for index, uuid in csv.reader(listing.splitlines())}
    for job in suite['jobs']:
        if Path(job['output']).exists():
            raise RuntimeError(f'Output already exists: {job["id"]}')
    atomic_json(marker, dict(pid=os.getpid(), started=time.time(), seed=0, repetitions=1))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    children = []
    handles = []

    def status(state):
        atomic_json(root / 'status.json', dict(state=state, controller_pid=os.getpid(),
                    updated=time.time(), jobs=suite['jobs']))

    try:
        for dataset in ('aitodv2', 'visdrone'):
            for family in ('yolo11m', 'yolov8m', 'rtdetr'):
                if STOP:
                    break
                children, handles = [], []
                pair = [job for job in suite['jobs'] if job['family'] == family and job['dataset'] == dataset]
                preflight = root / 'preflight' / f'{family}_{dataset}_s0.json'
                data_ready = root / 'data' / dataset / 'ready.json'
                while not STOP and not (preflight.is_file() and data_ready.is_file()
                                        and all(Path(job['paired_init']).is_file() for job in pair)):
                    if (root / 'assets_failed.json').exists():
                        raise RuntimeError('Asset preparation failed; inspect assets_failed.json')
                    status('waiting_assets')
                    time.sleep(10)
                if STOP:
                    break
                if not json.loads(preflight.read_text())['passed']:
                    raise RuntimeError('Method preflight failed')
                for job in pair:
                    physical = [0,1] if job['arm'] == 'baseline_native' else [2,3]
                    port = 29941 if job['arm'] == 'baseline_native' else 29942
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices[i] for i in physical),
                               OMP_NUM_THREADS='2', WANDB_DISABLED='true', PYTHONUNBUFFERED='1')
                    env['BRSA_PAIR_VISIBLE_UUIDS'] = env['CUDA_VISIBLE_DEVICES']
                    launch = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
                              '--master_port', str(port)]
                    if family != 'rtdetr':
                        spec = yaml.safe_load(Path(job['config']).read_text())
                        spec['gpus'] = [dict(physical_index=i, logical_index=rank, uuid=devices[i])
                                        for rank, i in enumerate(physical)]
                        config = root / 'configs' / (job['id'] + '_launch.yaml')
                        config.write_text(yaml.safe_dump(spec, sort_keys=False))
                        launch += [str(repository / 'scripts/train_yolo_once.py'), '--config', str(config)]
                        cwd = repository
                    else:
                        launch += ['tools/train.py', '--config', job['config'], '--tuning', job['paired_init'],
                                   '--amp', '--seed', '0']
                        cwd = repository / 'rtdetr/dota_hbb'
                    log = (root / 'logs' / (job['id'] + '.log')).open('x')
                    process = subprocess.Popen(launch, cwd=cwd, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(state='running', pid=process.pid, devices=physical, started=time.time())
                    children.append((job, process))
                    handles.append(log)
                    print('STARTED', job['id'], process.pid, physical, flush=True)
                status('running')
                while any(process.poll() is None for _, process in children) and not STOP:
                    for job, process in children:
                        code = process.poll()
                        if code is not None and job['state'] == 'running':
                            job.update(state='completed' if code == 0 else 'failed', returncode=code,
                                       finished=time.time())
                        if code not in (None, 0):
                            raise RuntimeError(f'{job["id"]} failed; no automatic retry')
                    status('running')
                    time.sleep(10)
                for job, process in children:
                    code = process.poll()
                    if code not in (0, None):
                        raise RuntimeError(f'{job["id"]} failed; no automatic retry')
                    if code == 0:
                        job.update(state='completed', returncode=0, finished=time.time())
                for handle in handles:
                    handle.close()
        status('stopped' if STOP else 'completed')
    except BaseException as error:
        suite['error'] = str(error)
        atomic_json(root / 'failure.json', dict(error=str(error), updated=time.time()))
        status('failed')
        raise
    finally:
        for job, process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()

import copy
import importlib.util
import json
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / 'scripts/run_once_suite.py'
spec = importlib.util.spec_from_file_location('once_queue', source)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def fixture_suite(tmp_path):
    (tmp_path / 'logs').mkdir()
    jobs = []
    for name, state in [('done', 'completed'), ('failed', 'failed'), ('interrupted', 'running')]:
        output = tmp_path / 'runs' / name
        output.mkdir(parents=True)
        if state == 'completed':
            (output / 'completed.json').write_text('{}')
        (tmp_path / 'logs' / (name + '.log')).write_text('old log')
        jobs.append(dict(id=name, state=state, output=str(output), config=name + '.yml',
                         paired_init=name + '.pt'))
    suite = dict(jobs=jobs)
    (tmp_path / 'status.json').write_text(json.dumps(dict(state='failed', jobs=jobs)))
    (tmp_path / 'failure.json').write_text('{}')
    return copy.deepcopy(suite)


def test_recovery_keeps_completed_and_archives_failed_starts(tmp_path):
    suite = fixture_suite(tmp_path)
    archive = queue.recover_failed_start(suite, tmp_path)
    assert [job['state'] for job in suite['jobs']] == ['completed', 'pending', 'pending']
    assert (tmp_path / 'runs/done/completed.json').is_file()
    assert (tmp_path / 'logs/done.log').read_text() == 'old log'
    assert (archive / 'failed.log').read_text() == 'old log'
    assert not (tmp_path / 'runs/failed').exists()
    assert not (tmp_path / 'failure.json').exists()


def test_recovery_refuses_partial_training_artifacts(tmp_path):
    suite = fixture_suite(tmp_path)
    (tmp_path / 'runs/failed/checkpoint.pth').write_text('retained state')
    with pytest.raises(RuntimeError, match='partial training'):
        queue.recover_failed_start(suite, tmp_path)
    assert not (tmp_path / 'attempts').exists()
    assert (tmp_path / 'logs/failed.log').is_file()


def test_recovery_refuses_live_pid(tmp_path, monkeypatch):
    suite = fixture_suite(tmp_path)
    previous = json.loads((tmp_path / 'status.json').read_text())
    previous['jobs'][1]['pid'] = 123
    (tmp_path / 'status.json').write_text(json.dumps(previous))
    monkeypatch.setattr(queue.os, 'kill', lambda pid, signal: None)
    with pytest.raises(RuntimeError, match='still alive'):
        queue.recover_failed_start(suite, tmp_path)
    assert not (tmp_path / 'attempts').exists()

"""Validate source syntax, includes, local links and release hygiene."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.pytest_cache', 'runtime', 'verification-local',
           'build', 'dist', 'smoke', 'runs', 'artifacts', 'weights'}


def files():
    if (ROOT / '.git').exists():
        listing = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
        candidates = {ROOT / item.decode() for item in listing.split(b'\0') if item}
    else:
        candidates = ROOT.rglob('*')
    return sorted(path for path in candidates if path.is_file() and not any(
        part in IGNORED or part.endswith('.egg-info') for part in path.relative_to(ROOT).parts))


def main():
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--check-index', action='store_true')
    args = parser.parse_args()
    errors = []
    payload = []
    sources, configs = 0, 0
    forbidden = re.compile(r'/tmp[/]zouhan/|/home/(?:hadoop-platcv|sankuai)/|'
                           r'GPU-[0-9a-f]{8}-|'
                           r'gh[pousr]_[A-Za-z0-9]{30,}|'
                           r'github_pat_[A-Za-z0-9_]{30,}|'
                           r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    for path in files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix in {'.pt', '.pth', '.so', '.pyc', '.zip', '.png', '.jpg', '.onnx'}:
            errors.append(f'Forbidden binary: {rel}')
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            errors.append(f'Unexpected binary: {rel}')
            continue
        if forbidden.search(content):
            errors.append(f'Private path, UUID or credential pattern: {rel}')
        if path.suffix == '.py':
            try:
                ast.parse(content, filename=rel)
                sources += 1
            except SyntaxError as error:
                errors.append(f'{rel}: {error}')
        if path.suffix in {'.yaml', '.yml'}:
            try:
                doc = yaml.safe_load(content)
                configs += 1
                if isinstance(doc, dict):
                    for include in doc.get('__include__', []):
                        if not (path.parent / include).is_file():
                            errors.append(f'Missing include: {rel} -> {include}')
            except yaml.YAMLError as error:
                errors.append(f'{rel}: {error}')
        if path.name != 'RELEASE_FILES.json':
            payload.append(dict(file=rel, size=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Check only release-authored docs; upstream READMEs reference omitted assets.
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                if args.write_manifest and target == 'RELEASE_FILES.json':
                    continue
                if not (path.parent / target.split('#')[0]).exists():
                    errors.append(f'Broken doc link: {path.name} -> {target}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        raise SystemExit(1)
    if args.check_index:
        listing = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
        tracked = {part.decode() for part in listing.split(b'\0') if part}
        expected = {item['file'] for item in payload} | {'RELEASE_FILES.json'}
        if tracked != expected:
            print('Missing from index:', sorted(expected - tracked), file=sys.stderr)
            print('Unexpected in index:', sorted(tracked - expected), file=sys.stderr)
            raise SystemExit(1)
    destination = ROOT / 'RELEASE_FILES.json'
    if args.write_manifest:
        destination.write_text(json.dumps(dict(files=payload), indent=2) + '\n')
    elif destination.exists():
        expected = json.loads(destination.read_text())['files']
        if expected != payload:
            raise SystemExit('Release manifest is stale; review changes and regenerate it.')
    print(f'Audit passed: {sources} Python files, {configs} YAML files, {len(payload)} payload files.')


if __name__ == '__main__':
    main()

"""Restore missing VisDrone images and finish RT-DETR single-run preparation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        command = ['curl', '-fL', '--connect-timeout', '20', '--max-time', '1800', '-C', '-',
                   url, '-o', str(destination)]
        if subprocess.run(command).returncode == 0:
            return
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f'Download failed after bounded retries: {url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    root, repository = Path(suite['root']), Path(suite['repository'])
    try:
        coco = root / 'weights/rtdetr_r50vd_coco.pth'
        # A complete source checkpoint was downloaded during preparation.
        subprocess.run([sys.executable, str(repository / 'scripts/init_rtdetr_once.py'),
                        '--suite', str(args.suite), '--weights', str(coco)], check=True)
        for dataset in ('aitodv2','visdrone'):
            subprocess.run([sys.executable, str(repository / 'scripts/preflight_once_suite.py'),
                            '--suite', str(args.suite), '--family', 'rtdetr', '--dataset', dataset,
                            '--device', 'cpu'], check=True)
        manifest = json.loads((root / 'data/visdrone/manifest.json').read_text())
        image_dir = Path(manifest['splits']['train']['image_dir'])
        annotation = json.loads((root / 'data/visdrone/visdrone_train.json').read_text())
        missing = [item for item in annotation['images'] if not (image_dir / item['file_name']).is_file()]
        if missing:
            archive = root / 'downloads/VisDrone2019-DET-train.zip'
            url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip'
            fetch(url, archive)
            image_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as source:
                entries = {Path(name).name: name for name in source.namelist()
                           if '/images/' in name and name.lower().endswith('.jpg')}
                for item in missing:
                    filename = Path(item['file_name']).name
                    if filename not in entries:
                        raise RuntimeError(f'Missing official image in archive: {filename}')
                    destination = image_dir / filename
                    temporary = destination.with_suffix('.download')
                    with source.open(entries[filename]) as stream, temporary.open('wb') as output:
                        shutil.copyfileobj(stream, output)
                    temporary.replace(destination)
        subprocess.run([sys.executable, str(repository / 'scripts/verify_once_data.py'),
                        '--root', str(root), '--dataset', 'visdrone'], check=True)
        (root / 'assets_complete.json').write_text(json.dumps(dict(complete=True, timestamp=time.time())) + '\n')
    except BaseException as error:
        (root / 'assets_failed.json').write_text(json.dumps(dict(error=str(error), timestamp=time.time())) + '\n')
        raise


if __name__ == '__main__':
    main()

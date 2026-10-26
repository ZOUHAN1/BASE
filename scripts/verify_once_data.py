"""Verify existing train/validation image lists before admitting a dataset."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['aitodv2','visdrone'], required=True)
    args = parser.parse_args()
    data = args.root / 'data' / args.dataset
    prefix = 'aitodv2' if args.dataset == 'aitodv2' else 'visdrone'
    result = {}
    for split in ('train','val'):
        annotation = json.loads((data / f'{prefix}_{split}.json').read_text())
        paths = [Path(line) for line in (data / f'{split}.txt').read_text().splitlines() if line]
        names = {Path(image['file_name']).name for image in annotation['images']}
        if len(paths) != len(names) or {path.name for path in paths} != names:
            raise RuntimeError(f'{split} image list differs from annotations')
        missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f'{len(missing)} missing {split} images; first: {missing[0]}')
        result[split] = dict(images=len(paths), annotations=len(annotation['annotations']))
    (data / 'ready.json').write_text(json.dumps(dict(verified=True, splits=result), indent=2) + '\n')
    print(args.dataset, result, flush=True)


if __name__ == '__main__':
    main()

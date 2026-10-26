import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--image-root', required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    image_root = Path(args.image_root).resolve()

    split_paths = {}
    for split in ('train', 'val'):
        document = json.loads((data_root / ('aitodv2_%s.json' % split)).read_text())
        paths = [image_root / split / Path(image['file_name']).name for image in document['images']]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError('%s is missing %d images; first=%s' % (split, len(missing), missing[0]))
        split_path = data_root / ('%s.txt' % split)
        split_path.write_text(''.join('%s\n' % path for path in paths))
        split_paths[split] = split_path

    train_lines = split_paths['train'].read_text().splitlines()
    (data_root / 'smoke_train.txt').write_text(''.join('%s\n' % path for path in train_lines[:128]))
    manifest_path = data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['image_root'] = str(image_root)
    manifest['remote_index_verified'] = {'train': len(train_lines),
                                          'val': len(split_paths['val'].read_text().splitlines())}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    config = {
        'path': str(data_root),
        'train': str(split_paths['train']),
        'val': str(split_paths['val']),
        'test': str(split_paths['val']),
        'nc': 8,
        'names': {index: category['name'] for index, category in
                  enumerate(manifest['categories'])},
    }
    (data_root / 'aitodv2.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(manifest['remote_index_verified'], sort_keys=True))


if __name__ == '__main__':
    main()

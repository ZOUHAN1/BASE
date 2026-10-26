"""Transfer one COCO detector initialization to both arms of each dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    sys.path.insert(0, str(Path(suite['repository']) / 'rtdetr/dota_hbb'))
    import torch
    from src.core import YAMLConfig
    torch.set_num_threads(2)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    source = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    sha256 = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    for dataset in ('aitodv2', 'visdrone'):
        job = next(job for job in suite['jobs'] if job['family'] == 'rtdetr'
                   and job['dataset'] == dataset and job['arm'] == 'baseline_native')
        destination = Path(job['paired_init'])
        if destination.exists():
            raise FileExistsError(f'Refusing to replace paired initialization: {destination}')
        torch.manual_seed(0)
        model = YAMLConfig(job['config']).model
        state = model.state_dict()
        loaded = {key: value for key, value in source.items() if key in state and state[key].shape == value.shape}
        missing = sorted(set(state) - set(loaded))
        allowed = ('decoder.denoising_class_embed', 'decoder.enc_score_head', 'decoder.dec_score_head')
        if any(not key.startswith(allowed) for key in missing):
            raise RuntimeError(f'Unexpected transfer omissions: {missing}')
        model.load_state_dict(loaded, strict=False)
        torch.save(dict(model=model.state_dict(), seed=0, coco_sha256=sha256,
                        loaded=len(loaded), reinitialized=missing), destination)
        print(dataset, 'COCO tensors loaded:', len(loaded), 'classification tensors reset:', len(missing), flush=True)


if __name__ == '__main__':
    main()

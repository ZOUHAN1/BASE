# BRSA-AT Detector Code

Source release for YOLOv8m, YOLO11m and RT-DETR-R50-vd baselines and their
BRSA-family integrations. This repository preserves distinct experimental
versions rather than silently treating every historical BRSA model as BRSA-AT.

## Included implementations

| Model | Baseline | Corresponding implementation | Status |
| --- | --- | --- | --- |
| YOLOv8m | Native P3/P4/P5 detector | Historical BRSA-only adapter | Restored inference graph; historical training configuration is archived |
| YOLOv8m | Same native detector | BRR + BRSA + ARR full neck | Architecture port used for cost auditing; not a historical accuracy checkpoint |
| YOLO11m | Native Ultralytics detector | BRR + BRSA + ARR asymmetric neck | AI-TODv2 and VisDrone training/evaluation source |
| RT-DETR-R50-vd | Native hybrid encoder | Original BRSA integration | AI-TODv2 training/evaluation source |
| RT-DETR-R50-vd | Native hybrid encoder | BRR + BRSA + ARR asymmetric neck | DOTA1.0 HBB training/evaluation source |

Baseline and modified implementations coexist in the same framework for each
experiment. Detector heads remain P3/P4/P5; the shallow feature is not an extra
P2 detection head. See [the code map](docs/MODEL_MAP.md) for entry points.

## Layout

```text
yolov8/                 historical graph and full_at architecture port
yolo11/aitodv2/         paired training, geometry, loss and evaluator
yolo11/visdrone/        paired training and VisDrone evaluator
rtdetr/aitodv2/         original RT-DETR BRSA experiment source
rtdetr/dota_hbb/        full asymmetric RT-DETR source and HBB merge evaluator
rtdetr/data_tools/      DOTA patch preparation and paired launchers
third_party/           pinned local Ultralytics and AI-TOD evaluator snapshots
configs/               historical recipes and selection rules
scripts/               portable launcher and source smoke/audit checks
```

Dataset images, annotations, checkpoints, logs, machine credentials, virtual
environments and prior Git histories are not included.

## Installation

Use a separate Python environment. Install a compatible PyTorch/torchvision
pair first. Packaging validation used PyTorch 2.4.1 and torchvision 0.19.1;
the upstream RT-DETR requirements are retained separately for provenance.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./third_party/cocoapi_aitod
python scripts/audit_release.py
python scripts/smoke_models.py
```

Each model entry point selects its vendored Ultralytics snapshot. Do not
install all snapshots into the same site-packages directory or mix them in
one Python process. Smoke checks intentionally use isolated subprocesses.

Optional optimized YOLO11 AI-TODv2 inference code is retained, including Triton
and C++ paths and the explicit CUDA Graph runner. Triton, a working compiler
and CUDA toolkit are only needed when enabling those paths. Training retains
the differentiable PyTorch implementation. CUDA Graph is not automatically
enabled by the CPU checks.

## Training and evaluation

See [reproduction instructions](docs/REPRODUCIBILITY.md) for dataset layout,
paired initialization, two-GPU launch commands, selection rules and RT-DETR
evaluation. Run commands from the working directories shown there.

Historical YOLOv8m training configuration is in
[`configs/historical_yolov8m.yaml`](configs/historical_yolov8m.yaml). It is not
presented as a complete restored historical training engine, nor as evidence
of training the full YOLOv8m BRSA-AT architecture port.

## Validation and provenance

The release preparation checked eleven CPU forward configurations and focused
module tests. These checks establish source integration, not detection AP or
GPU speed. See [validation](docs/VALIDATION.md).

[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records hashes before packaging
edits; [`RELEASE_FILES.json`](RELEASE_FILES.json) records the final released
payload. See [release notes](docs/RELEASE_NOTES.md) for path and launcher changes.

## Third-party notices

Ultralytics source retains its AGPL-3.0 license. RT-DETR retains its Apache-2.0
license. Evaluator licenses are retained beside their source. This repository
does not relicense those components under a blanket permissive license.
See [third-party notices](THIRD_PARTY_NOTICES.md).

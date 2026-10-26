# Reproduction Guide

## Scope

This is a source release, not a checkpoint or dataset release. The complete
historical YOLOv8m training engine was not available in the selected local
source tree; its restored inference graph and archived recipe are included
without substituting a newly written trainer for the historical one.
The full YOLOv8m BRSA-AT port is architecture code, not an accuracy claim.

For retained YOLO11 and RT-DETR experiments, training, auxiliary losses,
evaluation and model configurations are included. Source packaging has not
rerun full training. Use the data licenses and official splits separately.

## YOLO11 initialization and data

Put the corresponding official `yolo11m.pt` initialization in the task's
`weights/` directory. Loading checkpoints uses Python pickle in the retained
experimental code; load only trusted files.

For AI-TODv2, run from `yolo11/aitodv2/`. The image root contains
`train/`, `val/`, `test/`. The original local annotation mode expects
`instances_train.json`, `instances_val.json`, `instances_test.json`.
The preparation code retains split-count and ignore/crowd checks.

```bash
export AITODV2_IMAGE_ROOT=/absolute/path/to/images
python prepare.py --annotation-root /absolute/path/to/annotations --local
python prepare.py --init
```

This produces `data/local/`, its manifest and task YAML. Alternatively, use
the preparation code's official annotation mode with matching official JSON
names. Do not fabricate `formal_allowed` or replace annotations merely to
bypass validation checks.

For VisDrone, run from `yolo11/visdrone/`. `VISDRONE_ROOT/raw/` must contain
the extracted official `VisDrone2019-DET-train`, `VisDrone2019-DET-val` and
`VisDrone2019-DET-test-dev` directories, including annotations used by the
retained evaluation protocol.

```bash
export VISDRONE_ROOT=/absolute/path/to/VisDrone2019-DET
python prepare_visdrone.py --data --init
```

## YOLO11 paired training

Run the launcher from the repository root. It resolves the two chosen GPU
indices to UUIDs at launch rather than retaining private machine UUIDs.
Each arm uses two DDP ranks, global batch 16, and identical paired native
initialization. The default full arm is the unchanged full architecture,
not a reduced `speed_v2`-`speed_v5` structural variant.

```bash
python scripts/launch_yolo11.py --dataset aitodv2 --arm native \
  --data-root "$PWD/yolo11/aitodv2/data/local" --devices 0,1 --smoke
python scripts/launch_yolo11.py --dataset aitodv2 --arm full \
  --data-root "$PWD/yolo11/aitodv2/data/local" --devices 0,1 --smoke
python scripts/launch_yolo11.py --dataset aitodv2 --verify-smoke
python scripts/launch_yolo11.py --dataset aitodv2 --arm native \
  --data-root "$PWD/yolo11/aitodv2/data/local" --devices 0,1 --epochs 100
python scripts/launch_yolo11.py --dataset aitodv2 --arm full \
  --data-root "$PWD/yolo11/aitodv2/data/local" --devices 2,3 --epochs 100 --port 29572
```

Smoke runs are real two-epoch GPU jobs when explicitly invoked. They are
different from the CPU forward checks run during release preparation.
Formal launch requires both smoke completions and checkpoint round trips,
and binds that check to the initialization and dataset manifest hashes.
For VisDrone, use `--dataset visdrone` and its `data/visdrone` directory.
For the original longer AI-TODv2 schedule, set `--epochs 300`; the launcher
does not claim the 100-epoch recipe reproduces a 300-epoch checkpoint.

## RT-DETR

Run inside the selected `rtdetr/aitodv2/` or `rtdetr/dota_hbb/` directory.
Edit dataset paths in `configs/dataset/` before training. Relative paths
are resolved from this working directory. The release removes workstation
prefixes but does not invent or distribute missing datasets.

AI-TODv2 paired configurations:

| Arm | Config |
| --- | --- |
| Baseline | `configs/rtdetr/brsa_aitodv2_r50vd_baseline_e80_resume.yml` |
| Original BRSA | `configs/rtdetr/brsa_aitodv2_r50vd_brsa_enhance_aux_detached_e80_resume.yml` |

The `resume` filenames extend the schedule to 80 epochs; actual continuation
requires `--resume /path/to/checkpoint.pth`. For a new run, supply the intended
paired pretrained initialization through `--tuning`; do not compare one arm
from random weights with another from pretrained weights.

DOTA1.0 HBB paired configurations:

```bash
python tools/make_paired_native_init.py --help
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
  --master_port=29871 tools/train.py \
  -c configs/rtdetr/dota1_hbb_r50vd_baseline_e100.yml \
  --tuning /absolute/path/to/paired_native_init.pth --amp --seed 0
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 \
  --master_port=29872 tools/train.py \
  -c configs/rtdetr/dota1_hbb_r50vd_full_e100.yml \
  --tuning /absolute/path/to/paired_native_init.pth --amp --seed 0
```

Use `--test-only --resume /path/to/selected_checkpoint.pth --amp` with the
matching config for validation. Evaluation configuration and training loss
must match the checkpoint's class count and architecture.

## Evaluation and selection

| Protocol | Retained settings |
| --- | --- |
| AI-TODv2 YOLO11 | Input 800, AI-TOD COCO evaluator, score 0.001, class-aware NMS 0.7, maxDet 1500 |
| VisDrone YOLO11 | Input 640, VisDrone evaluator and ignored-region handling, maxDet 500 |
| DOTA RT-DETR | Input 640 from 1024 patches, overlap 200, visible fraction threshold 0.7 |
| DOTA scene merge | Per-class HBB NMS at 0.5; no multi-scale or flip testing |

YOLO11 best checkpoints are selected by validation AP with EMA; test metrics
are not used for selection. Archived historical YOLOv8m selection rules are
in `configs/selection/` and must not be blindly applied to other detectors.
RT-DETR DOTA selection logic is in `src/solver/det_solver.py`; merged-scene
HBB AP50 and patch COCO AP are separate fields and must not be mixed.

Historical learning schedules, optimizer configuration and ablations are
retained in the per-task configs and trainers. `EMA.warmups` is an EMA ramp,
not a learning-rate warmup. RT-DETR's milestone 1000 does not decay the
learning rate during an 80- or 100-epoch run.

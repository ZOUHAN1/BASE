# Single-Run Paper-Method Suite

## Scope

The full method follows the paper's BRR, BRSA and ARA computational topology.
Internal `AdaptiveResidualRelay` names are retained for checkpoint compatibility;
their score and gate MLPs both include SiLU. The receiving feature is corrected
before its refinement block. No extra P2 prediction head is added.

This suite changes the method integration, not every historical training
hyperparameter. It contains twelve formal jobs:

| Family | Dataset | Input | Arms | Seed | Epochs |
| --- | --- | --- | --- | --- | --- |
| YOLO11m | AI-TODv2 | 800 | native, full BASE | 0 | 100 |
| YOLOv8m | AI-TODv2 | 800 | native, full BASE | 0 | 100 |
| RT-DETR-R50-vd | AI-TODv2 | 800 | native, full BASE | 0 | 100 |
| YOLO11m | VisDrone | 640 | native, full BASE | 0 | 100 |
| YOLOv8m | VisDrone | 640 | native, full BASE | 0 | 100 |
| RT-DETR-R50-vd | VisDrone | 640 | native, full BASE | 0 | 100 |

Each arm uses two GPUs with physical batch 8 per rank and global batch 16.
Native/full arms run concurrently on GPU pairs 0,1 and 2,3. The next pair
starts after both current arms finish. There is no automatic training retry
or seed sweep. Download retries do not start another training run.

YOLO uses the retained AdamW, three-epoch warmup and cosine recipe. RT-DETR
retains AdamW and its existing milestone-1000 schedule without LR warmup;
the 100-epoch run therefore has no scheduled LR drop. These settings must
not be described as reproducing the different schedule in the manuscript.

## Model and loss contracts

- `yolov8/training.py` wraps the native detector or full asymmetric neck and
  adds `0.1 * occupancy_loss` with the same local-batch scaling as YOLO11m.
- `yolov8/full_at/asymmetric_yolov8.py` returns optional auxiliary tensors
  during training without changing the inference topology or parameter count.
- `rtdetr/dota_hbb/` contains the full shared RT-DETR method, including shallow
  C2 exposure, BRR, BRSA, independent ARA branches and occupancy loss.
- RT-DETR native validation selects AI-TODv2 AP with maxDet 1500, or VisDrone
  AP with its official ignore handling. Its unchanged postprocessor returns
  300 query/class predictions. VisDrone training categories are remapped from
  1..10 to 0..9, then converted back to 1..10 only for official evaluation.
- Best checkpoints use validation AP and EMA. This suite does not use test
  data for selection and does not require test-dev images for completion.

Every family/dataset pair loads one shared COCO-transferred initialization.
New target-class prediction tensors are initialized once and shared by both
arms. No target-dataset accuracy checkpoint initializes these runs.

## Entry points

```bash
python scripts/prepare_once_suite.py --help
python scripts/preflight_once_suite.py --help
python scripts/verify_once_data.py --help
python scripts/run_once_suite.py --help
```

Preparation writes an external experiment directory containing the manifest,
configs and initialization files. `--skip-rtdetr-init` can defer its checkpoint
transfer to `init_rtdetr_once.py` while an official source checkpoint is being
downloaded. It must not be used to substitute random weights for a paired
pretrained detector.

The disposable preflight performs one synthetic forward/backward per arm and
checks finite gradients and a nonzero BRSA modulation gradient. It does not
perform an optimizer update or produce an accuracy checkpoint. Formal runs
load fresh paired initialization afterward. Actual image lists are separately
checked against train/validation annotations before a dataset is admitted.

`run_once_suite.py --suite /path/to/suite.json` runs the persistent queue and
writes `status.json`. A lock and `queue_started.json` prevent duplicate starts.
Missing future assets cause waiting rather than training on an incomplete
dataset. Failed formal jobs are reported, not silently restarted.

`--recover-failed-start` is an explicit operator recovery, not an automatic
retry. It retains completed jobs, refuses live previous PIDs or nonempty
partial-training output directories, and archives failed-start logs before
reloading the same paired initialization. An interrupted startup is recorded
as an attempt, not reported as another completed seed. Loader worker counts
may be lowered equally for both arms to fit the host shared-memory budget.

Optional `prepare_once_assets.py` restores missing official VisDrone training
JPEGs without replacing annotations, initializes RT-DETR and validates method
contracts. Its logs and all data stay outside the Git source release.

## Reporting

Report one-run metrics as one-run metrics, not a three-seed mean. Distinguish
formal losses from preflight losses and evaluator unit-test AP. The unit test
with a perfect synthetic detection is not a dataset accuracy result.

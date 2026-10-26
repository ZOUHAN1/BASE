# Release Preparation Notes

The external method abbreviation and repository name are BASE. The original
BRSA, BRR and ARR symbols and historical experiment IDs are retained; this
naming change does not alter the architecture or checkpoint state keys.

## Source groups

| Release directory | Local experiment identifier |
| --- | --- |
| `yolo11/aitodv2` | `brsa_asym_aitodv2_20260930` |
| `yolo11/visdrone` | `brsa_asym_visdrone_640_20261003` |
| `yolov8/model.py`, `brsa.py` | `yolov8m_speed_20261019` restored historical graph |
| `yolov8/full_at` | `20261019-cost-profile-T4DYIW/source/source_yolo` architecture port |
| `rtdetr/aitodv2` | `brsa_rtdetr_h20_20260928` |
| `rtdetr/dota_hbb` | `brsa_rtdetr_dota1_hbb_640_20261003` |

The local YOLO11 Ultralytics source reports commit
`595885a26af753d545b240520af8e0eb840f62e4`; the YOLOv8 snapshot reports
`6e43d1e1e5db72afbf686dee6745669bcb124b0a`. They include local experimental
changes, so these commit IDs alone are not the complete payload identity.
The per-file manifests identify the exported contents.

## Packaging edits

Model mathematics, model parameters and the original auxiliary loss reduction
were not changed during release preparation. Changes are limited to:

- Redirecting bootstrap imports to included third-party snapshots.
- Replacing machine data-root defaults with relative paths or environment variables.
- Removing historical GPU UUIDs from configs and querying current UUIDs in a new launcher.
- Making the restored YOLOv8 builder accept random initialization for weight-free checks.
- Retaining original DDP, shared initialization and checkpoint gates; the new launcher
  verifies paired smoke completion before creating the expected gate marker.
- Relocating RT-DETR paired launchers and using the active Python interpreter.
- Adding documentation, CPU smoke checks, configuration checks and release manifests.
- Excluding generated Cython output containing build-machine paths; `_mask.pyx`
  and native C/C++ sources remain available to rebuild the extension.
- Explicitly enabling C99 for the evaluator extension on non-Windows compilers
  so modern Cython output also builds with older GCC defaults.

No speed-oriented structural variants were selected as the default full arm.
Optional inference optimizations remain in the exported source. No training
or speed results were fabricated or regenerated for this release.

## Subsequent single-run integration

The [single-run suite](SINGLE_RUN_SUITE.md) adds the full YOLOv8 training
wrapper, shared full RT-DETR integration for AI-TODv2 and VisDrone, and their
native validation adapters. Auxiliary YOLOv8 outputs expose existing logits
without changing the full inference graph. Native and full arms use shared
fresh COCO-transferred initialization and validation-only model selection.
These are new runs, not reconstructions of missing historical training logs.
External data, initialization files and running-job logs remain excluded.

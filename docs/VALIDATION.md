# Release Validation

Validation date: 2026-10-26. These are source packaging checks, not fresh
training, AP evaluation or GPU latency measurements.

## CPU forward checks

The smoke script constructs models without checkpoint loading or downloads,
uses FP32 evaluation with gradients disabled and synthetic 128 x 128 input,
and checks that all returned tensors are finite.

| Model / arm | Parameters |
| --- | ---: |
| YOLOv8m native, 8 classes | 25,860,952 |
| YOLOv8m historical BRSA, 8 classes | 25,885,820 |
| YOLOv8m full BASE port, 8 classes | 24,515,634 |
| YOLO11m native, AI-TODv2 8 classes | 20,059,176 |
| YOLO11m full BASE, AI-TODv2 8 classes | 17,435,522 |
| YOLO11m native, VisDrone 10 classes | 20,060,718 |
| YOLO11m full BASE, VisDrone 10 classes | 17,437,064 |
| RT-DETR native, DOTA 15 classes | 42,757,797 |
| RT-DETR full BASE, DOTA 15 classes | 47,410,175 |
| RT-DETR native, AI-TODv2 8 classes | 42,743,412 |
| RT-DETR original BRSA, AI-TODv2 8 classes | 42,826,455 |

The eleven configurations above passed. At 128 input size, YOLO outputs have
336 candidate locations; DOTA RT-DETR outputs have 300 queries. Different
class counts change the prediction heads, so these counts must not be mixed
with counts from another dataset's head.

The machine-readable local report is written to the ignored
`verification-local/smoke.json` directory.

## Focused module tests

The AI-TODv2 geometry/sampling, BRSA and ARR test files returned:

```text
17 passed, 10 skipped
```

The corresponding VisDrone geometry/sampling, BRSA and ARR tests also passed:

```text
16 passed
```

CUDA/Triton-specific tests were skipped with `CUDA_VISIBLE_DEVICES` empty.
One third-party setuptools metadata deprecation warning was emitted.

The included AI-TOD evaluator extension also built successfully into a Linux
CPython 3.8 wheel in an isolated build environment with NumPy 1.24.4 and
Cython 3.2.9. The build uses an explicit C99 flag on non-Windows platforms;
this changes compiler compatibility, not evaluation formulas.

## Limits

- No dataset or accuracy checkpoint was needed for these checks.
- No full training, DDP training, GPU fused-kernel validation or speed benchmark
  was repeated during packaging.
- A passing forward does not establish reproduction of historical AP.
- The YOLOv8m full port is not interchangeable with its historical BRSA-only model.
- GitHub Actions passed in a clean Python 3.10 CPU environment after declaring
  the RT-DETR `transformers` dependency. Local validation uses Python 3.8.

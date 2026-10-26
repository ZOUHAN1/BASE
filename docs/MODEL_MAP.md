# Model and Module Map

## YOLOv8m

| Purpose | File / symbol |
| --- | --- |
| Native and historical BRSA model | `yolov8/model.py::build_yolov8m` |
| Original BRSA inference adapter | `yolov8/brsa.py::BRSA` |
| Full asymmetric neck port | `yolov8/full_at/asymmetric_yolov8.py::AsymmetricYOLOv8` |
| BRR / ARR / BRSA / geometry | Sibling files in `yolov8/full_at/` |
| Native model and loss implementation | `third_party/yolov8_ultralytics/ultralytics/` |

The historical adapter reads stride-4 features from layer 2, enhances the
stride-8 semantic feature and reconnects it to the native detection head.
It does not add BRR or ARR. The full port separately replaces the neck using
the same BRR/BRSA/ARR pattern as YOLO11m, with YOLOv8 channel dimensions.

## YOLO11m

Use `yolo11/aitodv2/` or `yolo11/visdrone/` as the task directory.

| Purpose | Relative file / symbol |
| --- | --- |
| Native detector factory | `models/asymmetric_yolo11.py::native_model` |
| Baseline/full selection | `models/asymmetric_yolo11.py::PairedDetectionModel` |
| Background-referenced reading | `models/background_referenced_reading.py::BackgroundReferencedReading` |
| Frequency-domain enhancement | `models/brsa_original.py::OriginalBRSA` |
| Residual propagation | `models/adaptive_residual_relay.py::AdaptiveResidualRelay` |
| Sampling and validity | `models/geometry.py::sample_points`, `valid_cells` |
| Auxiliary occupancy loss | `OriginalBRSA.occupancy_loss` |
| Detection + auxiliary loss integration | `PairedDetectionModel.loss` |
| Training / checkpoint / EMA | `train_pair.py::PairTrainer` |

The full arm constructs M4 and T3 using BRR, enhances T3 with BRSA, computes
the residual E3-T3 and relays it to the coarser levels through ARR. Detect
consumes E3, P4 and P5. The original native detection loss is preserved;
the foreground occupancy term is added with coefficient 0.1.

The AI-TODv2 directory retains implementation optimizations from later speed
work. The VisDrone directory preserves its own experimental source snapshot.
Do not exchange checkpoints between these implementations without a state-key
and numerical-equivalence check.

## RT-DETR-R50-vd

| Purpose | File |
| --- | --- |
| Detector integration | `rtdetr/*/src/zoo/rtdetr/rtdetr.py` |
| Native and modified encoder | `rtdetr/*/src/zoo/rtdetr/hybrid_encoder.py` |
| Full BRR/BRSA/ARR classes and geometry | `rtdetr/dota_hbb/src/zoo/rtdetr/asymmetric_neck.py` |
| Auxiliary loss consumption | `rtdetr/*/src/zoo/rtdetr/rtdetr_criterion.py` |
| Shallow feature exposure | `rtdetr/*/src/nn/backbone/presnet.py` |
| DOTA patch-to-scene evaluation | `rtdetr/dota_hbb/src/data/dota_hbb_eval.py` |
| Train / validation CLI | `rtdetr/*/tools/train.py` |

The DOTA full configuration enables `PResNet.expose_c2` and
`HybridEncoder.use_asymmetric_neck`. Its outputs retain three 256-channel
feature levels consumed by the original RT-DETR decoder. The AI-TODv2
experiment is preserved separately because its original BRSA integration is
not the same full asymmetric neck.

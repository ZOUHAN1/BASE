import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY / 'third_party/yolo11_ultralytics'))
sys.path.insert(0, str(REPOSITORY / 'third_party/cocoapi_aitod'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime/ultralytics_settings'))
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '4')


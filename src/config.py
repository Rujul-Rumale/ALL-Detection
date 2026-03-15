"""Authoritative preprocessing and model constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_RES = 320
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
NUM_CLASSES = 2
CLASS_NAMES = ("ALL", "HEM")   # 0 = ALL (blast), 1 = HEM (healthy)

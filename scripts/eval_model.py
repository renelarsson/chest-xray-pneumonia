from __future__ import annotations

"""Evaluate a saved pneumonia model on the held-out test set.

This script is lightweight and does **not** retrain the model. It:

- Downloads/locates the Chest X-Ray Pneumonia dataset via KaggleHub
- Builds only the test (and val) tf.data pipelines
- Loads an existing Keras model (by default artifacts/resnet50v2_ft.keras)
- Prints loss/accuracy/AUC on the test set

Usage (from repo root):

  uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -e ".[notebooks]"

  # If your model is in notebooks/artifacts/resnet50v2_ft.keras:
  python scripts/eval_model.py \
    --model notebooks/artifacts/resnet50v2_ft.keras

  # If you have artifacts/resnet50v2_ft.keras at repo root:
  python scripts/eval_model.py
"""

import argparse
from pathlib import Path

import tensorflow as tf

from service.env import load_repo_dotenv


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate a saved Keras pneumonia model on the test split.",
  )
  parser.add_argument(
    "--dataset",
    default="paultimothymooney/chest-xray-pneumonia",
    help="KaggleHub dataset slug (default: %(default)s)",
  )
  parser.add_argument(
    "--env-file",
    default=".env.local",
    help=(
      "Env filename to load from repo root/parents before calling "
      "KaggleHub (default: %(default)s)"
    ),
  )
  parser.add_argument(
    "--model",
    default="artifacts/resnet50v2_ft.keras",
    help="Path to the saved Keras model to evaluate (default: %(default)s)",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
    help="Batch size for evaluation (default: %(default)s)",
  )
  return parser.parse_args()


def locate_dataset(dataset_slug: str) -> Path:
  import kagglehub  # lazy import

  path = kagglehub.dataset_download(dataset_slug)
  base = Path(path)
  candidates = [p for p in base.rglob("chest_xray") if p.is_dir()]
  return candidates[0] if candidates else base


def build_test_ds(chest_xray_root: Path, batch_size: int):
  test_dir = chest_xray_root / "test"

  test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(test_dir),
    image_size=IMG_SIZE,
    batch_size=batch_size,
    label_mode="binary",
  )

  class_names = getattr(test_ds, "class_names", ["NORMAL", "PNEUMONIA"])
  print("class_names:", class_names)

  def prep(x, y):
    x = tf.cast(x, tf.float32)
    return x, y

  test_ds = test_ds.map(prep).prefetch(tf.data.AUTOTUNE)
  return test_ds, class_names


def main() -> None:
  args = parse_args()

  dotenv_loaded = load_repo_dotenv(args.env_file)
  print({"dotenv_loaded": dotenv_loaded})

  chest_xray_root = locate_dataset(args.dataset)
  print("Using chest_xray_root:", chest_xray_root)

  test_ds, class_names = build_test_ds(
    chest_xray_root, batch_size=args.batch_size
  )

  model_path = Path(args.model)
  if not model_path.exists():
    raise SystemExit(f"Model file not found: {model_path}")

  print(f"Loading model from {model_path}...")
  model = tf.keras.models.load_model(model_path.as_posix())

  print("Evaluating on test set...")
  results = model.evaluate(test_ds, verbose=1, return_dict=True)
  print("Test metrics:")
  for k, v in results.items():
    print(f"  {k}: {v:.4f}")


if __name__ == "__main__":  
  main()

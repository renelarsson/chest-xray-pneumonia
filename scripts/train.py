from __future__ import annotations

"""Train the final ResNet50V2 model for chest X-ray pneumonia.

This script mirrors the notebook's best-performing configuration:

- Downloads the Chest X-Ray Pneumonia dataset via KaggleHub
- Builds tf.data pipelines for train/val/test
- Trains a ResNet50V2-based classifier (frozen head, then fine-tunes
  the top layers)
- Saves the fine-tuned model to artifacts/resnet50v2_ft.keras

Usage (from repo root, after creating and activating .venv):

    uv pip install -e ".[notebooks]"
    python train.py --env-file .env.local \
        --dataset paultimothymooney/chest-xray-pneumonia \
        --output artifacts/resnet50v2_ft.keras

Make sure KAGGLE_API_TOKEN is set in your env or .env.local.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from service.env import load_repo_dotenv


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ResNet50V2 pneumonia classifier on the Chest X-Ray "
            "Pneumonia dataset and save the fine-tuned model."
        ),
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
        "--output",
        default="artifacts/resnet50v2_ft.keras",
        help="Path to save the fine-tuned model (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=8,
        help="Epochs for training the frozen ResNet head (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-ft",
        type=int,
        default=5,
        help="Epochs for fine-tuning the top ResNet layers (default: %(default)s)",
    )
    return parser.parse_args()


def locate_dataset(dataset_slug: str) -> Path:
    """Download (or reuse cached) dataset via KaggleHub and return chest_xray root.

    The data is stored in the KaggleHub cache, not inside this repo.
    """

    import kagglehub  # imported lazily so service runtime doesn't need it

    path = kagglehub.dataset_download(dataset_slug)
    base = Path(path)
    candidates = [p for p in base.rglob("chest_xray") if p.is_dir()]
    return candidates[0] if candidates else base


def build_datasets(chest_xray_root: Path, batch_size: int):
    train_dir = chest_xray_root / "train"
    val_dir = chest_xray_root / "val"
    test_dir = chest_xray_root / "test"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(train_dir),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(val_dir),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(test_dir),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
    )

    class_names = getattr(train_ds, "class_names", ["NORMAL", "PNEUMONIA"])
    print("class_names:", class_names)

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        return x, y

    train_ds = train_ds.map(prep).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(prep).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_resnet_frozen(drop: float = 0.2) -> tf.keras.Model:
    """Create a ResNet50V2 model with augmentation and a frozen backbone."""

    base = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )

    base = models.Model(inputs=base.input, outputs=base.output, name="resnet_base")
    base.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    x_in = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(x_in)
    x = res_preprocess(x)
    x = base(x, training=False)
    x = layers.Dropout(drop)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(x_in, out)


def make_callbacks(monitor: str = "val_auc"):
    return [
        EarlyStopping(monitor=monitor, patience=3, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]


def train_and_finetune(
    train_ds,
    val_ds,
    epochs_head: int,
    epochs_ft: int,
) -> tf.keras.Model:
    """Train frozen ResNet head, then fine-tune the top block."""

    model = build_resnet_frozen(drop=0.2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    print("Training ResNet50V2 head (frozen backbone)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=make_callbacks("val_auc"),
        verbose=1,
    )

    # Fine-tune: unfreeze top N layers of the backbone (excluding BatchNorm)
    resnet_base = model.get_layer("resnet_base")
    resnet_base.trainable = True
    N = 50
    for layer in resnet_base.layers[:-N]:
        layer.trainable = False
    for layer in resnet_base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    print("Fine-tuning top ResNet50V2 layers...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_ft,
        callbacks=make_callbacks("val_auc"),
        verbose=1,
    )

    return model


def main() -> None:
    args = parse_args()

    dotenv_loaded = load_repo_dotenv(args.env_file)
    print({"dotenv_loaded": dotenv_loaded, "KAGGLE_API_TOKEN_set": bool(os.environ.get("KAGGLE_API_TOKEN"))})

    chest_xray_root = locate_dataset(args.dataset)
    print("Using chest_xray_root:", chest_xray_root)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        chest_xray_root, batch_size=args.batch_size
    )

    model = train_and_finetune(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs_head=args.epochs_head,
        epochs_ft=args.epochs_ft,
    )

    # Evaluate on the held-out test set
    print("Evaluating on test set...")
    eval_results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print("Test metrics:", eval_results)

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path.as_posix())
    print(f"Saved fine-tuned model to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

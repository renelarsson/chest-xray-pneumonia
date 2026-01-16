from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tensorflow as tf


#MODEL_PATH = Path("notebooks/artifacts/resnet50v2_ft.keras")
MODEL_PATH = Path(__file__).parent.parent / "notebooks/artifacts/resnet50v2_ft.keras"

app = FastAPI(title="chest-xray-pneumonia")

_model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    """Load the trained Keras model lazily on first request."""

    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH.as_posix())
    return _model


def preprocess(image: Image.Image) -> np.ndarray:
    """Prepare a single RGB image for the model.

    The training pipeline feeds float32 0..255 tensors into a model that
    handles ResNet preprocessing internally, so we don't normalize here.
    """

    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)  # shape (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)
    return arr


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    """Run pneumonia prediction for an uploaded chest X-ray image.

    Returns class probabilities for pneumonia vs normal.
    """

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:  
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    x = preprocess(img)

    try:
        model = load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Sigmoid output approximates P(pneumonia); labels in training are 0=NORMAL, 1=PNEUMONIA
    proba = float(model.predict(x, verbose=0)[0, 0])
    proba = max(0.0, min(1.0, proba))  # clamp for safety

    scores = {
        "pneumonia": proba,
        "normal": 1.0 - proba,
    }

    return {
        "scores": scores,
        "input_shape": list(x.shape),
        "model_path": MODEL_PATH.as_posix(),
    }

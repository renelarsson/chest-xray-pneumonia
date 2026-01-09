# Chest X-Ray Pneumonia

This project detects pneumonia in chest X-ray images using TensorFlow/Keras and serves predictions via a FastAPI web service.

## Quick Start

### Local (without Docker)
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install service/runtime deps
uv pip install -r requirements.txt

# If you want to run the notebooks too (EDA + training)
uv pip install -e ".[notebooks]"

# Run the FastAPI service locally (serves the trained model)
python -m uvicorn service.predict:app --host 0.0.0.0 --port 9696
```

### Notebooks (Codespaces)
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[notebooks]"

# Optional extras (only if you use sklearn/scipy-based metrics)
uv pip install -e ".[metrics]"

# Make the environment show up as a selectable kernel
.venv/bin/python -m ipykernel install --user --name xray-venv --display-name "Python (xray-venv)"
```

### Docker
```bash
docker build -t chest-xray-pneumonia .
docker run -it -p 9696:9696 chest-xray-pneumonia
```

### Training script (train.py)

To (re)train the final ResNet50V2 model used by the FastAPI service:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e ".[notebooks]"

python train.py --env-file .env.local \
	--dataset paultimothymooney/chest-xray-pneumonia \
	--output artifacts/resnet50v2_ft.keras
```

The FastAPI service in service/predict.py expects the trained model at
artifacts/resnet50v2_ft.keras (the same path used by the notebook).

### Evaluation script (scripts/eval_model.py)

If you already have a trained model (for example exported from Colab to
notebooks/artifacts/resnet50v2_ft.keras), you can evaluate it on the
held-out test set **without retraining**:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e ".[notebooks]"

python scripts/eval_model.py \
  --model notebooks/artifacts/resnet50v2_ft.keras
```

This script downloads/locates the Kaggle dataset via KaggleHub, builds the
test pipeline, loads the given Keras model, and prints loss/accuracy/AUC.

## KaggleHub token (.env)

For local development (especially if you run dataset download code from notebooks or local `.py` scripts), put your KaggleHub token in a `.env` file at the repo root.

```bash
cp .env.example .env
# edit .env and set KAGGLE_API_TOKEN=...
```

If you run any training/data scripts inside Docker locally, pass the same file through:

```bash
docker run --env-file .env -it <your-image>
```

For local scripts, a minimal example is included:

```bash
uv pip install -e ".[notebooks]"
python scripts/download_dataset.py
```

KaggleHub downloads the Chest X-Ray Pneumonia dataset into the user cache
(not under this repo). By default the cache path looks like:

- `/home/codespace/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray`

Both `scripts/download_dataset.py` and `notebooks/notebook2.ipynb` call
`kagglehub.dataset_download(...)` and then resolve `train/`, `val/`, and `test`
subfolders from that cached location.

## API
- GET /health → `{ "status": "ok" }`
- POST /predict → multipart `file` image; returns scores and input shape

## API Usage Example

Once the FastAPI service is running, you can test the `/predict` endpoint using `curl`:

```bash
curl -X POST -F "file=@path/to/sample_image.jpg" http://127.0.0.1:9696/predict
```

Example Response:
```json
{
  "label": "Pneumonia",
  "confidence": 0.95
}
```

## Tests Performed

### Local FastAPI Service
- **Health Check**:
  ```bash
  curl http://localhost:9696/health
  ```
  Response:
  ```json
  {"status":"ok"}
  ```

- **Prediction Endpoint**:
  ```bash
  curl -X POST -F "file=@data/normal.jpeg" http://localhost:9696/predict
  ```
  Response:
  ```json
  {
    "scores": {
      "pneumonia": 0.47770124673843384,
      "normal": 0.5222987532615662
    },
    "input_shape": [1, 224, 224, 3],
    "model_path": "/workspaces/ppe-compliance-detection/notebooks/artifacts/resnet50v2_ft.keras"
  }
  ```

### Dockerized Service
- **Health Check**:
  ```bash
  curl http://localhost:9696/health
  ```
  Response:
  ```json
  {"status":"ok"}
  ```

- **Prediction Endpoint**:
  ```bash
  curl -X POST -F "file=@data/pneumonia.jpeg" http://localhost:9696/predict
  ```
  Response:
  ```json
  {
    "scores": {
      "pneumonia": 0.47770124673843384,
      "normal": 0.5222987532615662
    },
    "input_shape": [1, 224, 224, 3],
    "model_path": "/app/notebooks/artifacts/resnet50v2_ft.keras"
  }
  ```

## Next Steps
- Use notebooks/notebook.ipynb for EDA and model comparison
- Train the final ResNet50V2 model via train.py or the notebook
- Optionally export to ONNX and use CPU-optimized runtime
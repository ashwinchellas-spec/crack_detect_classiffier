"""
FastAPI service for the crack/no_crack classifier.

Endpoints:
    POST /predict   -- upload an image, get back {predicted_class, confidence}
    GET  /health    -- liveness check
    GET  /logs      -- recent predictions + per-class summary (from SQLite)

Run locally:
    uvicorn api.main:app --reload --port 8000

Then test with:
    curl -X POST "http://localhost:8000/predict" -F "file=@some_image.png"
"""

import io
import json
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from api.db import init_db, log_prediction, get_recent_predictions

MODEL_PATH = "models/crack_classifier.keras"
CONFIG_PATH = "models/config.json"

app = FastAPI(title="Crack Defect Classifier API", version="1.0")

_model = None
_config = None


@app.on_event("startup")
def load_model():
    global _model, _config
    init_db()
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        _config = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    img_size = _config["img_size"]
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)[np.newaxis, ...]  # (1, H, W, 3)

    prob = float(_model.predict(arr, verbose=0)[0][0])

    # class index 0/1 mapping was saved during training (alphabetical by
    # tf.keras.utils.image_dataset_from_directory): ['crack', 'no_crack']
    class_names = _config["class_names"]
    predicted_idx = int(prob > 0.5)
    predicted_class = class_names[predicted_idx]
    confidence = prob if predicted_idx == 1 else 1 - prob

    log_prediction(file.filename, predicted_class, confidence)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "raw_score": round(prob, 4),
    }


@app.get("/logs")
def logs(limit: int = 20):
    return get_recent_predictions(limit=limit)

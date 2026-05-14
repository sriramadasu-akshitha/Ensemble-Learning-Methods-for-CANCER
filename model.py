"""
Connects `index.html` to a Python backend.

Run:
  python model.py

Then open:
  http://127.0.0.1:5000/

This file exposes:
  - GET  /        -> serves frontend.html
  - POST /predict -> accepts an image upload and returns JSON prediction

Notes:
  - If you have trained/saved models, place them under ./models/ and the server
    will try to load them. Otherwise it falls back to a lightweight heuristic
    so the frontend-backend connection still works end-to-end.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
IMG_SIZE: Tuple[int, int] = (128, 128)


def _load_keras_model(path: Path):
    import tensorflow as tf  # local import: keep startup fast if missing

    return tf.keras.models.load_model(path)


def _load_pickle(path: Path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


class EnsemblePredictor:
    """
    Tries to use saved models if present:
      - models/model1.keras  (MobileNet-based)
      - models/model2.keras  (ResNet-based)
      - models/rf.pkl
      - models/svm.pkl
      - models/feature_extractor.keras  (outputs feature vector)
    """

    def __init__(self) -> None:
        self.model1 = None
        self.model2 = None
        self.rf = None
        self.svm = None
        self.feature_model = None
        self.loaded_from_disk = False

        self._try_load()

    def _try_load(self) -> None:
        try:
            m1 = MODELS_DIR / "model1.keras"
            m2 = MODELS_DIR / "model2.keras"
            rf = MODELS_DIR / "rf.pkl"
            svm = MODELS_DIR / "svm.pkl"
            feat = MODELS_DIR / "feature_extractor.keras"

            if not (m1.exists() and m2.exists() and rf.exists() and svm.exists() and feat.exists()):
                return

            self.model1 = _load_keras_model(m1)
            self.model2 = _load_keras_model(m2)
            self.feature_model = _load_keras_model(feat)
            self.rf = _load_pickle(rf)
            self.svm = _load_pickle(svm)
            self.loaded_from_disk = True
        except Exception:
            # If anything fails, keep fallback mode.
            self.model1 = None
            self.model2 = None
            self.rf = None
            self.svm = None
            self.feature_model = None
            self.loaded_from_disk = False

    def _preprocess(self, file_bytes: bytes) -> np.ndarray:
        if Image is None:
            raise RuntimeError("Pillow is not installed. Install it: pip install pillow")

        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
        return arr

    def predict_proba(self, file_bytes: bytes) -> Tuple[float, Dict[str, Any]]:
        """
        Returns:
          (prob_disease, debug_info)
        """
        x = self._preprocess(file_bytes)

        if self.loaded_from_disk and self.model1 is not None and self.model2 is not None and self.rf is not None and self.svm is not None and self.feature_model is not None:
            p1 = float(self.model1.predict(x, verbose=0)[0][0])
            p2 = float(self.model2.predict(x, verbose=0)[0][0])
            feats = self.feature_model.predict(x, verbose=0)
            p3 = float(self.rf.predict_proba(feats)[0][1])
            p4 = float(self.svm.predict_proba(feats)[0][1])
            final = 0.5 * p1 + 0.4 * p2 + 0.05 * p3 + 0.05 * p4
            return float(final), {"mode": "ensemble", "p1": p1, "p2": p2, "p3": p3, "p4": p4}

        # Fallback (keeps the demo connected even without heavy model files)
        # Simple brightness-based heuristic: darker images -> higher "disease" probability.
        mean_intensity = float(x.mean())
        prob = float(np.clip(1.0 - mean_intensity, 0.0, 1.0))
        return prob, {"mode": "fallback", "mean_intensity": mean_intensity}


app = Flask(__name__)
predictor = EnsemblePredictor()


@app.get("/")
def index():
    # Serve the static frontend file from the project root
    return send_from_directory(str(APP_DIR), "index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400

    file = request.files["image"]
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    try:
        prob, debug = predictor.predict_proba(file_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    label = "Disease Detected" if prob >= 0.5 else "Normal"
    return jsonify(
        {
            "label": label,
            "probability": prob,
            "confidence_percent": round(prob * 100.0, 2),
            "debug": debug,
        }
    )


if __name__ == "__main__":
    # Use env vars if you want to expose on LAN:
    #   set FLASK_HOST=0.0.0.0
    #   set FLASK_PORT=5000
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
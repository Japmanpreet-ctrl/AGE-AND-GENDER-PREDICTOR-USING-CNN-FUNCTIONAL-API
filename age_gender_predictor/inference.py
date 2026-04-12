from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image
from mtcnn import MTCNN
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

from age_gender_predictor.config import (
    GENDER_UNCERTAINTY_HIGH,
    GENDER_UNCERTAINTY_LOW,
    IMAGE_SIZE,
    MODEL_PATH,
)


@lru_cache(maxsize=1)
def load_detector() -> MTCNN:
    return MTCNN()


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first or set AGE_GENDER_MODEL_PATH."
        )
    return keras.models.load_model(MODEL_PATH)


def detect_and_crop_face(image: Image.Image) -> Image.Image | None:
    image_array = np.array(image.convert("RGB"))
    detections = load_detector().detect_faces(image_array)
    if not detections:
        return None

    top_detection = max(detections, key=lambda item: item.get("confidence", 0.0))
    x, y, width, height = top_detection["box"]
    x = max(0, x)
    y = max(0, y)
    face = image_array[y : y + height, x : x + width]
    if face.size == 0:
        return None
    return Image.fromarray(face)


def preprocess_face(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.array(image).astype("float32")
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def format_gender(probability: float) -> str:
    if GENDER_UNCERTAINTY_LOW <= probability <= GENDER_UNCERTAINTY_HIGH:
        return "Uncertain"
    return "Female" if probability >= 0.5 else "Male"


def predict_image(image: Image.Image) -> dict:
    if image is None:
        return {"error": "Please upload an image."}

    face = detect_and_crop_face(image)
    if face is None:
        return {"error": "No face detected. Please upload a clear frontal face image."}

    model = load_model()
    batch = preprocess_face(face)
    age_output, gender_output = model.predict(batch, verbose=0)

    age = float(age_output[0][0])
    gender_probability = float(gender_output[0][0])
    return {
        "predicted_age": round(age, 1),
        "predicted_gender": format_gender(gender_probability),
        "gender_probability": round(gender_probability, 4),
        "model_path": str(MODEL_PATH),
    }

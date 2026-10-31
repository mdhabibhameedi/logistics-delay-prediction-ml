
import json
import os

import joblib
import pandas as pd


FEATURES = [
    "region",
    "distance_km",
    "traffic_level",
    "departure_hour",
    "planned_delivery_minutes",
    "temperature_c",
    "humidity_percent",
    "rainfall_mm",
]


def model_fn(model_dir):
    """Load the trained Scikit-learn pipeline."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    """Convert a JSON request into a DataFrame."""
    if request_content_type not in ("application/json", "application/json; charset=utf-8"):
        raise ValueError("Only application/json is supported")

    data = json.loads(request_body)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a JSON object or non-empty list of objects")

    frame = pd.DataFrame(data)
    missing = [column for column in FEATURES if column not in frame.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return frame[FEATURES]


def predict_fn(input_data, model):
    """Predict shipment delays in minutes."""
    return model.predict(input_data).tolist()


def output_fn(prediction, accept):
    """Return predictions as JSON."""
    if accept not in ("application/json", "*/*", "application/json; charset=utf-8"):
        raise ValueError("Only application/json responses are supported")

    return json.dumps({"predicted_delay_minutes": prediction})

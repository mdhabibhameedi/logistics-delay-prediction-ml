import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

CATEGORICAL = ["region", "traffic_level"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    args = parser.parse_args()

    train_dir = Path(
        os.environ.get("SM_CHANNEL_TRAIN", "data/processed")
    )
    model_dir = Path(
        os.environ.get("SM_MODEL_DIR", "deployment/local_model")
    )
    output_dir = Path(
        os.environ.get("SM_OUTPUT_DATA_DIR", "deployment/local_output")
    )

    dataset = pd.read_csv(train_dir / "ml_dataset.csv")

    X = dataset[FEATURES]
    y = dataset["delay_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL,
        ),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": round(float(mean_absolute_error(y_test, predictions)), 4),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(y_test, predictions))), 4
        ),
        "R2": round(float(r2_score(y_test, predictions)), 4),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("TRAINING_COMPLETED")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

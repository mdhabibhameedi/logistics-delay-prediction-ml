from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ml_dataset.csv"
MODELS = ROOT / "models"
PLOTS = ROOT / "docs" / "plots"

MODELS.mkdir(exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

target = "delay_minutes"

# Only use information available before delivery.
base_features = [
    "region",
    "distance_km",
    "traffic_level",
    "departure_hour",
    "planned_delivery_minutes"
]

weather_features = [
    "temperature_c",
    "humidity_percent",
    "rainfall_mm"
]

all_features = base_features + weather_features

assert not df[all_features + [target]].isna().any().any()

# Same train/test records for both models.
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42
)

def make_pipeline(features):
    categorical = [
        c for c in features
        if c in ["region", "traffic_level"]
    ]
    numerical = [
        c for c in features
        if c not in categorical
    ]

    preprocessing = ColumnTransformer([
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical
        )
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ])

def train_model(features):
    X = df[features]
    y = df[target]

    search = GridSearchCV(
        estimator=make_pipeline(features),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, None],
            "model__min_samples_leaf": [1, 3]
        },
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1
    )

    search.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    predictions = search.predict(X.iloc[test_idx])

    metrics = {
        "MAE": round(
            mean_absolute_error(
                y.iloc[test_idx], predictions
            ), 4
        ),
        "RMSE": round(
            np.sqrt(mean_squared_error(
                y.iloc[test_idx], predictions
            )), 4
        ),
        "R2": round(
            r2_score(
                y.iloc[test_idx], predictions
            ), 4
        )
    }

    return search, predictions, metrics

print("Training baseline model...")
baseline, baseline_predictions, baseline_metrics = (
    train_model(base_features)
)

print("Training weather-aware model...")
weather_model, predictions, weather_metrics = (
    train_model(all_features)
)

# Save the complete fitted preprocessing + model pipeline.
joblib.dump(
    weather_model.best_estimator_,
    MODELS / "delay_prediction_model.joblib"
)

results = {
    "dataset_records": len(df),
    "training_records": len(train_idx),
    "testing_records": len(test_idx),
    "features": all_features,
    "baseline_metrics": baseline_metrics,
    "weather_model_metrics": weather_metrics,
    "best_parameters": weather_model.best_params_,
    "synthetic_target": True
}

with open(
    MODELS / "evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

actual = df[target].iloc[test_idx].to_numpy()
residuals = actual - predictions

# Plot 1: Actual vs predicted
plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions, alpha=0.6)
limits = [
    min(actual.min(), predictions.min()),
    max(actual.max(), predictions.max())
]
plt.plot(limits, limits, linestyle="--")
plt.xlabel("Actual delay (minutes)")
plt.ylabel("Predicted delay (minutes)")
plt.title("Actual vs Predicted Delivery Delay")
plt.tight_layout()
plt.savefig(
    PLOTS / "actual_vs_predicted.png",
    dpi=160
)
plt.close()

# Plot 2: Prediction residuals
plt.figure(figsize=(7, 5))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted delay (minutes)")
plt.ylabel("Residual (minutes)")
plt.title("Prediction Residuals")
plt.tight_layout()
plt.savefig(
    PLOTS / "prediction_residuals.png",
    dpi=160
)
plt.close()

# Plot 3: Weather model vs baseline
labels = ["Without weather", "With weather"]
values = [
    baseline_metrics["MAE"],
    weather_metrics["MAE"]
]

plt.figure(figsize=(7, 5))
plt.bar(labels, values)
plt.ylabel("Mean Absolute Error (minutes)")
plt.title("Effect of Weather Features on MAE")
plt.tight_layout()
plt.savefig(
    PLOTS / "weather_model_comparison.png",
    dpi=160
)
plt.close()

print("\nSUCCESS: Model training completed")
print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Baseline metrics:", baseline_metrics)
print("Weather model metrics:", weather_metrics)
print("Best parameters:", weather_model.best_params_)
print("Model saved:", MODELS / "delay_prediction_model.joblib")
print("Evaluation plots saved:", PLOTS)

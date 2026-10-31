from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "ml_dataset.csv"

shipments = pd.read_csv(RAW / "shipments.csv")
weather = pd.read_csv(RAW / "weather.csv")

# Check source integrity
assert len(shipments) == 1000
assert shipments["shipment_id"].is_unique
assert not weather.duplicated(
    ["region", "shipment_date"]
).any()

# Integrate both data sources
df = shipments.merge(
    weather,
    on=["region", "shipment_date"],
    how="left",
    validate="many_to_one",
    indicator=True
)

assert (df["_merge"] == "both").all(), (
    "Some shipments have no matching weather"
)
df = df.drop(columns="_merge")

# Validate numerical inputs
numeric = [
    "distance_km", "departure_hour",
    "planned_delivery_minutes",
    "actual_delivery_minutes", "delay_minutes",
    "temperature_c", "humidity_percent",
    "rainfall_mm"
]

assert not df[numeric].isna().any().any()
assert np.isfinite(df[numeric].to_numpy()).all()
assert df["departure_hour"].between(0, 23).all()
assert df["humidity_percent"].between(0, 100).all()
assert (df["rainfall_mm"] >= 0).all()

# Feature engineering
df["weather_severity_index"] = (
    df["rainfall_mm"] * 2
    + df["humidity_percent"] / 20
    + abs(df["temperature_c"] - 25) / 10
)

df["distance_category"] = pd.cut(
    df["distance_km"],
    bins=[0, 50, 150, np.inf],
    labels=["Short", "Medium", "Long"]
)

df["traffic_encoded"] = df["traffic_level"].map({
    "Low": 0, "Moderate": 1, "High": 2
})

df["time_of_day"] = pd.cut(
    df["departure_hour"],
    bins=[-1, 5, 11, 17, 23],
    labels=[
        "Night", "Morning",
        "Afternoon", "Evening"
    ]
)

# Verify delay calculation, allowing rounding
difference = (
    df["actual_delivery_minutes"]
    - df["planned_delivery_minutes"]
    - df["delay_minutes"]
).abs()

assert (difference < 0.02).all()
assert df["traffic_encoded"].notna().all()
assert df["distance_category"].notna().all()
assert df["time_of_day"].notna().all()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("SUCCESS: Final dataset validated")
print("Shipment records:", len(df))
print("Total columns:", len(df.columns))
print("Missing values:", df.isna().sum().sum())
print("Duplicate shipment IDs:",
      df["shipment_id"].duplicated().sum())
print("Weather matches:", len(df))
print("Saved to:", OUTPUT)

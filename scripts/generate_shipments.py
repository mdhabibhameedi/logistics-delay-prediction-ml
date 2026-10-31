from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = RAW / "shipments.csv"

weather = pd.read_csv(RAW / "weather.csv")

required = [
    "region", "shipment_date", "temperature_c",
    "humidity_percent", "rainfall_mm"
]

if weather[required].isna().any().any():
    raise ValueError("Weather data contains missing values")

weather = weather.drop_duplicates(
    subset=["region", "shipment_date"]
)

rng = np.random.default_rng(42)
records = []

for i in range(1000):
    # Select a real region-date weather observation
    w = weather.iloc[int(rng.integers(len(weather)))]

    distance = round(float(rng.uniform(5, 250)), 2)

    traffic = rng.choice(
        ["Low", "Moderate", "High"],
        p=[0.30, 0.45, 0.25]
    )

    departure_hour = int(rng.integers(0, 24))

    planned = round(distance / 55 * 60 + 20, 2)

    traffic_delay = {
        "Low": 2,
        "Moderate": 12,
        "High": 30
    }[traffic]

    # Explicitly simulated weather effects
    rain_delay = float(w["rainfall_mm"]) * 0.8
    heat_delay = max(
        float(w["temperature_c"]) - 35, 0
    ) * 0.5

    rush_hour_delay = (
        8 if 7 <= departure_hour <= 9
        or 16 <= departure_hour <= 19 else 0
    )

    noise = float(rng.normal(5, 8))

    delay = round(max(
        0,
        traffic_delay + rain_delay
        + heat_delay + rush_hour_delay + noise
    ), 2)

    actual = round(planned + delay, 2)

    records.append({
        "shipment_id": f"SHP{i+1:04d}",
        "region": w["region"],
        "shipment_date": w["shipment_date"],
        "distance_km": distance,
        "traffic_level": traffic,
        "departure_hour": departure_hour,
        "planned_delivery_minutes": planned,
        "actual_delivery_minutes": actual,
        "delay_minutes": delay
    })

df = pd.DataFrame(records)

assert len(df) == 1000
assert df["shipment_id"].is_unique
assert (df["delay_minutes"] >= 0).all()

df.to_csv(OUTPUT, index=False)

print("SUCCESS: Improved shipment dataset created")
print("Records:", len(df))
print("Columns:", len(df.columns))
print("Saved to:", OUTPUT)
print(df.head())

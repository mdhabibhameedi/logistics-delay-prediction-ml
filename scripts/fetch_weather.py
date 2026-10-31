from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "weather.csv"

CITIES = {
    "Riyadh": (24.7136, 46.6753),
    "Jeddah": (21.5433, 39.1728),
    "Dammam": (26.4207, 50.0888),
    "Makkah": (21.3891, 39.8579),
}

URL = "https://archive-api.open-meteo.com/v1/archive"

records = []

for city, (latitude, longitude) in CITIES.items():
    print(f"Fetching weather for {city}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "daily": (
            "temperature_2m_mean,"
            "relative_humidity_2m_mean,"
            "rain_sum"
        ),
        "timezone": "Asia/Riyadh"
    }

    response = requests.get(
        URL, params=params, timeout=60
    )
    response.raise_for_status()

    daily = response.json()["daily"]

    for i, date in enumerate(daily["time"]):
        records.append({
            "region": city,
            "shipment_date": date,
            "temperature_c": daily[
                "temperature_2m_mean"
            ][i],
            "humidity_percent": daily[
                "relative_humidity_2m_mean"
            ][i],
            "rainfall_mm": daily["rain_sum"][i]
        })

df = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nSUCCESS: Weather dataset created")
print("Total records:", len(df))
print("Saved to:", OUTPUT)
print(df.head())

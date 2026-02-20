import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

def _weather_archive(lat, lon, days=120):
    end_date = (datetime.utcnow().date() - timedelta(days=1))  # yesterday
    start_date = end_date - timedelta(days=days)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        "&timezone=UTC"
    )
    w = requests.get(url, timeout=30).json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(w["hourly"]["time"], utc=True),
        "temp": w["hourly"]["temperature_2m"],
        "humidity": w["hourly"]["relative_humidity_2m"],
        "wind": w["hourly"]["wind_speed_10m"],
        "pressure": w["hourly"]["pressure_msl"],
    })
    return df

def _weather_forecast(lat, lon, forecast_days=3):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        f"&forecast_days={forecast_days}&timezone=UTC"
    )
    w = requests.get(url, timeout=30).json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(w["hourly"]["time"], utc=True),
        "temp": w["hourly"]["temperature_2m"],
        "humidity": w["hourly"]["relative_humidity_2m"],
        "wind": w["hourly"]["wind_speed_10m"],
        "pressure": w["hourly"]["pressure_msl"],
    })
    return df

def get_weather_live(lat, lon, days_back=120, forecast_days=3):
    df_hist = _weather_archive(lat, lon, days=days_back)
    df_fc   = _weather_forecast(lat, lon, forecast_days=forecast_days)

    df = pd.concat([df_hist, df_fc], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["ts_key"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df

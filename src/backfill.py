"""
Backfill script to populate the engineered feature group with historical data.
Run this once to fill the feature store with past data.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]  # goes up from src/data/ to project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.config import LAT, LON, OPENWEATHER_KEY
from src.hopsworks_client import get_project_and_fs
from src.fetch_weather import get_weather_live
from src.fetch_pollution import get_pollution_live
from src.features import build_hourly_join, resample_df, add_lag_roll
from src.store_features import get_or_create_engineered_fg

def add_nonlinear_features(df):
    """Add the extra features used in the Streamlit app."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["wind_sq"] = df["wind"] ** 2
    return df

def backfill_date_range(start_date, end_date, engine_fg):
    """
    Fetch data for the given date range, compute features, and insert into Hopsworks.
    start_date and end_date must be timezone-aware (UTC).
    engine_fg is the pre‑acquired feature group object.
    """
    # Ensure dates are timezone-aware (should already be)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    print(f"Processing {start_date.date()} to {end_date.date()}...")

    days = (end_date - start_date).days
    if days <= 0:
        print("  → Invalid date range (zero or negative days). Skipping.")
        return

    # Fetch raw data; these functions return UTC-aware timestamps
    df_weather = get_weather_live(lat=LAT, lon=LON, days_back=days, forecast_days=0)
    df_poll = get_pollution_live(lat=LAT, lon=LON, api_key=OPENWEATHER_KEY, days_back=days)

    # Filter to the exact range
    mask = (df_weather["timestamp"] >= start_date) & (df_weather["timestamp"] < end_date)
    df_weather = df_weather.loc[mask].copy()
    mask = (df_poll["timestamp"] >= start_date) & (df_poll["timestamp"] < end_date)
    df_poll = df_poll.loc[mask].copy()

    if df_weather.empty or df_poll.empty:
        print(f"  → No weather or pollution data for this period. Skipping.")
        return

    print(f"  → Weather rows: {len(df_weather)}, Pollution rows: {len(df_poll)}")

    # Join and engineer features
    df_hourly = build_hourly_join(df_weather, df_poll)
    df_6h = resample_df(df_hourly, freq="6H")
    df_feat = add_lag_roll(df_6h)
    df_feat = add_nonlinear_features(df_feat)

    df_feat = df_feat.reset_index()
    df_feat["ts_key"] = df_feat["timestamp"].astype("int64") // 10**9

    # Upsert into Hopsworks (operation="upsert" updates existing rows)
    engine_fg.insert(df_feat, operation="upsert", write_options={"wait_for_job": False})
    print(f"  → Inserted {len(df_feat)} rows.")

if __name__ == "__main__":
    # Connect to Hopsworks and get the feature group ONCE
    project, fs = get_project_and_fs()
    engine_fg = get_or_create_engineered_fg(fs)

    # Define backfill range (last 120 days)
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=120)
    chunk = timedelta(days=30)

    print(f"Starting backfill from {start_date} to {end_date}...")
    current = start_date
    while current < end_date:
        next_date = min(current + chunk, end_date)
        backfill_date_range(current, next_date, engine_fg)
        current = next_date

    print("Backfill complete.")
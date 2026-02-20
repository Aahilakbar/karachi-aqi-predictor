"""
Hourly feature pipeline: fetch latest data, compute features, upsert to Hopsworks.
Run this script every hour (e.g., via GitHub Actions or cron).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

def run_feature_pipeline(days_back=7):
    """
    Fetch the last `days_back` days of data, compute features,
    and upsert into the engineered feature group.
    In production you would store the last run timestamp and fetch only new data.
    """
    # 1. Connect to Hopsworks
    project, fs = get_project_and_fs()

    # 2. Get or create the engineered feature group
    engine_fg = get_or_create_engineered_fg(fs)

    # 3. Fetch raw data for the last N days
    #    (forecast_days=0 because we only want historical for this pipeline)
    df_weather = get_weather_live(lat=LAT, lon=LON, days_back=days_back, forecast_days=0)
    df_poll = get_pollution_live(lat=LAT, lon=LON, api_key=OPENWEATHER_KEY, days_back=days_back)

    # 4. Join and engineer features (using your existing functions)
    df_hourly = build_hourly_join(df_weather, df_poll)
    df_6h = resample_df(df_hourly, freq="6H")
    df_feat = add_lag_roll(df_6h)
    df_feat = add_nonlinear_features(df_feat)

    # 5. Reset index to make timestamp a column and create ts_key
    df_feat = df_feat.reset_index()
    df_feat["ts_key"] = df_feat["timestamp"].astype("int64") // 10**9

    # 6. Upsert into Hopsworks
    engine_fg.insert(df_feat, operation="upsert", write_options={"wait_for_job": False})
    print(f"Inserted {len(df_feat)} rows into aqi_features_6h")

if __name__ == "__main__":
    run_feature_pipeline(days_back=7)   # adjust as needed
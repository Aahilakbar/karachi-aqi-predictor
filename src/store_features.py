import hopsworks
import pandas as pd
from src.config import HOPSWORKS_PROJECT, HOPSWORKS_API_KEY, HOPSWORKS_HOST, HOPSWORKS_PORT

def get_project_and_fs():
    """Connect to Hopsworks and return (project, feature_store)."""
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
    )
    fs = project.get_feature_store()
    return project, fs

def store_online(fs, df_weather, df_pollution):
    """
    Insert raw weather and pollution data into their respective feature groups.
    (Already existing, kept for backward compatibility.)
    """
    weather_fg = fs.get_or_create_feature_group(
        name="karachi_weather_features_online",
        version=1,
        primary_key=["ts_key"],
        online_enabled=True,
        description="Weather features keyed by epoch seconds"
    )
    pollution_fg = fs.get_or_create_feature_group(
        name="karachi_pollution_aqi_online",
        version=1,
        primary_key=["ts_key"],
        online_enabled=True,
        description="Pollution + AQI keyed by epoch seconds"
    )
    weather_fg.insert(df_weather, operation="upsert", write_options={"wait_for_job": False})
    pollution_fg.insert(df_pollution, operation="upsert", write_options={"wait_for_job": False})
    return weather_fg, pollution_fg

def get_or_create_engineered_fg(fs):
    """
    Get or create the feature group for 6‑hourly engineered features.
    This group contains all features used by the ML models.
    """
    fg = fs.get_or_create_feature_group(
        name="aqi_features_6h",
        version=1,
        primary_key=["ts_key"],
        event_time="timestamp",
        online_enabled=True,
        description="6‑hourly engineered features for AQI prediction",
        # Optional: add partition key for better performance
        # partition_key=["year", "month"]
    )
    return fg
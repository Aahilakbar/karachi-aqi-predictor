import pandas as pd

def build_hourly_join(w_on, p_on):
    df = w_on.merge(p_on, on="ts_key", how="inner", suffixes=("_w", "_p"))

    # pick the timestamp column
    if "timestamp_w" in df.columns:
        ts_col = "timestamp_w"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        ts_col = "timestamp_p"

    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    keep = [
        "temp","humidity","wind","pressure","hour","dow","month",
        "pm2_5","pm10","no2","so2","co","o3","aqi","ts_key"
    ]
    df = df[[c for c in keep if c in df.columns]].dropna()
    return df

def resample_df(df, freq="6H", aqi_mode="mean"):
    # Resample numeric columns
    out = df.resample(freq).mean(numeric_only=True)

    # If you want AQI as last value instead of mean
    if aqi_mode == "last":
        out["aqi"] = df["aqi"].resample(freq).last()

    # time features again (because resample changes index)
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month

    return out.dropna()

def _infer_step_hours(idx: pd.DatetimeIndex) -> int:
    """
    Infer step size in hours from the index (robust even when idx.freq is None).
    """
    if len(idx) < 2:
        return 1

    # Use median diff between consecutive timestamps
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return 1

    step_seconds = diffs.median().total_seconds()
    if step_seconds <= 0:
        return 1

    step_hours = max(1, int(round(step_seconds / 3600)))
    return step_hours

def add_lag_roll(df, freq_hours=None):
    df = df.copy()

    # detect hours between points
    if freq_hours is None:
        diffs = df.index.to_series().diff().dropna()
        step = diffs.median() if len(diffs) else pd.Timedelta(hours=1)
        freq_hours = max(1, int(step.total_seconds() // 3600))

    steps_24h = max(1, 24 // freq_hours)

    df["pm25_roll24"] = df["pm2_5"].rolling(steps_24h, min_periods=max(1, steps_24h//2)).mean()
    df["aqi_roll24"]  = df["aqi"].rolling(steps_24h, min_periods=max(1, steps_24h//2)).mean()

    df["aqi_lag1"]   = df["aqi"].shift(1)
    df["aqi_lag24"]  = df["aqi"].shift(steps_24h)
    df["pm25_lag1"]  = df["pm2_5"].shift(1)
    df["pm25_lag24"] = df["pm2_5"].shift(steps_24h)

    return df.dropna()

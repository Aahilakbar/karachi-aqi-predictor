import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1] 
    sys.path.insert(0, str(ROOT))

from src.config import LAT, LON, OPENWEATHER_KEY

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .stApp { background-color: white !important; }
    section.main * , .stApp * { color: #111827 !important; }
    section[data-testid="stSidebar"] { background-color: #111827 !important; }
    section[data-testid="stSidebar"] * { color: #F9FAFB !important; }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        color: #111827 !important;
        background: #F9FAFB !important;
    }
    section[data-testid="stSidebar"] button { color: #111827 !important; }
    .section-header {
        font-size: 1.5rem;
        color: #111827 !important;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        color: #111827 !important;
    }
    .aqi-good { background-color: #10B981; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 700; display:inline-block; }
    .aqi-moderate { background-color: #F59E0B; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 700; display:inline-block; }
    .aqi-unhealthy-sensitive { background-color: #F97316; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 700; display:inline-block; }
    .aqi-unhealthy { background-color: #EF4444; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 700; display:inline-block; }
    .aqi-very-unhealthy { background-color: #8B5CF6; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 700; display:inline-block; }
    .aqi-hazardous { background-color: #7C3AED; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 700; display:inline-block; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.10);
    }
    .metric-card * { color: white !important; }
    section[data-testid="stSidebar"] .info-box {
        background-color: #2D3748 !important;
        border-left: 4px solid #3B82F6 !important;
        color: #F9FAFB !important;
    }
    section[data-testid="stSidebar"] .info-box * {
        color: #F9FAFB !important;
    }
    .vega-tooltip {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #111827 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 12px !important;
    }
    .vega-tooltip table,
    .vega-tooltip tbody,
    .vega-tooltip tr,
    .vega-tooltip td,
    .vega-tooltip th {
        color: #111827 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helper functions for AQI categories and health advice
# ---------------------------
def get_aqi_category(aqi):
    """Return (category_name, css_class, emoji) based on AQI value."""
    if pd.isna(aqi) or aqi is None:
        return "Unknown", "", "❓"
    if aqi <= 50:
        return "Good", "aqi-good", "😊"
    elif aqi <= 100:
        return "Moderate", "aqi-moderate", "😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "aqi-unhealthy-sensitive", "😷"
    elif aqi <= 200:
        return "Unhealthy", "aqi-unhealthy", "😷"
    elif aqi <= 300:
        return "Very Unhealthy", "aqi-very-unhealthy", "⚠️"
    else:
        return "Hazardous", "aqi-hazardous", "🚨"

def get_health_advice(aqi):
    """Return health advisory text based on AQI."""
    if pd.isna(aqi) or aqi is None:
        return "No data available."
    if aqi <= 50:
        return "Air quality is satisfactory. Enjoy your usual outdoor activities."
    elif aqi <= 100:
        return "Air quality is acceptable. Very sensitive people should reduce prolonged outdoor exertion."
    elif aqi <= 150:
        return "Sensitive groups may experience effects. General public is less likely to be affected."
    elif aqi <= 200:
        return "Everyone may begin to experience effects. Sensitive groups should avoid prolonged outdoor exertion."
    elif aqi <= 300:
        return "Health alert: everyone may experience more serious effects. Avoid outdoor exertion."
    else:
        return "Emergency conditions. Everyone should avoid outdoor activities."


def apply_altair_theme(chart):
    """Apply custom theme to Altair chart (white background, dark text)."""
    return (
        chart.configure(background="white")
        .configure_axis(labelColor="#111827", titleColor="#111827", gridColor="#E5E7EB")
        .configure_title(color="#111827")
        .configure_legend(labelColor="#111827", titleColor="#111827")
    )

def altair_line_single(df, y_col, title="", height=320):
    """
    Create a simple line chart with one variable.
    Expects DataFrame with timestamp as index or a 'timestamp' column.
    """
    tmp = df.copy().reset_index()
    if "timestamp" not in tmp.columns:
        tmp = tmp.rename(columns={tmp.columns[0]: "timestamp"})
    
    tmp = tmp.dropna(subset=[y_col])
    if tmp.empty:
        return None

    chart = (
        alt.Chart(tmp)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip(f"{y_col}:Q", title=y_col, format=".2f"),
            ],
        )
        .properties(title=title, height=height)
    )
    return apply_altair_theme(chart)

def altair_line_multi(df, cols, title="", height=360, y_title="Value"):
    """
    Create a multi-line chart for several variables.
    Expects DataFrame with timestamp as index or a 'timestamp' column.
    """
    tmp = df.copy().reset_index()
    if "timestamp" not in tmp.columns:
        tmp = tmp.rename(columns={tmp.columns[0]: "timestamp"})
    
    valid_cols = [c for c in cols if c in tmp.columns]
    if not valid_cols:
        return None
    
    melted = tmp.melt(id_vars=["timestamp"], value_vars=valid_cols, var_name="series", value_name="value")
    melted = melted.dropna(subset=["value"])
    if melted.empty:
        return None

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
        .properties(title=title, height=height)
    )
    return apply_altair_theme(chart)


def aqi_from_pm25(pm):
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if pm is None or pd.isna(pm):
        return np.nan
    pm = float(pm)
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= pm <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm - c_lo) + i_lo)
    return 500 if pm > 500.4 else 0

# ---------------------------
# Data fetching functions 
# ---------------------------
@st.cache_data(ttl=600)
def fetch_weather_history(days_back=120):
    """Fetch historical weather data from Open‑Meteo archive."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        "&timezone=UTC"
    )
    w = requests.get(url, timeout=30).json()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(w["hourly"]["time"], utc=True),
            "temp": w["hourly"]["temperature_2m"],
            "humidity": w["hourly"]["relative_humidity_2m"],
            "wind": w["hourly"]["wind_speed_10m"],
            "pressure": w["hourly"]["pressure_msl"],
        }
    ).sort_values("timestamp")

]    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["ts_key"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df

def _unix(dt: datetime) -> int:
    """Convert datetime to Unix timestamp (seconds since epoch)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

@st.cache_data(ttl=600)
def fetch_pollution_history(days_back=120):
    """Fetch historical pollution data from OpenWeather API."""
    if not OPENWEATHER_KEY:
        raise ValueError("OPENWEATHER_KEY missing in .env")

    end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)

    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

    all_rows = []
    chunk = timedelta(days=7)
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + chunk, end_dt)
        params = {"lat": LAT, "lon": LON, "start": _unix(cur), "end": _unix(nxt), "appid": OPENWEATHER_KEY}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()

        for item in js.get("list", []):
            ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            comps = item.get("components", {})
            all_rows.append(
                {
                    "timestamp": ts,
                    "pm2_5": comps.get("pm2_5"),
                    "pm10": comps.get("pm10"),
                    "no2": comps.get("no2"),
                    "so2": comps.get("so2"),
                    "co": comps.get("co"),
                    "o3": comps.get("o3"),
                }
            )

        cur = nxt
        time.sleep(0.12) 

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["aqi"] = df["pm2_5"].apply(aqi_from_pm25)
    df["ts_key"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df

# ---------------------------
# Feature engineering functions
# ---------------------------
def build_hourly_join(df_w, df_p):
    """Join weather and pollution DataFrames on ts_key, clean timestamps."""
    df = df_w.merge(df_p, on="ts_key", how="inner", suffixes=("_w", "_p"))
    ts_col = "timestamp_w" if "timestamp_w" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    keep = [
        "temp", "humidity", "wind", "pressure",
        "hour", "dow", "month",
        "pm2_5", "pm10", "no2", "so2", "co", "o3",
        "aqi", "ts_key",
    ]
    df = df[[c for c in keep if c in df.columns]].dropna()
    return df

def resample_df(df, freq="6H"):
    """Resample hourly data to 6‑hour intervals (mean)."""
    df_num = df.resample(freq).mean(numeric_only=True)
    # Re‑add time features based on new index
    df_num["hour"] = df_num.index.hour
    df_num["dow"] = df_num.index.dayofweek
    df_num["month"] = df_num.index.month
    return df_num.dropna()

def add_lag_roll(df):
    """
    Add lag and rolling features for AQI and PM2.5.
    Also create nonlinear features (sin/cos, temp*humidity, wind²).
    """
    df = df.copy().sort_index()
    steps_24h = 4  

    df["pm25_roll24"] = df["pm2_5"].rolling(steps_24h).mean()
    df["aqi_roll24"] = df["aqi"].rolling(steps_24h).mean()
    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_lag24"] = df["aqi"].shift(steps_24h)
    df["pm25_lag1"] = df["pm2_5"].shift(1)
    df["pm25_lag24"] = df["pm2_5"].shift(steps_24h)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Interaction and non‑linear terms
    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["wind_sq"] = df["wind"] ** 2

    return df.dropna()

# Final list of features used by the models
FEATURES = [
    "temp", "humidity", "wind", "pressure",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "hour", "dow", "month",
    "pm25_roll24", "aqi_roll24",
    "aqi_lag1", "aqi_lag24",
    "pm25_lag1", "pm25_lag24",
    "hour_sin", "hour_cos", "temp_humidity", "wind_sq",
]

# ---------------------------
# Model training with three algorithms
# ---------------------------
def train_three_models(df):
    """
    Train Ridge, Random Forest, and Gradient Boosting on the feature DataFrame.
    Return results, best model name, trained model objects, and test predictions.
    """
    X = df[FEATURES].copy()
    y = df["aqi"].copy()

    if y.nunique() <= 1:
        st.warning("⚠️ Target variable (AQI) is constant. Using fallback model.")
        results_df = pd.DataFrame({
            "Model": ["Ridge", "Random Forest", "Gradient Boosting"],
            "MAE": [0.0, 0.0, 0.0],
            "RMSE": [0.0, 0.0, 0.0],
            "R2": [1.0, 1.0, 1.0]
        })
        best_name = "Ridge"
        trained = {}
        preds_table = pd.DataFrame({"timestamp": df.index[int(len(df)*0.8):], "actual_aqi": y.iloc[int(len(df)*0.8):].values})
        for name in ["Ridge", "Random Forest", "Gradient Boosting"]:
            preds_table[f"pred_{name}"] = y.iloc[int(len(df)*0.8):].values
            trained[name] = (Ridge(alpha=1.0), None)
        return results_df, best_name, trained, preds_table

    if X.isnull().any().any():
        X = X.fillna(X.mean())
    if y.isnull().any():
        y = y.fillna(y.mean())

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(y_test) == 0:
        st.warning("⚠️ Test set is empty. Adjusting split to 50%.")
        split = max(1, int(len(df) * 0.5))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Define models
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = []
    preds_table = pd.DataFrame({"timestamp": df.index[split:], "actual_aqi": y_test.values})
    trained = {}

    for name, m in models.items():
        try:
            m.fit(X_train, y_train)
            pred = m.predict(X_test)

            if name == "Ridge":
                scaler = m.named_steps["scaler"]
                trained[name] = (m.named_steps["ridge"], scaler)
            else:
                trained[name] = (m, None)

            mae = mean_absolute_error(y_test, pred) if len(y_test) > 0 else 0.0
            rmse = mean_squared_error(y_test, pred) ** 0.5 if len(y_test) > 0 else 0.0
            r2 = r2_score(y_test, pred) if len(y_test) > 0 and y_test.std() > 0 else 0.0

            results.append([name, mae, rmse, r2])
            preds_table[f"pred_{name}"] = pred
        except Exception as e:
            st.warning(f"⚠️ Model {name} failed: {e}")
            results.append([name, 999.0, 999.0, 0.0])
            preds_table[f"pred_{name}"] = np.nan
            trained[name] = (Ridge(alpha=1.0), StandardScaler()) 

    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("RMSE")
    best_name = results_df.iloc[0]["Model"]
    return results_df, best_name, trained, preds_table

# ---------------------------
# Weather forecast for future days 
# ---------------------------
@st.cache_data(ttl=600)
def get_weather_forecast(days=3):
    """Fetch weather forecast from Open‑Meteo for the next `days` days."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        f"&forecast_days={days}&timezone=UTC"
    )
    wf = requests.get(url, timeout=30).json()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(wf["hourly"]["time"]),
            "temp": wf["hourly"]["temperature_2m"],
            "humidity": wf["hourly"]["relative_humidity_2m"],
            "wind": wf["hourly"]["wind_speed_10m"],
            "pressure": wf["hourly"]["pressure_msl"],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df = df.set_index("timestamp").sort_index()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def future_predictions_6h(df_hist_resampled, trained_models, forecast_days=3, best_name="Ridge", freq="6H"):
    """
    Generate 6‑hourly forecasts for `forecast_days` using a recursive approach.
    Returns DataFrame with timestamps and predictions from all models.
    """
    try:
        freq_hours = 6
        n_steps = forecast_days * 24 // freq_hours  

        st.info(f"Generating {n_steps} forecast points ({forecast_days} days at {freq_hours}-hour intervals)")

        df_weather_hourly = get_weather_forecast(days=forecast_days + 1) 
        df_weather = df_weather_hourly.resample(freq).mean().dropna()

        last_hist_time = df_hist_resampled.index[-1]
        next_forecast_start = last_hist_time + pd.Timedelta(hours=freq_hours)
        df_weather = df_weather[df_weather.index >= next_forecast_start]

        if len(df_weather) > n_steps:
            df_weather = df_weather.iloc[:n_steps]
        elif len(df_weather) < n_steps:
            st.warning("Limited weather forecast data. Using available data and extending with trends.")
            last_rows = df_weather.iloc[-1:].copy()
            while len(df_weather) < n_steps:
                last_rows.index = last_rows.index + pd.Timedelta(hours=freq_hours)
                df_weather = pd.concat([df_weather, last_rows])

        df_weather["hour"] = df_weather.index.hour
        df_weather["dow"] = df_weather.index.dayofweek
        df_weather["month"] = df_weather.index.month
        df_weather["hour_sin"] = np.sin(2 * np.pi * df_weather["hour"] / 24)
        df_weather["hour_cos"] = np.cos(2 * np.pi * df_weather["hour"] / 24)
        df_weather["temp_humidity"] = df_weather["temp"] * df_weather["humidity"]
        df_weather["wind_sq"] = df_weather["wind"] ** 2

        hist = df_hist_resampled.sort_index().copy()
        steps_24h = 4  

        aqi_buf = deque(hist["aqi"].tail(steps_24h * 2).tolist(), maxlen=steps_24h * 2)
        pm25_buf = deque(hist["pm2_5"].tail(steps_24h * 2).tolist(), maxlen=steps_24h * 2)

        rng = np.random.default_rng(42)
        hourly_pattern = hist.groupby(hist.index.hour)["pm2_5"].mean().to_dict()

        future_pm25 = []
        for i in range(n_steps):
            current_time = df_weather.index[i]
            hour = current_time.hour

            base_pm25 = pm25_buf[-1] if len(pm25_buf) > 0 else 50
            hourly_factor = hourly_pattern.get(hour, 1.0) / hourly_pattern.get(12, 1.0)

            temp = df_weather.iloc[i]["temp"]
            humidity = df_weather.iloc[i]["humidity"]
            wind = df_weather.iloc[i]["wind"]

            weather_factor = 1.0 + (temp - 25) * 0.01 - (wind - 10) * 0.02 + (humidity - 50) * 0.005

            noise = float(rng.normal(1.0, 0.1))

            pm25 = base_pm25 * hourly_factor * weather_factor * noise
            pm25 = max(5.0, min(pm25, 350.0))
            future_pm25.append(pm25)

        df_weather["pm2_5"] = future_pm25
        df_weather["pm10"] = df_weather["pm2_5"] * 1.5
        df_weather["no2"] = df_weather["pm2_5"] * 0.3
        df_weather["so2"] = df_weather["pm2_5"] * 0.1
        df_weather["co"] = df_weather["pm2_5"] * 0.05
        df_weather["o3"] = df_weather["pm2_5"] * 0.2

        preds_out = {name: [] for name in trained_models.keys()}
        idxs = []

        for i, (ts, row) in enumerate(df_weather.iterrows()):
            row_dict = row.to_dict()

            aqi_vals = list(aqi_buf)[-steps_24h:] if len(aqi_buf) >= steps_24h else [50] * steps_24h
            pm25_vals = list(pm25_buf)[-steps_24h:] if len(pm25_buf) >= steps_24h else [50] * steps_24h

            row_dict["pm25_roll24"] = np.mean(pm25_vals)
            row_dict["aqi_roll24"] = np.mean(aqi_vals)
            row_dict["aqi_lag1"] = aqi_vals[-1] if aqi_vals else 50
            row_dict["aqi_lag24"] = aqi_vals[0] if aqi_vals else 50
            row_dict["pm25_lag1"] = pm25_vals[-1] if pm25_vals else 50
            row_dict["pm25_lag24"] = pm25_vals[0] if pm25_vals else 50

            X = pd.DataFrame([row_dict])[FEATURES]

            # Predict with all models
            for name, (m, scaler) in trained_models.items():
                try:
                    if name == "Ridge" and scaler is not None:
                        X_scaled = scaler.transform(X)
                        pred = float(m.predict(X_scaled)[0])
                    else:
                        pred = float(m.predict(X)[0])
                    pred = max(0, min(pred, 500))
                except Exception:
                    pred = 50.0  
                preds_out[name].append(pred)

            next_aqi = preds_out[best_name][-1]
            aqi_buf.append(next_aqi)
            pm25_buf.append(row_dict["pm2_5"])
            idxs.append(ts)

        out = pd.DataFrame(index=pd.DatetimeIndex(idxs))
        for name in trained_models.keys():
            out[f"pred_{name}"] = preds_out[name]

        st.success(f"Successfully generated {len(out)} forecast points")
        return out

    except Exception as e:
        st.error(f"❌ Forecast generation failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        # Return a dummy forecast of 12 steps (3 days)
        now = pd.Timestamp.now(tz='UTC')
        start = now.ceil("6H")
        empty_idx = pd.date_range(start=start, periods=12, freq="6H")
        out = pd.DataFrame(index=empty_idx)
        for name in trained_models.keys():
            out[f"pred_{name}"] = 50.0
        return out

# ---------------------------
# Walk-forward backtest 
# ---------------------------
def walk_forward_backtest(df, model, start_ratio=0.7):
    """
    Perform walk‑forward validation:
    - Train on first `start_ratio` of data, then predict next point,
      retrain every 10 steps.
    Returns MAE, RMSE, R² and DataFrame of actual vs predicted.
    """
    X = df[FEATURES].copy()
    y = df["aqi"].copy()

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    start = max(1, int(len(df) * start_ratio))
    if start >= len(df):
        start = len(df) // 2

    preds, actuals, idxs = [], [], []

    try:
        model.fit(X.iloc[:start], y.iloc[:start])

        for i in range(start, len(df)):
            xi = X.iloc[i:i+1]
            yi = float(y.iloc[i])
            pi = float(model.predict(xi)[0])

            preds.append(pi)
            actuals.append(yi)
            idxs.append(df.index[i])

            if i % 10 == 0:
                model.fit(X.iloc[:i+1], y.iloc[:i+1])

        if len(actuals) > 0 and len(preds) > 0:
            mae = mean_absolute_error(actuals, preds)
            rmse = mean_squared_error(actuals, preds) ** 0.5
            if np.std(actuals) > 0:
                r2 = r2_score(actuals, preds)
            else:
                r2 = 1.0 if np.allclose(actuals, preds) else 0.0
        else:
            mae, rmse, r2 = 0.0, 0.0, 0.0

        out = pd.DataFrame({"actual": actuals, "pred": preds}, index=pd.DatetimeIndex(idxs))
        return mae, rmse, r2, out

    except Exception as e:
        st.warning(f"⚠️ Walk-forward backtest failed: {e}")
        return 0.0, 0.0, 0.0, pd.DataFrame()

# ---------------------------
# Build combined timeline table (history + forecast)
# ---------------------------
def build_full_timeline_table(df_hist_resampled, future_pred, best_name, history_steps=12):
    """
    Combine last `history_steps` of historical AQI with forecast predictions.
    Returns a DataFrame with columns: timestamp, type, actual_aqi, pred_*, aqi_category.
    """
    hist = df_hist_resampled.sort_index().copy()
    hist_last = hist.tail(history_steps)[["aqi"]].rename(columns={"aqi": "actual_aqi"})

    fut = future_pred.copy()
    fut["actual_aqi"] = np.nan

    hist_last["type"] = "history"
    fut["type"] = "forecast"

    out = pd.concat([hist_last, fut], axis=0).sort_index()
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    for col in pred_cols:
        out[col] = out[col].apply(lambda x: int(round(x)) if pd.notna(x) and x is not None else None)
    out["actual_aqi"] = out["actual_aqi"].apply(lambda x: int(round(x)) if pd.notna(x) else None)

    cols = ["timestamp", "type", "actual_aqi"] + pred_cols
    out = out[cols]

    out["aqi_category"] = out.apply(
        lambda row: (
            get_aqi_category(row["actual_aqi"])[0]
            if pd.notna(row["actual_aqi"])
            else (
                get_aqi_category(row[pred_cols[0]])[0]
                if pred_cols and pd.notna(row[pred_cols[0]])
                else "Unknown"
            )
        ),
        axis=1
    )
    return out

# ---------------------------
# Sidebar UI
# ---------------------------
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 16px;'>
        <h3 style='color: white; margin: 0;'>🌫️ AQI Predictor</h3>
        <p style='color: white; margin: 6px 0 0 0; font-size: 12px;'>Karachi, Pakistan</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.header("⚙️ Controls")

    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    with st.expander("📊 Data Settings", expanded=True):
        days_back = st.slider("History (days)", 30, 60, 5)
        st.info("Data frequency: 6-Hourly")
        chart_range = st.radio(
            "Show historical chart for:",
            options=["1 day", "30 days", "60 days"],
            index=1,
            horizontal=True,
        )

    st.caption("📍 Location: Karachi, Pakistan")
    st.caption("🕐 Data refresh cache: 10 minutes")

# ---------------------------
# Main content
# ---------------------------
st.markdown("<h1 style='margin-bottom:0;'>🌫️ Karachi Air Quality Index Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict AQI using machine learning models with 6-hourly intervals")

progress_bar = st.progress(0)
status_text = st.empty()

try:
    status_text.text("Fetching weather data...")
    df_weather = fetch_weather_history(days_back=days_back)
    progress_bar.progress(30)

    status_text.text("Fetching pollution data...")
    df_poll = fetch_pollution_history(days_back=days_back)
    progress_bar.progress(60)

    status_text.text("Processing and joining data...")
    df_hourly = build_hourly_join(df_weather, df_poll)
    progress_bar.progress(90)
except Exception as e:
    progress_bar.empty()
    status_text.empty()
    st.error(f"❌ Data fetch failed: {e}")
    st.stop()

progress_bar.progress(100)
time.sleep(0.2)
progress_bar.empty()
status_text.empty()

if df_hourly.empty:
    st.error("❌ No data available. Check API key / internet.")
    st.stop()

# Resample to 6‑hourly and add features
df_used = resample_df(df_hourly, freq="6H")
df_used = add_lag_roll(df_used)

if len(df_used) < 10:
    st.warning(f"⚠️ Very limited data available ({len(df_used)} rows). Results may be unreliable.")
elif len(df_used) < 50:
    st.warning(f"⚠️ Limited data available ({len(df_used)} rows).")

with st.spinner("Training models..."):
    results_df, best_name, trained, preds_table = train_three_models(df_used)

# ---------------------------
# Current Status Section
# ---------------------------
st.markdown("<div class='section-header'>📊 Current Status</div>", unsafe_allow_html=True)

current_aqi = float(df_hourly["aqi"].iloc[-1]) if not df_hourly.empty else 0
current_temp = float(df_hourly["temp"].iloc[-1]) if not df_hourly.empty else 0
last_update = df_hourly.index[-1] if not df_hourly.empty else datetime.now()

aqi_category, aqi_class, aqi_emoji = get_aqi_category(current_aqi)
health_advice = get_health_advice(current_aqi)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        f"""
    <div class='metric-card'>
        <div style='font-size: 14px; opacity: 0.9;'>Current AQI</div>
        <div style='font-size: 36px; font-weight: 800;'>{current_aqi:.0f}</div>
        <div class='{aqi_class}'>{aqi_emoji} {aqi_category}</div>
        <div style='font-size: 12px; opacity: 0.9; margin-top: 8px;'>
            Last Updated: {last_update.strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
    <div style='background-color:#F3F4F6; padding:15px; border-radius:10px;'>
        <div style='font-size: 14px; color:#6B7280;'>Temperature</div>
        <div style='font-size: 28px; font-weight: 700; color:#DC2626;'>{current_temp:.1f}°C</div>
        <div style='font-size: 12px; color:#6B7280; margin-top: 5px;'>Current reading</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
<div class='info-box'>
    <b>🏥 Health Advisory</b><br>
    {health_advice}<br>
    <span style='font-size: 12px; opacity: 0.85;'>AQI: {current_aqi:.0f} | Category: {aqi_category}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# 24-hour trend 
# ---------------------------
st.markdown("<div class='section-header'>📈 AQI Trend (Last 24 Hours)</div>", unsafe_allow_html=True)
df_last_24h = df_hourly.tail(24)[["aqi"]] 
chart = altair_line_single(df_last_24h, "aqi", title="AQI (Last 24 Hours)", height=260)
if chart:
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No data for 24-hour trend.")

if len(df_last_24h) > 1:
    first_aqi = float(df_last_24h["aqi"].iloc[0])
    last_aqi = float(df_last_24h["aqi"].iloc[-1])
    trend = "↗️ Increasing" if last_aqi > first_aqi else "↘️ Decreasing" if last_aqi < first_aqi else "➡️ Stable"
    t1, t2 = st.columns(2)
    t1.metric("24h Trend", trend)
    t2.metric("Change", f"{last_aqi-first_aqi:+.0f}", delta=f"From {first_aqi:.0f} to {last_aqi:.0f}")

# ---------------------------
# Tabs for Historical Data, Model Performance, and Forecast
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📈 Historical Data", "🤖 Model Performance", "🔮 Future Forecast"])

with tab1:
    st.markdown("<div class='section-header'>📈 Historical AQI Data</div>", unsafe_allow_html=True)

    range_map = {"1 day": 1, "30 days": 30, "60 days": 60, "120 days": 120}
    days_to_show = range_map[chart_range]

    df_display = df_used.last(f"{days_to_show}D") if len(df_used) > 0 else df_used

    if df_display.empty:
        st.warning("No data in selected range.")
    else:
        chart = altair_line_single(df_display[["aqi"]], "aqi", title=f"Historical AQI (Last {days_to_show} day{'s' if days_to_show>1 else ''})", height=380)
        if chart:
            st.altair_chart(chart, use_container_width=True)

        df_last_60d = df_used.last("60D")
        if not df_last_60d.empty:
            avg_aqi = df_last_60d["aqi"].mean()
            max_aqi = df_last_60d["aqi"].max()
            st.markdown("<div class='section-header'>📊 Statistics (Last 60 Days)</div>", unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric("Average AQI (60d)", f"{avg_aqi:.1f}")
            s2.metric("Maximum AQI (60d)", f"{max_aqi:.0f}")
        else:
            st.info("Not enough data for 60-day statistics.")

        with st.expander("📋 View Raw Data"):
            st.dataframe(df_display[["aqi", "pm2_5", "temp", "humidity", "wind"]].tail(200), use_container_width=True, height=320)
            st.download_button(
                "📥 Download Historical Data (CSV)",
                data=df_display.to_csv(),
                file_name=f"aqi_historical_{datetime.utcnow().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab2:
    st.markdown("<div class='section-header'>🤖 Model Performance Comparison</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class='info-box'>
    <b>🏆 Best Performing Model:</b> {best_name}<br>
    Selected based on lowest RMSE.
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("### 📊 Model Performance Metrics")
    display_df = results_df.copy()
    display_df["MAE"] = display_df["MAE"].round(2)
    display_df["RMSE"] = display_df["RMSE"].round(2)
    display_df["R2"] = display_df["R2"].round(3)
    st.dataframe(display_df, use_container_width=True, height=160)

    st.markdown("<div class='section-header'>📊 Test Set Predictions vs Actual</div>", unsafe_allow_html=True)
    if not preds_table.empty:
        preds_table2 = preds_table.set_index("timestamp")
        chart_test_data = preds_table2[["actual_aqi"]].copy()
        for model_name in ["Ridge", "Random Forest", "Gradient Boosting"]:
            coln = f"pred_{model_name}"
            if coln in preds_table2.columns:
                chart_test_data[model_name] = preds_table2[coln]
        chart = altair_line_multi(chart_test_data, cols=list(chart_test_data.columns), title="Test Set: Actual vs Models", height=420)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No test data available for visualization.")
    else:
        st.info("No prediction data available.")

    st.markdown("<div class='section-header'>🔄 Walk-Forward Backtest Validation</div>", unsafe_allow_html=True)

    best_model, best_scaler = trained[best_name]

    with st.spinner("Running walk-forward validation..."):
        if best_name == "Ridge" and best_scaler is not None:
            X_scaled = pd.DataFrame(
                best_scaler.transform(df_used[FEATURES]),
                index=df_used.index,
                columns=FEATURES
            )
            df_scaled = df_used.copy()
            df_scaled[FEATURES] = X_scaled
            mae_bt, rmse_bt, r2_bt, bt_df = walk_forward_backtest(df_scaled, best_model)
        else:
            mae_bt, rmse_bt, r2_bt, bt_df = walk_forward_backtest(df_used, best_model)

    b1, b2, b3 = st.columns(3)
    b1.metric("Walk-Forward MAE", f"{mae_bt:.2f}")
    b2.metric("Walk-Forward RMSE", f"{rmse_bt:.2f}")
    b3.metric("Walk-Forward R²", f"{r2_bt:.3f}")

    if not bt_df.empty:
        chart = altair_line_multi(bt_df, cols=["actual", "pred"], title="Walk-forward Backtest (Actual vs Pred)", height=360, y_title="AQI")
        if chart:
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No backtest data available.")

with tab3:
    st.markdown("<div class='section-header'>🔮 AQI Forecast Predictions</div>", unsafe_allow_html=True)

    with st.spinner("Generating 3-day forecast (6-hourly intervals)..."):
        future_6h = future_predictions_6h(
            df_used,
            trained,
            forecast_days=3,
            best_name=best_name,
            freq="6H"
        )

    if future_6h.empty or len(future_6h) < 12:
        st.warning(f"Expected 12 forecast points but got {len(future_6h)}. Using default values.")
        # Create a dummy forecast if generation failed
        now = pd.Timestamp.now(tz='UTC')
        start = now.ceil("6H")
        default_idx = pd.date_range(start=start, periods=12, freq="6H")
        future_6h = pd.DataFrame(index=default_idx)
        for name in trained.keys():
            future_6h[f"pred_{name}"] = 50.0

    best_col = f"pred_{best_name}"
    next_step_aqi = float(future_6h.iloc[0][best_col]) if best_col in future_6h.columns else 50.0
    avg_forecast_aqi = float(future_6h[best_col].mean()) if best_col in future_6h.columns else 50.0
    max_forecast_aqi = float(future_6h[best_col].max()) if best_col in future_6h.columns else 50.0

    next_category, next_class, next_emoji = get_aqi_category(next_step_aqi)
    avg_category, _, _ = get_aqi_category(avg_forecast_aqi)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
            <div style='background-color:#F0F9FF; padding:20px; border-radius:10px; text-align:center; border:2px solid #3B82F6;'>
                <div style='font-size:14px; color:#1E40AF;'>Next 6H Forecast</div>
                <div style='font-size:42px; font-weight:800; color:#1E40AF;'>{next_step_aqi:.0f}</div>
                <div class='{next_class}' style='margin-top:10px;'>{next_emoji} {next_category}</div>
            </div>""", unsafe_allow_html=True)

    with f2:
        st.markdown(f"""
            <div style='background-color:#FEF3C7; padding:20px; border-radius:10px; text-align:center; border:2px solid #F59E0B;'>
                <div style='font-size:14px; color:#92400E;'>Avg Forecast (3 Days)</div>
                <div style='font-size:32px; font-weight:800; color:#92400E;'>{avg_forecast_aqi:.0f}</div>
                <div style='font-size:16px; color:#92400E; margin-top:10px;'>{avg_category}</div>
            </div>""", unsafe_allow_html=True)

    with f3:
        st.markdown(f"""
            <div style='background-color:#FEE2E2; padding:20px; border-radius:10px; text-align:center; border:2px solid #EF4444;'>
                <div style='font-size:14px; color:#991B1B;'>Peak Forecast</div>
                <div style='font-size:32px; font-weight:800; color:#991B1B;'>{max_forecast_aqi:.0f}</div>
                <div style='font-size:16px; color:#991B1B; margin-top:10px;'>Maximum expected</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📈 6‑Hourly Forecast (Next 3 Days)</div>", unsafe_allow_html=True)
    forecast_chart = future_6h.copy()
    forecast_chart.columns = [c.replace("pred_", "") for c in forecast_chart.columns]
    st.caption(f"Showing {len(forecast_chart)} forecast points (12 points = 3 days of 6-hourly data)")

    chart_data = forecast_chart.reset_index().rename(columns={"index": "timestamp"})
    melted_data = pd.melt(
        chart_data,
        id_vars=["timestamp"],
        value_vars=list(forecast_chart.columns),
        var_name="Model",
        value_name="AQI"
    )

    chart = alt.Chart(melted_data).mark_line(point=True).encode(
        x=alt.X("timestamp:T", title="Date & Time", axis=alt.Axis(format="%b %d %H:%M", labelAngle=-45)),
        y=alt.Y("AQI:Q", title="AQI", scale=alt.Scale(domain=[0, 200])),
        color=alt.Color("Model:N", title="Model"),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M"),
            alt.Tooltip("Model:N"),
            alt.Tooltip("AQI:Q", format=".0f")
        ]
    ).properties(title="6‑Hourly Forecast for Next 3 Days", height=420).configure_axis(
        labelFontSize=11, titleFontSize=12
    ).configure_legend(titleFontSize=12, labelFontSize=11)

    chart = apply_altair_theme(chart)
    st.altair_chart(chart, use_container_width=True)

    with st.expander("View Forecast Data"):
        st.dataframe(forecast_chart)

    st.markdown("<div class='section-header'>📋 Detailed Timeline (3 days history + 3 days forecast)</div>", unsafe_allow_html=True)
    combined = build_full_timeline_table(df_used, future_6h, best_name, history_steps=12)
    st.dataframe(combined, use_container_width=True, height=420)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download Timeline (CSV)", data=combined.to_csv(index=False), file_name=f"aqi_timeline_{datetime.utcnow().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("📥 Download Forecast Only (CSV)", data=future_6h.to_csv(), file_name=f"aqi_forecast_{datetime.utcnow().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.divider()
fc1, fc2, fc3 = st.columns(3)

with fc1:
    st.markdown(
        """
**Data Sources:**
- Open-Meteo (Weather)
- OpenWeather (Pollution History)
"""
    )
with fc2:
    st.markdown(
        """
**Models Used:**
- Ridge Regression (scaled)
- Random Forest
- Gradient Boosting
"""
    )
with fc3:
    st.markdown(
        f"""  
**📍 Location:** Karachi (Lat: {LAT:.4f}, Lon: {LON:.4f})
"""
    )

st.markdown("<div style='text-align:center; color:#6B7280; font-size: 13px; padding: 8px 0;'>AQI Predictor • 6-Hourly Intervals • Powered by ML • Refresh for latest data</div>", unsafe_allow_html=True)

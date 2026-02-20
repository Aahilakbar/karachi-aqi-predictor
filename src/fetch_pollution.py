import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

def unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def fetch_openweather_air_history(lat, lon, api_key, start_dt, end_dt):
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": unix(start_dt), "end": unix(end_dt), "appid": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_openweather_air_current(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def aqi_from_pm25(pm):
    bps = [
        (0.0, 12.0,   0,  50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101,150),
        (55.5, 150.4,151,200),
        (150.5,250.4,201,300),
        (250.5,350.4,301,400),
        (350.5,500.4,401,500),
    ]
    if pm is None:
        return None
    pm = float(pm)
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= pm <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm - c_lo) + i_lo)
    return 500 if pm > 500.4 else 0

def get_pollution_live(lat, lon, api_key, days_back=120):
    end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)

    # history in chunks
    all_rows = []
    chunk = timedelta(days=7)
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + chunk, end_dt)
        js = fetch_openweather_air_history(lat, lon, api_key, cur, nxt)
        for item in js.get("list", []):
            ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            comps = item.get("components", {})
            all_rows.append({
                "timestamp": ts,
                "pm2_5": comps.get("pm2_5"),
                "pm10": comps.get("pm10"),
                "no2": comps.get("no2"),
                "so2": comps.get("so2"),
                "co": comps.get("co"),
                "o3": comps.get("o3"),
            })
        cur = nxt
        time.sleep(0.15)

    # current (latest)
    js_now = fetch_openweather_air_current(lat, lon, api_key)
    for item in js_now.get("list", []):
        ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comps = item.get("components", {})
        all_rows.append({
            "timestamp": ts,
            "pm2_5": comps.get("pm2_5"),
            "pm10": comps.get("pm10"),
            "no2": comps.get("no2"),
            "so2": comps.get("so2"),
            "co": comps.get("co"),
            "o3": comps.get("o3"),
        })

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["aqi"] = df["pm2_5"].apply(aqi_from_pm25)
    df["ts_key"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df

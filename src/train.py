import hopsworks
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config import HOPSWORKS_PROJECT, HOPSWORKS_API_KEY

# Full feature list 
FEATURES = [
    "temp", "humidity", "wind", "pressure",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "hour", "dow", "month",
    "pm25_roll24", "aqi_roll24",
    "aqi_lag1", "aqi_lag24",
    "pm25_lag1", "pm25_lag24",
    "hour_sin", "hour_cos", "temp_humidity", "wind_sq",
]

def load_features_from_fs(as_of_date=None):
    """Load feature DataFrame from Hopsworks, optionally as of a certain date."""
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features_6h", version=1)

]    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df

def train_three_models(df):
    """Same training function as before, but now df comes from feature store."""
    X = df[FEATURES].copy()
    y = df["aqi"].copy()

    # Handle missing values
    if X.isnull().any().any():
        X = X.fillna(X.mean())
    if y.isnull().any():
        y = y.fillna(y.mean())

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(y_test) == 0:
        split = max(1, int(len(df) * 0.5))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = []
    preds_table = pd.DataFrame({"timestamp": df.index[split:], "actual_aqi": y_test.values})
    trained = {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)

        if name == "Ridge":
            trained[name] = (m.named_steps["ridge"], m.named_steps["scaler"])
        else:
            trained[name] = (m, None)

        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred) if y_test.std() > 0 else 0.0

        results.append([name, mae, rmse, r2])
        preds_table[f"pred_{name}"] = pred

    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("RMSE")
    best_name = results_df.iloc[0]["Model"]


    return results_df, best_name, trained, preds_table

if __name__ == "__main__":
    df = load_features_from_fs()
    results, best, _, _ = train_three_models(df)
    print("Training complete. Best model:", best)
    print(results)

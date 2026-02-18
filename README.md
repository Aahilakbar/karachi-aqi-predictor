Karachi AQI Predictor ( https://aahil-karachi-aqi-predictor.streamlit.app/)
An end‑to‑end machine learning system that forecasts the Air Quality Index (AQI) for Karachi, Pakistan, at 6‑hour intervals for the next three days. The application fetches real‑time weather and pollution data, engineers features, trains three regression models (Ridge, Random Forest, Gradient Boosting), and displays forecasts through an interactive Streamlit dashboard.


Features

Live Data – Fetches hourly weather (Open‑Meteo) and pollutant data (OpenWeather) for Karachi.
Feature Engineering – Computes 6‑hourly aggregates, rolling windows, lags, and non‑linear interactions.
Multiple Models – Trains and compares Ridge Regression, Random Forest, and Gradient Boosting.
3‑Day Forecast – Generates 6‑hourly AQI predictions using a recursive multi‑step strategy.
Interactive Dashboard – Built with Streamlit + Altair; shows current AQI, trends, model performance, and forecast charts.
Walk‑Forward Backtest – Simulates real‑world performance by iteratively retraining the model.

Prerequisites
Python 3.9 or later
OpenWeather API key (free tier)
Internet connection (to fetch live data)

Installation
Clone the repository

bash
git clone https://github.com/yourusername/karachi-aqi-predictor.git
cd karachi-aqi-predictor
Create and activate a virtual environment

bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # Linux/Mac
Install dependencies

bash
pip install -r requirements.txt
Set up environment variables
Copy .env.example to .env and fill in your OpenWeather API key:

text
OPENWEATHER_KEY=your_api_key_here
LAT=24.8607
LON=67.0011
(The latitude and longitude are set for Karachi – change if needed.)

Usage
Run the Streamlit app from the project root:

bash
streamlit run src/app.py
The dashboard will open in your default browser. Use the sidebar controls to refresh data.


Project Structure
text
karachi-aqi-predictor/
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
├── src/
│   ├── app.py                   # Streamlit dashboard (main entry)
│   ├── config.py                # Loads environment variables
│   ├── fetch_weather.py         # Open‑Meteo weather data
│   ├── fetch_pollution.py       # OpenWeather pollution data
│   ├── features.py              # Feature engineering functions
│   ├── forecast.py              # 3‑day forecasting logic
│   ├── train.py                 # Model training & evaluation
│   ├── hopsworks_client.py      # Hopsworks integration
│   ├── store_features.py        # Write to Hopsworks
│   └── feature_pipeline.py      # Hourly feature pipeline
└── README.md
How It Works
Data Acquisition
Weather history: fetch_weather_history() calls Open‑Meteo Archive API.
Weather forecast: get_weather_forecast() calls Open‑Meteo Forecast API.
Pollution history: fetch_pollution_history() calls OpenWeather Air Pollution History API.
Current pollution: included in the same call.
Feature Engineering
Raw data is joined on a Unix timestamp (ts_key).
Resampled to 6‑hour intervals (mean of numeric columns).
Added: rolling 24h means, 1‑step and 24‑step lags for AQI and PM₂.₅, plus hour_sin, hour_cos, temp_humidity, wind_sq.
Model Training
Data is split chronologically (80/20).
Three models are trained: Ridge (with StandardScaler), Random Forest (200 trees), Gradient Boosting (200 estimators).

The best model (lowest RMSE) is selected for forecasting.

Forecasting

Weather forecast for the next 3 days is fetched and resampled to 6‑hour intervals.

Pollutant levels are estimated using historical patterns and weather influence.

Features are built recursively using a buffer of past values.

All three models predict AQI; the best model’s output is used to update the buffer for subsequent steps.

Dashboard

Displays current AQI, temperature, health advisory.

Shows 24‑hour trend, historical charts, model comparison, and 3‑day forecast.

Includes a timeline table of last 3 days history + next 3 days forecast.


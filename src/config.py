import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqipredictor")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", "443"))

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))

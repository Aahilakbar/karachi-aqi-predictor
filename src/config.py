import os
import streamlit as st

# Try to get from Streamlit secrets first (cloud), fall back to .env (local)
try:
    # When running on Streamlit Cloud
    OPENWEATHER_KEY = st.secrets["OPENWEATHER_KEY"]
    LAT = st.secrets["LAT"]
    LON = st.secrets["LON"]
except:
    # When running locally with .env file
    from dotenv import load_dotenv
    load_dotenv()
    OPENWEATHER_KEY = os.getenv('OPENWEATHER_KEY')
    LAT = os.getenv('LAT')
    LON = os.getenv('LON')

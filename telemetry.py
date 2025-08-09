import fastf1 as ff
import pandas as pd
import logging
import streamlit as st

logging.getLogger("fastf1").setLevel(logging.ERROR)

@st.cache_data(show_spinner="Loading Race Data...")
def load_session_data(year, circuit, racetype):
    CACHE = "./fastf1_cache"
    ff.Cache.enable_cache(CACHE)
    
    session = ff.get_session(year, circuit, racetype)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    return session.laps , session.results, session.total_laps, session.weather_data

def get_driver_laps(all_laps, driver_code):
    driver_laps = all_laps.pick_drivers(driver_code)
    driver_laps = driver_laps[driver_laps['LapTime'].notnull()]
    return driver_laps

def get_driver_telemetry(all_laps, driver_code):
    driver_laps = all_laps.pick_driver(driver_code).pick_quicklaps()
    
    if driver_laps.empty:
        return pd.DataFrame()

    all_telemetry = []
    for _, lap in driver_laps.iterrows():
        try:
            telemetry_chunk = lap.get_car_data().add_distance()
            telemetry_chunk['LapNumber'] = lap['LapNumber']
            telemetry_chunk['Compound'] = lap['Compound']
            telemetry_chunk['Stint'] = lap['Stint']
            
            all_telemetry.append(telemetry_chunk)
        except Exception as e:
            print(f"Could not load telemetry for Lap {lap['LapNumber']}: {e}")
            continue

    if not all_telemetry:
        return pd.DataFrame()
    return pd.concat(all_telemetry)
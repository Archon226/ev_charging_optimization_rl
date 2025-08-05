import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def load_ev_metadata():
    return pd.read_csv(os.path.join(DATA_DIR, 'EV_Metadata.csv'))

def load_charging_curves():
    return pd.read_csv(os.path.join(DATA_DIR, 'EV_Charging_Curve_Data.csv'))

def load_station_metadata():
    return pd.read_csv(os.path.join(DATA_DIR, 'charging_station_metadata.csv'))

def load_connectors():
    return pd.read_csv(os.path.join(DATA_DIR, 'charging_station_connectors.csv'))

def load_pricing_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'Pricing_Dataset.csv'))

def load_time_sensitive_pricing():
    return pd.read_csv(os.path.join(DATA_DIR, 'Time_Sensitive_Pricing_Data.csv'))

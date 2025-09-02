import pandas as pd
from simulator.trip_simulator import TripSimulator

from utils.charging_curves import ChargingCurveLibrary
from utils.data_loader import (
    load_ev_metadata,
    load_station_metadata,
    load_connectors,
    load_pricing_data,
    load_charging_curves
)

def main():
    # === Load All Necessary Data ===
    print("Loading data...")
    user_df = pd.read_csv("data/simulated_users.csv")
    ev_metadata = load_ev_metadata()
    station_data = load_station_metadata()
    connector_data = load_connectors()
    pricing_data = load_pricing_data()
    charging_library = ChargingCurveLibrary(load_charging_curves())

    # === Initialize Simulator ===
    print("Initializing TripSimulator...")
    simulator = TripSimulator(
        user_df=user_df,
        station_df=station_data,
        connector_df=connector_data,
        pricing_df=pricing_data,
        ev_metadata_df=ev_metadata,
        charging_library=charging_library
    )

    # === Run All Simulated Trips ===
    print("\n=== Running Simulated Trips ===")
    simulator.run_all()

if __name__ == "__main__":
    main()

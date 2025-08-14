from simulator.trip_simulator import TripSimulator, TripPlan
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.pricing import load_pricing_catalog
from utils.data_loader import build_stations_merged_files  # NEW helper we just added
import pandas as pd
from datetime import datetime

# Load pricing catalog
catalog = load_pricing_catalog("data")

# Load merged stations (metadata + connectors)
stations_merged = build_stations_merged_files("data")

# Load EV metadata / charging curves
ev_meta = pd.read_csv("data/EV_Metadata.csv")
curves = pd.read_csv("data/EV_Charging_Curve_Data.csv")

# Create simulator (point to your SUMO network file)
sim = TripSimulator(
    ev_meta,
    curves,
    stations_merged,
    catalog,
    sumo_net=r"net\london_small.net.xml"  # <-- adjust path if needed
)

# Create a trip plan
plan = TripPlan(
    agent_id="U001",
    model=ev_meta.iloc[0]["model"],
    user_type="Payg",
    objective="hybrid",  # time, cost, or hybrid
    start_lat=51.5143,
    start_lon=-0.0889,
    dest_lat=51.5120,
    dest_lon=-0.0980,
    depart_dt=datetime.now(),
    init_soc_pct=35.0
)

# Run the trip
result = sim.run_single_trip(plan)

# Output result summary
print("Total cost:", result.total_cost)
print("Arrived:", result.arrived)
print("Number of charges:", len(result.charges))

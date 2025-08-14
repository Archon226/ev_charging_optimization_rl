import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulator.trip_simulator import TripSimulator, TripPlan
from utils.pricing import load_pricing_catalog
import pandas as pd
from datetime import datetime

DATA="data"; NET=r"london_boroughs.net.xml"
catalog = load_pricing_catalog(DATA)
ev_meta = pd.read_csv(f"{DATA}/EV_Metadata.csv")
curves  = pd.read_csv(f"{DATA}/EV_Charging_Curve_Data.csv")
stations_merged = pd.read_csv(f"{DATA}/stations_in_net.csv")  # filtered!

sim = TripSimulator(ev_meta, curves, stations_merged, catalog, sumo_net=NET)

plan = TripPlan(
  agent_id="U_forced", model=ev_meta.iloc[0]["model"], user_type="Payg", objective="hybrid",
  start_lat=51.5176, start_lon=-0.0824,   # Liverpool St
  dest_lat=51.4813, dest_lon=-0.1446,     # Battersea Power Station
  depart_dt=datetime(2025,8,12,18,0),
  init_soc_pct=8.0, reserve_soc_pct=12.0, target_soc_pct=80.0,
  max_detour_km=12.0, top_k_candidates=5, step_horizon_s=180
)

res = sim.run_single_trip(plan)
print("Arrived:", res.arrived, "| charges:", len(res.charges), "| £:", res.total_cost)
for ch in res.charges: print(ch)

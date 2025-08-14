from simulator.trip_simulator import TripSimulator, TripPlan
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.pricing import load_pricing_catalog
from utils.data_loader import build_stations_merged_files
import pandas as pd
from datetime import datetime

DATA = "data"
NET  = r"london_small.net.xml"  # use your larger net if available

# Load core data
catalog = load_pricing_catalog(DATA)
stations_merged = build_stations_merged_files(DATA)
ev_meta = pd.read_csv(f"{DATA}/EV_Metadata.csv")
curves  = pd.read_csv(f"{DATA}/EV_Charging_Curve_Data.csv")

# Pick an EV model that exists in your metadata
model = ev_meta.iloc[0]["model"]

# Farther apart points to force distance (swap to your net’s coverage)
A = (51.5176, -0.0824)   # Liverpool St
B = (51.4813, -0.1446)   # Battersea Power Station

# Low SOC to force charging
plan = TripPlan(
    agent_id="U_forced",
    model=model,
    user_type="Payg",
    objective="hybrid",
    start_lat=A[0], start_lon=A[1],
    dest_lat=B[0], dest_lon=B[1],
    depart_dt=datetime(2025, 8, 12, 18, 0),  # ~peak time to exercise pricing windows
    init_soc_pct=8.0,
    reserve_soc_pct=12.0,
    target_soc_pct=80.0,
    max_detour_km=12.0,          # widen to find candidates
    top_k_candidates=5,
    step_horizon_s=180
)

sim = TripSimulator(ev_meta, curves, stations_merged, catalog, sumo_net=NET)

print("Running trip… (this case should require a charge)")
res = sim.run_single_trip(plan)

print("\n=== Trip summary ===")
print("Arrived:", res.arrived)
print("Distance (km):", res.distance_km)
print("Drive min:", res.total_drive_min, "Charge min:", res.total_charge_min, "Wait min:", res.total_wait_min)
print("Total cost (£):", res.total_cost)
print("Charges:", len(res.charges))

for i, ch in enumerate(res.charges, 1):
    print(f"\n--- Charge #{i} ---")
    print("station_id:", ch.station_id, "| company_id:", ch.company_id)
    print("type/power:", ch.charger_type, f"{ch.rated_power_kw:.0f}kW")
    print("detour_km:", f"{ch.detour_km:.2f}", "wait_min:", f"{ch.wait_min:.1f}")
    print("delivered_kwh:", f"{ch.delivered_kwh:.2f}", "charge_min:", f"{ch.charge_min:.1f}")
    print("unit_price:", ch.unit_price, "source:", ch.unit_source)
    print("energy_cost:", ch.energy_cost, "total_cost:", ch.total_cost)

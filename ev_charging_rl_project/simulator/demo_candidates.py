import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sumo_adapter import SumoSim
from candidates import find_charger_candidates, EVSpecLite

# Paths (adjust if your data lives elsewhere)
NET = r"london_small.net.xml"
DATA = r"data"

stations = pd.read_csv(os.path.join(DATA, "charging_station_metadata.csv"))
connects = pd.read_csv(os.path.join(DATA, "charging_station_connectors.csv"))

# Example user: somewhere in the bbox, heading west
A = (51.5143, -0.0889)  # origin
B = (51.5120, -0.0980)  # destination

# Very rough EV spec (replace with your actual user/EV data)
ev = EVSpecLite(
    battery_kwh=64.0,
    eff_kwh_per_km=0.18,            # 18 kWh/100km
    allowed_connectors=("CCS2","Type2"),
    max_dc_kw=120.0,
    max_ac_kw=11.0,
)
current_soc = 0.35

sim = SumoSim(NET)
cands = find_charger_candidates(sim, A, B, stations, connects, ev,
                                current_soc=current_soc, top_k=5, max_detour_km=5.0)

print(f"Found {len(cands)} candidates:")
for c in cands:
    print(f"- {c.station_id} | {c.charger_type} {c.rated_power_kw:.0f}kW | "
          f"detour {c.detour_km:.2f}km / {c.detour_time_s/60:.1f}min | "
          f"reach_now={c.reachable_with_current_soc} | conn_ok={c.connector_ok}")

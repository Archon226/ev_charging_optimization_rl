# tests/check_net_coverage.py
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import os
from simulator.sumo_adapter import SumoSim
import sumolib  # comes from SUMO_HOME/tools added in your adapter
NET = r"london_boroughs.net.xml"   # your larger net
DATA = "data"

stations = pd.read_csv(os.path.join(DATA, "stations_merged.csv")) \
            if os.path.exists(os.path.join(DATA, "stations_merged.csv")) \
            else pd.read_csv(os.path.join(DATA, "charging_station_metadata.csv")).rename(
                 columns={"chargeDeviceID":"station_id"})

sim = SumoSim(NET)
net = sim.net

xmin,ymin,xmax,ymax = net.getBoundary()
def inside(lat, lon):
    x,y = net.convertLonLat2XY(lon, lat)
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

in_bounds = stations[stations.apply(lambda r: inside(r["latitude"], r["longitude"]), axis=1)]
pct = (len(in_bounds)/len(stations))*100 if len(stations)>0 else 0.0
print(f"In-bounds stations: {len(in_bounds)}/{len(stations)} ({pct:.1f}%)")

def snap_ok(lat, lon, r=120.0):
    x,y = net.convertLonLat2XY(lon, lat)
    return len(net.getNeighboringEdges(x,y,r=r))>0

snapped = in_bounds[in_bounds.apply(lambda r: snap_ok(r["latitude"], r["longitude"], 120.0), axis=1)]
pct2 = (len(snapped)/len(in_bounds))*100 if len(in_bounds)>0 else 0.0
print(f"Snap-to-road OK: {len(snapped)}/{len(in_bounds)} ({pct2:.1f}%) within 120 m")

# Save filtered list for your sim
out = os.path.join(DATA, "stations_in_net.csv")
snapped.to_csv(out, index=False)
print("Saved usable stations to:", out)

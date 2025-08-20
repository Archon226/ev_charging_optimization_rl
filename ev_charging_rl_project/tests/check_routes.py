# tools/check_routes.py (ad-hoc)
import pandas as pd
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulator.sumo_adapter import SumoSim

users = pd.read_csv("data/simulated_users.csv")
sim = SumoSim("london_inner.net.xml")  # or your actual net path

for tid in [0,1,2,3,4,10,25,42]:
    r = users.iloc[tid]
    route_km, route_s = sim.route_between_cached(r.start_lat, r.start_lon, r.dest_lat, r.dest_lon)
    def hav(lat1, lon1, lat2, lon2):
        import math
        R=6371.0088
        dlat=math.radians(lat2-lat1)
        dlon=math.radians(lon2-lon1)
        a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(a))
    hav_km = hav(r.start_lat, r.start_lon, r.dest_lat, r.dest_lon)
    print(f"trip_id {tid}: route={route_km:.2f} km, hav={hav_km:.2f} km, time={route_s/60:.1f} min")

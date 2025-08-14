from sumo_adapter import SumoSim

NET = r"london_small.net.xml"

# Two points within the bbox above
A = (51.5143, -0.0889)   # near Bank of England
B = (51.5120, -0.0980)   # west of Bank
POI = (51.5133, -0.0930) # pretend charger between them

sim = SumoSim(NET)

route = sim.route_between(A, B)
print(f"Route: edges={len(route.edges)}, dist={route.dist_km:.2f} km, time={route.time_s/60:.1f} min")

adv = sim.advance(route, start_idx=0, horizon_s=180)  # 3 minutes
print(f"Advance 3 min: end_idx={adv['end_idx']}, dist={adv['dist_km']:.2f} km, time={adv['time_s']:.1f} s")

det = sim.detour_metrics(A, B, POI)
print(f"Detour via POI: +{det['detour_dist_km']:.2f} km, +{det['detour_time_s']/60:.1f} min")
